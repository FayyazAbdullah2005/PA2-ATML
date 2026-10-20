from __future__ import annotations

import argparse

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task3_grpo.continue_train import response_token_logprobs_chunked


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def evaluate_policy(bundle, name, adapter=None, limit=None, batch_size=1, max_new_tokens=None, results_dir=None):
    """Held-out protocol shared by every GRPO condition (standard run and normalization forks).

    `batch_size` counts *prompts*; each prompt is sampled `num_generations` times so the
    held-out set reports the same group-informativeness diagnostics as the training loop.
    """

    cfg = bundle["cfg"]
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    rows = bundle["rows"][: int(limit)] if limit else bundle["rows"]
    results_dir = repo_path(results_dir or cfg["results_dir"])
    K = int(cfg["num_generations"])
    max_new_tokens = int(max_new_tokens or cfg.get("eval_max_response_length", cfg["max_completion_length"]))
    gen = cfg.get("generation", {})

    records = []
    group_stds = []
    for i in range(0, len(rows), batch_size):
        prompt_batch = [prompt_messages(r) for r in rows[i : i + batch_size]]
        expanded = [p for p in prompt_batch for _ in range(K)]
        set_seed(int(cfg["seed"]))
        rollouts = batch_generate(
            policy,
            tokenizer,
            expanded,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_new_tokens,
            temperature=float(gen.get("temperature", 0.7)),
            top_p=float(gen.get("top_p", 0.9)),
            do_sample=bool(gen.get("do_sample", True)),
        )
        rewards = score_reward_pairs(
            bundle["reward"][0], bundle["reward"][1], expanded, rollouts["responses"],
            max_length=int(cfg.get("reward_max_length", max_new_tokens + int(cfg["max_prompt_length"]))),
        )
        pol_logp = response_token_logprobs_chunked(
            policy, rollouts["sequences"], rollouts["attention_mask"], rollouts["prompt_width"], with_grad=False
        )
        with reference_mode(policy):
            ref_logp = response_token_logprobs_chunked(
                policy, rollouts["sequences"], rollouts["attention_mask"], rollouts["prompt_width"], with_grad=False
            )
        mask = rollouts["response_mask"]
        for j in range(len(expanded)):
            records.append(
                {
                    "condition": name,
                    "prompt_index": i + j // K,
                    "completion_index": j % K,
                    "prompt": [m["content"] for m in expanded[j]][-1],
                    "response": rollouts["responses"][j],
                    "reward_model_score": float(rewards[j]),
                    "response_tokens": int(rollouts["response_lengths"][j]),
                    "terminated_with_eos": bool(rollouts["terminated_with_eos"][j]),
                    "truncated": bool(rollouts["truncated"][j]),
                    "kl_token_mean": float(sampled_kl(pol_logp[j : j + 1], ref_logp[j : j + 1], mask[j : j + 1]).item()),
                    "entropy_token_mean": float(sample_entropy(pol_logp[j : j + 1], mask[j : j + 1]).item()),
                }
            )
        for g in range(len(prompt_batch)):
            chunk = rewards[g * K : (g + 1) * K]
            group_stds.append(float(chunk.detach().float().std(unbiased=False).item()))

    write_jsonl(results_dir / "samples" / f"{name}_generations.jsonl", records)
    lengths = [r["response_tokens"] for r in records]
    informative = sum(1 for s in group_stds if s > 1e-6) / max(1, len(group_stds))
    result = {
        "condition": name,
        "adapter": adapter or bundle.get("adapter"),
        "num_prompts": len(rows),
        "num_generations": K,
        "eval_max_response_length": max_new_tokens,
        "reward_model_mean": sum(r["reward_model_score"] for r in records) / max(1, len(records)),
        "kl_token_mean": sum(r["kl_token_mean"] for r in records) / max(1, len(records)),
        "entropy_token_mean": sum(r["entropy_token_mean"] for r in records) / max(1, len(records)),
        "response_length_mean": float(sum(lengths) / max(1, len(lengths))),
        "response_length_max": max(lengths) if lengths else 0,
        "eos_termination_rate": sum(r["terminated_with_eos"] for r in records) / max(1, len(records)),
        "truncation_rate": sum(r["truncated"] for r in records) / max(1, len(records)),
        "informative_group_fraction": informative,
        "uninformative_group_fraction": 1.0 - informative,
        "within_group_reward_std_mean": sum(group_stds) / max(1, len(group_stds)),
        "samples_file": str((results_dir / "samples" / f"{name}_generations.jsonl").relative_to(repo_path("."))),
    }
    save_json(results_dir / f"{name}_heldout_eval.json", result)
    print(
        f"[{name}] n={result['num_prompts']} reward={result['reward_model_mean']:.3f} "
        f"kl={result['kl_token_mean']:.4f} len={result['response_length_mean']:.1f} "
        f"informative={informative:.2f}"
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    bundle = load_evaluation_bundle(args.config, args.adapter)
    bundle["adapter"] = args.adapter
    evaluate_policy(bundle, args.name, adapter=args.adapter, limit=args.limit, batch_size=args.batch_size)
    bundle.clear()
    clear_gpu()


if __name__ == "__main__":
    main()
