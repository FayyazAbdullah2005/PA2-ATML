from __future__ import annotations

import argparse

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, chunked_response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import sample_entropy, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode


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
def evaluate_policy(bundle, name, adapter=None, limit=None, batch_size=4, max_new_tokens=None, results_dir=None):
    """Held-out generation protocol shared by every PPO condition (standard run and forks)."""

    cfg = bundle["cfg"]
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    rows = bundle["rows"][: int(limit)] if limit else bundle["rows"]
    prompts = [prompt_messages(r) for r in rows]
    max_new_tokens = int(max_new_tokens or cfg["eval_max_response_length"])
    gen = cfg.get("generation", {})
    results_dir = repo_path(results_dir or cfg["results_dir"])

    records = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        set_seed(int(cfg["seed"]))
        rollouts = batch_generate(
            policy,
            tokenizer,
            chunk,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_new_tokens,
            temperature=float(gen.get("temperature", 0.7)),
            top_p=float(gen.get("top_p", 0.9)),
            do_sample=bool(gen.get("do_sample", True)),
        )
        rewards = score_reward_pairs(
            bundle["reward"][0],
            bundle["reward"][1],
            chunk,
            rollouts["responses"],
            max_length=int(cfg["reward_max_length"]),
        )
        pol_logp = chunked_response_token_logprobs(
            policy, rollouts["sequences"], rollouts["attention_mask"], rollouts["prompt_width"]
        )
        with reference_mode(policy):
            ref_logp = chunked_response_token_logprobs(
                policy, rollouts["sequences"], rollouts["attention_mask"], rollouts["prompt_width"]
            )
        mask = rollouts["response_mask"]
        eos_penalty = float(cfg["missing_eos_penalty"])
        for j in range(len(chunk)):
            records.append(
                {
                    "condition": name,
                    "prompt": [m["content"] for m in chunk[j]][-1],
                    "response": rollouts["responses"][j],
                    "reward_model_score": float(rewards[j]),
                    "reward_model_score_penalized": float(
                        rewards[j] - (0.0 if rollouts["terminated_with_eos"][j] else eos_penalty)
                    ),
                    "response_tokens": int(rollouts["response_lengths"][j]),
                    "terminated_with_eos": bool(rollouts["terminated_with_eos"][j]),
                    "truncated": bool(rollouts["truncated"][j]),
                    "kl_token_mean": float(sampled_kl(pol_logp[j : j + 1], ref_logp[j : j + 1], mask[j : j + 1]).item()),
                    "entropy_token_mean": float(sample_entropy(pol_logp[j : j + 1], mask[j : j + 1]).item()),
                }
            )

    write_jsonl(results_dir / "samples" / f"{name}_generations.jsonl", records)
    reward_mean = sum(r["reward_model_score"] for r in records) / max(1, len(records))
    penalized = sum(r["reward_model_score_penalized"] for r in records) / max(1, len(records))
    # `response_tokens` is already one count per response, so average the list directly.
    # `mean_response_length` expects a [batch, seq] mask and sums within each row; handing it a
    # flat vector of counts would add every response together instead of averaging them.
    lengths = [r["response_tokens"] for r in records]
    result = {
        "condition": name,
        "adapter": adapter or bundle.get("adapter"),
        "num_prompts": len(records),
        "eval_max_response_length": max_new_tokens,
        "reward_model_mean": reward_mean,
        "reward_model_mean_penalized": penalized,
        "kl_token_mean": sum(r["kl_token_mean"] for r in records) / max(1, len(records)),
        "entropy_token_mean": sum(r["entropy_token_mean"] for r in records) / max(1, len(records)),
        "response_length_mean": float(sum(lengths) / max(1, len(lengths))),
        "response_length_max": max(lengths) if lengths else 0,
        "eos_termination_rate": sum(r["terminated_with_eos"] for r in records) / max(1, len(records)),
        "truncation_rate": sum(r["truncated"] for r in records) / max(1, len(records)),
        "samples_file": str((results_dir / "samples" / f"{name}_generations.jsonl").relative_to(repo_path("."))),
    }
    save_json(results_dir / f"{name}_heldout_eval.json", result)
    print(
        f"[{name}] n={result['num_prompts']} reward={reward_mean:.3f} kl={result['kl_token_mean']:.4f} "
        f"len={result['response_length_mean']:.1f} eos={result['eos_termination_rate']:.3f}"
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    bundle = load_evaluation_bundle(args.config, args.adapter)
    bundle["adapter"] = args.adapter
    evaluate_policy(bundle, args.name, adapter=args.adapter, limit=args.limit, batch_size=args.batch_size)
    bundle.clear()
    clear_gpu()


if __name__ == "__main__":
    main()
