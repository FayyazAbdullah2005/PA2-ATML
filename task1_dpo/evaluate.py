from __future__ import annotations

import argparse

import torch

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from task1_dpo import protocol


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    cache = {}

    def reward():
        """Load the frozen reward model on first use.

        It costs 1.8 GB of the 8 GB card and is only needed after generation, so keeping it
        out of the bundle leaves room for the pair scoring and the log-probability passes.
        """

        if "model" not in cache:
            cache["model"] = load_reward_model(cfg)
        return cache["model"]

    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": reward,
    }


def run_evaluation(
    bundle,
    name: str,
    rows=None,
    beta=None,
    eval_prompt_limit=None,
    gen_batch_size=4,
    results_dir=None,
    sample_dir=None,
    qualitative_examples=8,
    max_new_tokens=None,
    free_policy=True,
):
    cfg = bundle["cfg"]
    rows = cfg["paths"]["dpo_standard_eval"] if rows is None else rows
    rows = read_jsonl(rows) if isinstance(rows, str) else list(rows)
    results_dir = repo_path(results_dir or cfg["results_dir"])

    pair_rows = rows
    pair_rows, dropped = protocol.filter_fits(bundle["tokenizer"], pair_rows, int(cfg["max_sequence_length"]))
    if dropped:
        print(f"dropped {dropped} held-out pairs whose prompt exceeds max_length={cfg['max_sequence_length']}")
    if eval_prompt_limit:
        pair_rows = pair_rows[: int(eval_prompt_limit)]
    prompts = [prompt_messages_from_preference(r) for r in pair_rows]

    pair_metrics = protocol.evaluate_preference_pairs(bundle["policy"], bundle["tokenizer"], pair_rows, cfg, beta=beta)

    rollouts = protocol.generate_with_diagnostics(
        bundle["policy"],
        bundle["tokenizer"],
        prompts,
        cfg,
        batch_size=gen_batch_size,
        max_new_tokens=max_new_tokens,
    )
    reference = protocol.aggregate_diagnostics(rollouts)
    # The policy is not needed again: free its 3 GB before the frozen int8 reward model is
    # loaded and run, otherwise bitsandbytes' int8 workspace pushes the 8 GB card over.
    if free_policy and torch.cuda.is_available():
        bundle["policy"].to("cpu")
        clear_gpu()
    rewards = protocol.reward_scores(*bundle["reward"](), prompts, rollouts["responses"])
    lengths = protocol.length_stats(rollouts["response_lengths"])

    samples = [
        {
            "prompt": [m["content"] for m in p][-1],
            "response": r,
            "reward_model_score": s,
            "response_tokens": int(n),
            "terminated_with_eos": bool(t),
            "truncated": bool(tr),
        }
        for p, r, s, n, t, tr in zip(
            prompts,
            rollouts["responses"],
            rewards,
            rollouts["response_lengths"],
            rollouts["terminated_with_eos"],
            rollouts["truncated"],
        )
    ]
    sample_path = repo_path(sample_dir or (results_dir / "samples")) / f"{name}_generations.jsonl"
    write_jsonl(sample_path, samples)

    reward_mean, reward_std = protocol.mean_std(rewards)

    result = {
        "name": name,
        "adapter": str(bundle.get("adapter", "")),
        "beta": pair_metrics["beta"],
        "heldout_pairs": pair_metrics,
        "heldout_pairs_dropped_too_long": dropped,
        "reference_offsets": reference,
        "generated": {
            "reward_model_mean": reward_mean,
            "reward_model_std": reward_std,
            "length": lengths,
            "eos_termination_rate": float(sum(rollouts["terminated_with_eos"]) / max(1, len(rollouts["terminated_with_eos"]))),
            "truncation_rate": float(sum(rollouts["truncated"]) / max(1, len(rollouts["truncated"]))),
            "num_prompts": len(prompts),
            "samples_file": str(sample_path.relative_to(repo_path("."))),
        },
    }
    save_json(results_dir / f"{name}_eval.json", result)
    print(
        f"[{name}] pairs={pair_metrics['num_pairs']} "
        f"acc={pair_metrics['preference_accuracy']:.4f} "
        f"loss={pair_metrics['dpo_loss']:.4f} "
        f"kl={reference['kl_token_mean']:.4f} "
        f"reward={result['generated']['reward_model_mean']:.4f} "
        f"len={lengths['mean']:.1f}"
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--eval-set", help="override the held-out pair file")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--eval-prompt-limit", type=int)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int)
    args = ap.parse_args()

    bundle = load_evaluation_bundle(args.config, args.adapter)
    bundle["adapter"] = args.adapter
    run_evaluation(
        bundle,
        args.name,
        rows=args.eval_set,
        beta=args.beta,
        eval_prompt_limit=args.eval_prompt_limit,
        gen_batch_size=args.gen_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    clear_gpu(bundle)
    bundle.clear()


if __name__ == "__main__":
    main()