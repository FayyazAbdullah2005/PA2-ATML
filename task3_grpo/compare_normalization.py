from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_policy, load_evaluation_bundle

LENGTH_GRADIENT_KEYS = (
    "gradient_share_longest",
    "gradient_share_shortest",
    "gradient_share_ratio_longest_over_shortest",
    "spearman_length_vs_gradient_share",
)


def summarize(summary, heldout):
    log = summary["updates_log"]
    tail = log[-max(1, len(log) // 3) :] if log else []

    def avg(chunk, key):
        # Undefined ratios are logged as None (e.g. a zero-advantage shortest completion);
        # skip them rather than raising on the average.
        values = [r[key] for r in chunk if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "condition": summary["run_name"],
        "loss_type": summary["loss_type"],
        "adapter": summary["adapter"],
        "fork_updates": summary["updates"],
        "heldout": heldout,
        "length_gradient": {
            key: {
                "first_third": avg(log[: max(1, len(log) // 3)], key),
                "last_third": avg(tail, key),
            }
            for key in LENGTH_GRADIENT_KEYS
            if log and key in log[0]
        },
        "trajectory": {
            "reward_first_third": avg(log[: max(1, len(log) // 3)], "reward_mean"),
            "reward_last_third": avg(tail, "reward_mean"),
            "kl_first_third": avg(log[: max(1, len(log) // 3)], "kl_token_mean"),
            "kl_last_third": avg(tail, "kl_token_mean"),
            "length_first_third": avg(log[: max(1, len(log) // 3)], "response_length_mean"),
            "length_last_third": avg(tail, "response_length_mean"),
            "informative_group_fraction_last_third": avg(tail, "informative_group_fraction"),
        },
        "wall_time_s": summary["wall_time_s"],
        "peak_vram_gb": summary["peak_vram_gb"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--fork-updates", type=int)
    ap.add_argument("--eval-limit", type=int)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    fork_updates = int(args.fork_updates or cfg["fork_updates"])

    rows = []
    results_dir = repo_path(cfg["results_dir"])
    for loss_type in ("grpo", "dr_grpo"):
        tag = loss_type
        adapter = f"outputs/task3_grpo/{tag}"
        fork_path = results_dir / f"{tag}_fork.json"
        existing = load_json(fork_path) if fork_path.exists() else None
        if existing and existing.get("heldout", {}).get("num_prompts") == 200:
            print(f"[skip] {tag}: reusing completed full-set fork", flush=True)
            rows.append(existing)
            continue
        summary_path = results_dir / f"{tag}_summary.json"
        if summary_path.exists():
            print(f"[reuse] {tag}: training is complete; re-running full held-out evaluation", flush=True)
            summary = load_json(summary_path)
        else:
            print(f"=== loss_type={loss_type}: {fork_updates}-update continuation fork ===", flush=True)
            summary = run_grpo(
                args.config,
                output=adapter,
                updates=fork_updates,
                loss_type=loss_type,
                run_name=tag,
            )
        bundle = load_evaluation_bundle(args.config, adapter)
        bundle["adapter"] = adapter
        heldout = evaluate_policy(bundle, tag, adapter=adapter, limit=args.eval_limit)
        bundle.clear()
        clear_gpu()
        fork = summarize(summary, heldout)
        # Persist each fork immediately: a crash on the second arm must not discard the first.
        save_json(fork_path, fork)
        save_json(results_dir / "normalization_partial.json", {"conditions": rows + [fork]})
        rows.append(fork)
        print(f"[done] {tag}: heldout_reward={heldout['reward_model_mean']:.3f} "
              f"len={heldout['response_length_mean']:.1f}", flush=True)

    delta = {}
    if len(rows) == 2:
        canonical, dr = rows[0]["heldout"], rows[1]["heldout"]
        for key in ("reward_model_mean", "kl_token_mean", "entropy_token_mean", "response_length_mean",
                    "informative_group_fraction", "within_group_reward_std_mean"):
            delta[key] = {
                "canonical_grpo": canonical.get(key),
                "dr_grpo": dr.get(key),
                "delta": (dr.get(key) or 0.0) - (canonical.get(key) or 0.0),
            }
        for key in LENGTH_GRADIENT_KEYS:
            a = rows[0]["length_gradient"].get(key, {})
            b = rows[1]["length_gradient"].get(key, {})
            if a and b:
                delta[f"gradient_{key}"] = {
                    "canonical_grpo": a.get("last_third"),
                    "dr_grpo": b.get("last_third"),
                    "delta": (b.get("last_third") or 0.0) - (a.get("last_third") or 0.0),
                }

    payload = {
        "description": (
            "Task 3 Step 3: length-normalization study. Both forks start from the identical supplied GRPO "
            "midpoint, use the same prompts in the same order, the same per-update seed, the same reward, "
            "beta, epsilon and generated-token budget; only the sequence normalization changes. The reward "
            "function is unchanged, so any shift in gradient allocation or response length is attributable "
            "to the normalization."
        ),
        "length_gradient_definition": (
            "Per-completion gradient magnitude |w_k * sum_t objective_k,t| with w_k = 1/T_k for canonical "
            "GRPO and w_k = 1/T_max for Dr. GRPO; 'share' is that magnitude divided by the total over the "
            "group, so it is exactly the fraction of the update gradient each completion receives."
        ),
        "fork_updates": fork_updates,
        "conditions": rows,
        "delta": delta,
    }
    out = repo_path(cfg["results_dir"]) / "normalization_study.json"
    save_json(out, payload)
    print(f"saved -> {out}")
    for r in rows:
        print(
            f"{r['loss_type']:8s} heldout_reward={r['heldout']['reward_model_mean']:.3f} "
            f"kl={r['heldout']['kl_token_mean']:.4f} len={r['heldout']['response_length_mean']:.1f} "
            f"grad_share_longest={r['length_gradient'].get('gradient_share_longest', {}).get('last_third', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
