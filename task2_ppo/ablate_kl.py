from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_policy, load_evaluation_bundle


def trajectory_deltas(log_path):
    """First-vs-last third mean for the logged trajectories (a stability/overoptimisation view)."""

    rows = log_path
    n = len(rows)
    if not n:
        return {}
    head = rows[: max(1, n // 3)]
    tail = rows[-max(1, n // 3) :]

    def avg(chunk, key):
        return sum(r[key] for r in chunk) / len(chunk)

    keys = ["learned_reward_mean", "kl_token_mean", "entropy_token_mean", "response_length_mean", "clip_fraction", "policy_loss", "value_loss"]
    return {k: {"first_third": avg(head, k), "last_third": avg(tail, k), "delta": avg(tail, k) - avg(head, k)} for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--betas", type=float, nargs="*")
    ap.add_argument("--fork-updates", type=int)
    ap.add_argument("--eval-limit", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    betas = args.betas if args.betas else list(cfg["kl_values"])
    fork_updates = int(args.fork_updates or cfg["fork_updates"])
    results_dir = repo_path(cfg["results_dir"])

    rows = []
    for beta in betas:
        tag = f"kl_{beta:g}"
        adapter = f"outputs/task2_ppo/{tag}"
        fork_path = results_dir / f"{tag}_fork.json"
        existing = load_json(fork_path) if fork_path.exists() else None
        if existing and existing.get("heldout", {}).get("num_prompts") == 200:
            print(f"[skip] {tag}: reusing completed full-set fork", flush=True)
            rows.append(existing)
            continue

        summary_path = results_dir / f"{tag}_summary.json"
        if summary_path.exists():
            # A previous attempt reached the end of training but failed during
            # evaluation.  Its adapter is a valid matched fork; do not retrain it.
            print(f"[reuse] {tag}: training is complete; re-running full held-out evaluation", flush=True)
            summary = load_json(summary_path)
        else:
            print(f"=== beta_KL={beta}: {fork_updates}-update continuation fork ===", flush=True)
            summary = run_ppo(
                args.config,
                output=adapter,
                updates=fork_updates,
                kl_beta=beta,
                run_name=tag,
                resume=args.resume,
            )
        bundle = load_evaluation_bundle(args.config, adapter)
        bundle["adapter"] = adapter
        heldout = evaluate_policy(bundle, tag, adapter=adapter, limit=args.eval_limit)
        bundle.clear()
        clear_gpu()

        deltas = trajectory_deltas(summary["updates_log"])
        fork = {
            "condition": tag,
            "kl_beta": float(beta),
            "adapter": adapter,
            "fork_updates": fork_updates,
            "heldout": heldout,
            "trajectory": deltas,
            "final_update": summary["updates_log"][-1] if summary["updates_log"] else None,
            "wall_time_s": summary["wall_time_s"],
            "peak_vram_gb": summary["peak_vram_gb"],
        }
        # Persist each fork immediately: a later crash must not discard completed conditions.
        save_json(fork_path, fork)
        save_json(results_dir / "kl_ablation_partial.json", {"results": rows + [fork]})
        rows.append(fork)
        print(f"[done] {tag}: heldout_reward={heldout['reward_model_mean']:.3f} "
              f"kl={heldout['kl_token_mean']:.4f}", flush=True)

    payload = {
        "description": (
            "Task 2 Step 3: reward-overoptimisation / KL-pressure study. Each fork restarts from the same "
            "supplied PPO midpoint, uses the same prompts in the same order, the same seed per update, the "
            "same clip epsilon and the same update budget; only beta_KL changes."
        ),
        "kl_values": betas,
        "fork_updates": fork_updates,
        "clip_epsilon_held_fixed": float(cfg["clip_epsilon"]),
        "standard_continuation_reference": {
            "final_update": load_json(results_dir / "standard_summary.json")["updates_log"][-1]
            if (results_dir / "standard_summary.json").exists()
            else None
        },
        "results": rows,
    }
    out = results_dir / "kl_ablation.json"
    save_json(out, payload)
    print(f"saved -> {out}")
    for r in rows:
        print(
            f"beta_KL={r['kl_beta']:<5} heldout_reward={r['heldout']['reward_model_mean']:.3f} "
            f"kl={r['heldout']['kl_token_mean']:.4f} entropy={r['heldout']['entropy_token_mean']:.4f} "
            f"len={r['heldout']['response_length_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
