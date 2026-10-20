from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task1_dpo import evaluate as dpo_eval
from task1_dpo.train import run_training


def _row(result, name, beta, adapter, train_summary):
    pairs = result["heldout_pairs"]
    gen = result["generated"]
    ref = result["reference_offsets"]
    return {
        "condition": name,
        "beta": beta,
        "adapter": adapter,
        "train_examples": train_summary["num_examples"],
        "train_epochs": train_summary["epochs"],
        "optimizer_steps": train_summary["optimizer_steps"],
        "heldout_pairs_scored": pairs["num_pairs"],
        "heldout_pairs_dropped_too_long": pairs.get("num_dropped_too_long", 0),
        "heldout_preference_accuracy": pairs["preference_accuracy"],
        "heldout_preference_accuracy_stderr": pairs["preference_accuracy_stderr"],
        "heldout_dpo_loss": pairs["dpo_loss"],
        "heldout_implied_reward_margin": pairs["implied_reward_margin_mean"],
        "kl_token_mean": ref["kl_token_mean"],
        "entropy_token_mean": ref["entropy_token_mean"],
        "reward_model_mean": gen["reward_model_mean"],
        "reward_model_std": gen["reward_model_std"],
        "response_length_mean": gen["length"]["mean"],
        "response_length_std": gen["length"]["std"],
        "response_length_iqr": gen["length"]["iqr"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="*")
    ap.add_argument("--max-examples", type=int, help="override short_ablation_examples")
    ap.add_argument("--eval-prompt-limit", type=int)
    ap.add_argument("--output-root", default="outputs/task1_dpo")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    betas = args.betas if args.betas else list(cfg["betas"])
    short_examples = int(args.max_examples or cfg["short_ablation_examples"])
    eval_rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    if args.eval_prompt_limit:
        eval_rows = eval_rows[: args.eval_prompt_limit]

    rows = []
    for beta in betas:
        tag = f"beta_{beta:g}"
        adapter = f"{args.output_root}/{tag}"
        eval_path = repo_path(cfg["results_dir"]) / f"{tag}_eval.json"
        summary_path = repo_path(cfg["results_dir"]) / f"{tag}_train_summary.json"

        # Reuse completed training, but never reuse an evaluation that was run on a
        # shortened subset.  The manual requires the fixed held-out set for every
        # beta condition, while an earlier recovery run used only 40 prompts.
        train_summary = load_json(summary_path) if summary_path.exists() else None
        result = load_json(eval_path) if eval_path.exists() else None
        evaluation_is_full = bool(
            result
            and result.get("generated", {}).get("num_prompts", 0) >= 200
            and result.get("heldout_pairs", {}).get("num_pairs", 0) >= 200
        )
        if train_summary is None:
            print(f"=== beta={beta} : training short fork ({short_examples} examples) ===", flush=True)
            train_summary = run_training(
                args.config,
                run_name=tag,
                beta=beta,
                max_examples=short_examples,
                output_path=adapter,
            )
        if not evaluation_is_full:
            print(f"=== beta={beta} : evaluating on the common held-out set ===", flush=True)
            bundle = dpo_eval.load_evaluation_bundle(args.config, adapter)
            bundle["adapter"] = adapter
            result = dpo_eval.run_evaluation(
                bundle,
                tag,
                rows=eval_rows,
                beta=beta,
            )
            bundle.clear()
            clear_gpu()
        else:
            print(f"[skip] {tag}: reusing completed full-set evaluation", flush=True)

        rows.append(_row(result, tag, float(beta), adapter, train_summary))
        # Persist after every condition so the aggregate is always complete-so-far.
        save_json(repo_path(cfg["results_dir"]) / "beta_ablation_partial.json", {"results": rows})
        print(f"[done] {tag}: acc={rows[-1]['heldout_preference_accuracy']:.4f} "
              f"+/-{rows[-1]['heldout_preference_accuracy_stderr']:.4f} "
              f"pairs={rows[-1]['heldout_pairs_scored']}", flush=True)

    payload = {
        "description": (
            "Task 1 Step 2: DPO regularization-strength study. Every fork starts from the same "
            "Qwen2.5-1.5B-Instruct LoRA initialization and shares the data subset, optimizer, "
            "seed, LoRA configuration, held-out pairs and decoding settings; only beta changes."
        ),
        "budget_note": (
            f"These short forks use {short_examples} examples and differ from the standard "
            "one-epoch DPO run in Task 1 Step 1, which uses the full released training set."
        ),
        "short_ablation_examples": short_examples,
        "heldout_eval_pairs_requested": len(eval_rows),
        "heldout_eval_note": (
            "heldout_pairs_scored is the number actually evaluated; a pair is dropped when its prompt alone "
            "exceeds max_sequence_length."
        ),
        "beta": cfg["beta"],
        "learning_rate": cfg["learning_rate"],
        "batch_size": cfg["batch_size"],
        "grad_accum_steps": cfg["grad_accum_steps"],
        "seed": cfg["seed"],
        "results": rows,
    }
    out = repo_path(cfg["results_dir"]) / "beta_ablation.json"
    save_json(out, payload)
    print(f"saved -> {out}")
    for r in rows:
        print(
            f"beta={r['beta']:<5} acc={r['heldout_preference_accuracy']:.4f} "
            f"loss={r['heldout_dpo_loss']:.4f} kl={r['kl_token_mean']:.4f} "
            f"reward={r['reward_model_mean']:.3f} len={r['response_length_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
