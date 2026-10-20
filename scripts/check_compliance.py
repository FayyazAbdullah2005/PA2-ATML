"""Verify that every "Required Evidence" item in the assignment manual is present on disk.

Usage:
    python -m scripts.check_compliance          # full matrix
    python -m scripts.check_compliance --json   # machine-readable

Each row cites the manual page/section it comes from. A row is PASS only when the required
file exists AND (where applicable) actually contains the named quantities.
"""

from __future__ import annotations

import argparse
import json

from common.data import repo_path

REPO = repo_path(".")


def load(path):
    p = REPO / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_lines(path):
    p = REPO / path
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def has(data, *keys):
    """True when every dotted key path is present and not None."""

    for key in keys:
        node = data
        for part in key.split("."):
            if isinstance(node, list):
                if not node or not isinstance(node[0], dict):
                    return False
                node = node[0]
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if node is None:
            return False
    return True


ROWS = []


def row(manual, task, item, path, keys=(), check=None):
    data = load(path) if path.endswith(".json") else None
    if data is not None:
        ok = has(data, *keys) if keys else True
        if check:
            ok = ok and check(data)
    elif path.endswith(".jsonl"):
        lines = load_lines(path)
        ok = check(lines) if check else bool(lines)
    else:
        # JSON file that does not exist yet: nothing to inspect.
        ok = False
    ROWS.append({"manual": manual, "task": task, "item": item, "file": path, "ok": bool(ok)})


def build():
    # ---------------- Task 1 ----------------
    row("p4 Task1 Step1", 1, "Standard DPO: loss, preference acc, KL, reward score, length",
        "results/task1_dpo/standard_eval.json",
        ("heldout_pairs.preference_accuracy", "heldout_pairs.dpo_loss",
         "reference_offsets.kl_token_mean", "generated.reward_model_mean", "generated.length.mean"))
    row("p4 Task1 Step2", 1, "Beta study: 3 conditions x acc/loss/KL/reward/length + budget note",
        "results/task1_dpo/beta_ablation.json",
        ("budget_note",),
        lambda d: len(d.get("results", [])) == 3 and all(
            has(r, "heldout_preference_accuracy", "heldout_dpo_loss", "kl_token_mean",
                "reward_model_mean", "response_length_mean", "response_length_iqr")
            for r in d["results"]))
    row("p5 Task1 Step3", 1, "Length strata: preferred-longer / matched / rejected-longer per condition",
        "results/task1_dpo/length_confounding.json",
        ("strata",),
        lambda d: len(d.get("strata", [])) == 3 and all(
            len(v.get("by_stratum", {})) == 3 for v in d.get("conditions", {}).values()))
    row("p5 Task1 Step3", 1, "Word-limit compliance on the common prompt set",
        "results/task1_dpo/length_confounding.json",
        ("conditions",), lambda d: all(
            v.get("word_limit_compliance") is not None for v in d.get("conditions", {}).values()))
    row("p5 Task1 qualitative", 1, "Qualitative examples (preference/reward vs quality; length/compliance)",
        "results/task1_dpo/qualitative_examples.json",
        ("preference_vs_quality", "length_or_compliance"))

    # ---------------- Task 2 ----------------
    row("p7 Task2 Step1", 2, "Trajectories: reward, KL, policy/value loss, entropy, clip frac, grad norm, length",
        "results/task2_ppo/standard_updates.jsonl",
        check=lambda rows: bool(rows) and all(
            all(k in r for k in ("learned_reward_mean", "kl_token_mean", "policy_loss", "value_loss",
                                 "entropy_token_mean", "clip_fraction", "policy_grad_norm",
                                 "response_length_mean")) for r in rows))
    row("p7 Task2 Step1", 2, "Peak VRAM and wall-clock for the standard continuation",
        "results/task2_ppo/standard_summary.json", ("peak_vram_gb", "wall_time_s"))
    row("p7 Task2 Step2", 2, "Clipping: 3 epsilons x cached clip/affected fraction",
        "results/task2_ppo/clipping_study.json",
        ("cache_geometry",), lambda d: len(d.get("cache_geometry", [])) == 3 and all(
            has(g, "affected_token_fraction", "clipped_surrogate_mean") for g in d["cache_geometry"]))
    row("p7 Task2 Step2", 2, "Clipping: matched forks x held-out reward/KL/length + a stability statistic",
        "results/task2_ppo/clipping_study.json",
        ("forks",), lambda d: len(d.get("forks", [])) == 3 and all(
            has(f, "heldout.reward_model_mean", "heldout.kl_token_mean",
                "heldout.response_length_mean", "stability") for f in d["forks"]))
    row("p7 Task2 Step3", 2, "KL pressure: 3 betas x held-out reward/KL/entropy/length",
        "results/task2_ppo/kl_ablation.json",
        ("results",), lambda d: len(d.get("results", [])) == 3 and all(
            has(r, "heldout.reward_model_mean", "heldout.kl_token_mean",
                "heldout.entropy_token_mean", "heldout.response_length_mean") for r in d["results"]))
    row("p7 Task2 qualitative", 2, "One case where reward and quality agree, one where they disagree",
        "results/task2_ppo/qualitative_examples.json", ("reward_quality_agree", "reward_quality_disagree"))

    # ---------------- Task 3 ----------------
    row("p9 Task3 Step1", 3, "Trajectories: reward, KL, group std, uninformative frac, loss, grad norm, entropy, length",
        "results/task3_grpo/standard_updates.jsonl",
        check=lambda rows: bool(rows) and all(
            all(k in r for k in ("reward_mean", "kl_token_mean", "within_group_reward_std_mean",
                                 "uninformative_group_fraction", "policy_loss", "gradient_norm",
                                 "response_length_mean")) for r in rows))
    row("p9 Task3 Step1", 3, "Peak VRAM and wall-clock for the standard continuation",
        "results/task3_grpo/standard_summary.json", ("peak_vram_gb", "wall_time_s"))
    row("p9 Task3 Step2", 3, "Equal-generation group study: K=2,4,8 informative rate, std, signal variance",
        "results/task3_grpo/group_size_study.json",
        ("group_sizes",), lambda d: sorted(d.get("group_sizes", [])) == [2, 4, 8] and all(
            has(d["results"][str(k)]["overall"], "informative_group_fraction",
                "within_group_reward_std_mean", "relative_signal_variance")
            for k in d["group_sizes"]))
    row("p9 Task3 Step2", 3, "At least two prompt-difficulty bins defined once",
        "results/task3_grpo/group_size_study.json",
        ("prompt_difficulty_rule",), lambda d: all(
            len(d["results"][str(k)].get("by_prompt_difficulty", {})) >= 2 for k in d["group_sizes"]))
    row("p9 Task3 Step3", 3, "Canonical vs Dr.GRPO: reward, KL, length + a length-conditioned gradient statistic",
        "results/task3_grpo/normalization_study.json",
        ("conditions", "length_gradient_definition"), lambda d: len(d.get("conditions", [])) == 2 and all(
            has(c, "heldout.reward_model_mean", "heldout.kl_token_mean",
                "heldout.response_length_mean", "length_gradient") for c in d["conditions"]))
    row("p9 Task3 qualitative", 3, "Qualitative examples for normalization / group informativeness",
        "results/task3_grpo/qualitative_examples.json", ("normalization", "group_informativeness"))

    # ---------------- Task 4 ----------------
    row("p11 Task4 Step2", 4, "Per policy: safe-answer, over-refusal, unsafe-compliance, justified-refusal, ambiguous, length",
        "results/task4_safety/safety_metrics.json",
        ("policies",), lambda d: len(d.get("policies", {})) >= 4 and all(
            has(v, "safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate",
                "unsafe_justified_refusal_rate", "ambiguous_rate", "mean_response_tokens")
            for v in d["policies"].values()))
    row("p11 Task4 Step2", 4, "Category-level judge-label distribution for all four policies",
        "results/task4_safety/safety_metrics.json",
        ("policies",), lambda d: all(v.get("by_category") for v in d.get("policies", {}).values()))
    row("p11 Task4 Step3", 4, "Manual-audit agreement, confusion breakdown, ambiguous rate",
        "results/task4_safety/safety_metrics.json",
        ("manual_audit.n_audited", "manual_audit.agreement", "manual_audit.cohen_kappa"),
        lambda d: (d.get("manual_audit") or {}).get("n_audited", 0) >= 30)
    row("p11 Task4 qualitative", 4, "Harmful compliance, justified refusal, exaggerated refusal examples",
        "results/task4_safety/safety_metrics.json",
        ("qualitative_examples.unsafe_compliance", "qualitative_examples.justified_refusal",
         "qualitative_examples.over_refusal"))

    # ---------------- Task 5 ----------------
    row("p13 Task5 Step1", 5, "In-domain: exact accuracy, pairwise, format compliance, length, verifier-judge agreement",
        "results/task5_feedback/gsm_eval.json",
        ("policies",), lambda d: len(d.get("policies", {})) >= 3 and all(
            has(v, "exact_accuracy", "format_compliance", "mean_response_tokens") for v in d["policies"].values())
        and has(d, "verifier_judge_agreement.agreement"))
    row("p13 Task5 Step2", 5, "Diagnostics: 5 variants x both mechanisms x better/tie/wrong + S_reason + S_outcome",
        "results/task5_feedback/perturbation_study.json",
        ("by_variant", "S_reason", "S_outcome"),
        lambda d: len(d.get("by_variant", {})) == 5 and all(
            has(v, "rlvr.better", "rlvr.tie", "rlvr.wrong", "rlaif.better", "rlaif.tie", "rlaif.wrong")
            for v in d["by_variant"].values()))
    row("p13 Task5 Step3", 5, "Transfer: accuracy, pairwise, length, drop from in-domain",
        "results/task5_feedback/transfer_eval.json",
        ("policies",), lambda d: len(d.get("policies", {})) >= 3)
    row("p14 Task5 synthesis", 5, "Feedback-source synthesis incl. coverage/noise/exploitability/compute",
        "results/task5_feedback/feedback_comparison.json",
        ("feedback_source_summary.coverage", "feedback_source_summary.noise",
         "feedback_source_summary.exploitability", "feedback_source_summary.compute"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    ROWS.clear()
    build()

    if args.json:
        print(json.dumps(ROWS, indent=2))
        return

    by_task = {}
    for r in ROWS:
        by_task.setdefault(r["task"], []).append(r)

    total_ok = 0
    for task, rows in sorted(by_task.items()):
        print(f"\n=== Task {task} " + "=" * 52)
        for r in rows:
            mark = "PASS" if r["ok"] else "MISSING"
            total_ok += r["ok"]
            print(f"  [{mark:7s}] {r['item'][:62]:64s} {r['file']}")
    print("\n" + "=" * 70)
    print(f"{total_ok}/{len(ROWS)} required-evidence items present")
    missing = [r for r in ROWS if not r["ok"]]
    if missing:
        print("\nOutstanding:")
        for r in missing:
            print(f"  - [Task {r['task']}] {r['item']}  -> {r['file']}")


if __name__ == "__main__":
    main()