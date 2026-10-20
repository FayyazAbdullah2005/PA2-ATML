from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

# (variant, baseline) controlled pairs, with the diagnostic question each pair tests.
CONTROLLED_PAIRS = {
    "corrupt_reasoning_correct_final": ("clean_correct", "reasoning_sensitivity"),
    "good_reasoning_wrong_final": ("clean_correct", "outcome_sensitivity"),
    "persuasive_filler_correct": ("clean_correct", "filler_susceptibility"),
    "gold_distractor_wrong_final": ("clean_correct", "distractor_robustness"),
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def problem_text(row):
    for key in ("question", "problem", "prompt"):
        if isinstance(row.get(key), str):
            return row[key]
    return str(row.get("problem_text", ""))


def mechanism_scores(verdict, better_is_variant):
    """Turn a mechanism's verdict on one pair into (better, tie, wrong)."""

    if verdict == "tie":
        return "tie"
    chosen_better = verdict == "variant" if better_is_variant else verdict == "baseline"
    return "better" if chosen_better else "wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--cache", default="results/task5_feedback/judge_cache.json")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    judge = PairwiseAIJudge(cfg, args.cache)

    by_problem_clean = {}
    per_variant = defaultdict(lambda: {"rlvr": defaultdict(int), "rlaif": defaultdict(int), "n": 0})
    examples = []
    for pid, variants in sorted(groups.items()):
        problem = problem_text(variants["clean_correct"])
        for variant, (baseline, question) in CONTROLLED_PAIRS.items():
            v, b = variants[variant], variants[baseline]
            gold = str(v.get("gold_final", b.get("gold_final", "")))

            # Exact verifier: binary reward, so it can only say better / tie / wrong.
            r_variant = exact_reward(v["response"], gold)
            r_baseline = exact_reward(b["response"], gold)
            if r_variant > r_baseline:
                v_verdict = "variant"
            elif r_variant < r_baseline:
                v_verdict = "baseline"
            else:
                v_verdict = "tie"

            # AI pairwise judge.
            pref = judge.compare(problem, v["response"], b["response"])
            j_verdict = {"A": "variant", "B": "baseline", "TIE": "tie"}[pref]

            by_problem_clean[pid] = variants["clean_correct"]
            entry = per_variant[variant]
            entry["n"] += 1
            entry["rlvr"][mechanism_scores(v_verdict, False)] += 1
            entry["rlaif"][mechanism_scores(j_verdict, False)] += 1
            examples.append(
                {
                    "problem_id": pid,
                    "variant_type": variant,
                    "baseline_type": baseline,
                    "diagnostic_question": question,
                    "gold_final": gold,
                    "exact_reward_variant": r_variant,
                    "exact_reward_baseline": r_baseline,
                    "verifier_verdict": v_verdict,
                    "judge_verdict": j_verdict,
                    "verifier_judge_agree": (v_verdict == "tie") or (j_verdict == v_verdict),
                    "problem": problem,
                    "variant_response": v["response"],
                    "baseline_response": b["response"],
                }
            )

    def rate(entry, mechanism, kind):
        n = max(1, entry["n"])
        return entry[mechanism][kind] / n

    results = {}
    # The manual asks for all five supplied perturbation categories. clean_correct is the
    # baseline every controlled pair is measured against, so its preference rates are
    # undefined by construction; report its own correctness rates instead and mark it.
    clean = list(by_problem_clean.values()) if by_problem_clean else []
    clean_verifier = [exact_reward(v["response"], str(v.get("gold_final", ""))) for v in clean]
    clean_judge_wins = sum(1 for e in examples if e["judge_verdict"] == "baseline")
    results["clean_correct"] = {
        "diagnostic_question": "reference response every controlled pair is measured against",
        "baseline": None,
        "is_reference": True,
        "num_pairs": len(clean),
        "rlvr": {"better": 0.0, "tie": 1.0, "wrong": 0.0},
        "rlaif": {"better": 0.0, "tie": 1.0, "wrong": 0.0},
        "note": (
            "Reference arm: preference rates are undefined because it is compared against nothing. "
            "verifier_correct_rate is the share of these responses whose designated final answer matches gold."
        ),
        "verifier_correct_rate": (sum(clean_verifier) / len(clean_verifier)) if clean_verifier else None,
        "judge_beat_the_degraded_variant_fraction": (clean_judge_wins / len(examples)) if examples else None,
    }

    for variant, entry in per_variant.items():
        results[variant] = {
            "diagnostic_question": CONTROLLED_PAIRS[variant][1],
            "baseline": CONTROLLED_PAIRS[variant][0],
            "num_pairs": entry["n"],
            "rlvr": {k: rate(entry, "rlvr", k) for k in ("better", "tie", "wrong")},
            "rlaif": {k: rate(entry, "rlaif", k) for k in ("better", "tie", "wrong")},
        }
        for mechanism in ("rlvr", "rlaif"):
            r = results[variant][mechanism]
            print(
                f"{variant:34s} {mechanism}: better={r['better']:.3f} tie={r['tie']:.3f} wrong={r['wrong']:.3f}"
            )

    def sensitivity(variant):
        if variant not in per_variant:
            return None
        entry = per_variant[variant]
        n = max(1, entry["n"])
        return {
            "num_pairs": entry["n"],
            "rlvr": entry["rlvr"]["better"] / n,
            "rlaif": entry["rlaif"]["better"] / n,
            "rlvr_tie": entry["rlvr"]["tie"] / n,
            "rlaif_tie": entry["rlaif"]["tie"] / n,
            "rlvr_wrong": entry["rlvr"]["wrong"] / n,
            "rlaif_wrong": entry["rlaif"]["wrong"] / n,
        }

    payload = {
        "description": (
            "Task 5 Step 2: controlled reward diagnostics. Each released problem contributes one controlled pair "
            "per perturbation, always against the clean, correct response. 'better' means the mechanism prefers "
            "the diagnostically better response, 'wrong' means it prefers the degraded one, 'tie' means the "
            "mechanism cannot separate them (for the binary verifier this is the expected outcome whenever the "
            "designated final answer is unchanged)."
        ),
        "definitions": {
            "S_reason": "corrupt_reasoning_correct_final vs clean_correct (final answer held correct, reasoning degraded)",
            "S_outcome": "good_reasoning_wrong_final vs clean_correct (reasoning held correct, designated final answer changed)",
            "filler_susceptibility": "persuasive_filler_correct vs clean_correct (verbosity/persuasion probe)",
            "distractor_robustness": "gold_distractor_wrong_final vs clean_correct (gold number present as a distractor)",
        },
        "num_problems": len(groups),
        "by_variant": results,
        "S_reason": sensitivity("corrupt_reasoning_correct_final"),
        "S_outcome": sensitivity("good_reasoning_wrong_final"),
        "filler_susceptibility": sensitivity("persuasive_filler_correct"),
        "distractor_robustness": sensitivity("gold_distractor_wrong_final"),
        "verifier_judge_pair_agreement": sum(1 for e in examples if e["verifier_judge_agree"]) / max(1, len(examples)),
    }
    write_jsonl(outdir / "samples" / "perturbation_examples.jsonl", examples)
    out = outdir / "perturbation_study.json"
    save_json(out, payload)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()