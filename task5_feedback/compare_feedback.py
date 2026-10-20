from __future__ import annotations

import argparse

import json

from common.data import load_yaml, repo_path


def load_json(path):
    # Inlined: common.logging_utils imports torch at module scope, and this script only
    # merges saved result files, so it must not require the model stack to run.
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def save_json(path, payload):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--results-dir", default="results/task5_feedback")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = repo_path(args.results_dir)

    in_domain = load_json(root / "gsm_eval.json")
    transfer = load_json(root / "transfer_eval.json")
    diagnostics = load_json(root / "perturbation_study.json")

    policies = in_domain["policies"]
    comparison = {}
    for name, stats in policies.items():
        entry = {"in_domain": stats}
        if name in transfer["policies"]:
            t = transfer["policies"][name]
            entry["transfer"] = t
            entry["drop_in_exact_accuracy"] = stats["exact_accuracy"] - t["exact_accuracy"]
            if "pairwise_vs_sft" in stats and "pairwise_vs_sft" in t:
                entry["drop_in_pairwise_win_rate"] = (
                    stats["pairwise_vs_sft"]["win_rate"] - t["pairwise_vs_sft"]["win_rate"]
                )
            entry["response_length_ratio_transfer_over_in_domain"] = (
                t["mean_response_tokens"] / max(1e-9, stats["mean_response_tokens"])
            )
        comparison[name] = entry

    payload = {
        "description": (
            "Task 5 synthesis: RLVR (exact final-answer verification) versus RLAIF (direct AI pairwise "
            "preferences) with the policy family and group-based RL procedure held approximately fixed. "
            "Numerical scales of the two rewards are not comparable; the comparison uses correctness, "
            "ranking, disagreement patterns and out-of-domain transfer."
        ),
        "policy_family": cfg["base_model"],
        "policies": comparison,
        "verifier_judge_agreement_in_domain": in_domain.get("verifier_judge_agreement"),
        "verifier_judge_agreement_transfer": transfer.get("verifier_judge_agreement"),
        "controlled_diagnostics": {
            "S_reason": diagnostics.get("S_reason"),
            "S_outcome": diagnostics.get("S_outcome"),
            "filler_susceptibility": diagnostics.get("filler_susceptibility"),
            "distractor_robustness": diagnostics.get("distractor_robustness"),
            "by_variant": diagnostics.get("by_variant"),
            "verifier_judge_pair_agreement": diagnostics.get("verifier_judge_pair_agreement"),
        },
        "feedback_source_summary": {
            "coverage": (
                "The verifier sees only the designated final answer, so it cannot distinguish two wrong "
                "solutions or a correct answer reached by corrupted reasoning; the judge can separate them "
                "but only through its own notion of quality."
            ),
            "noise": "The judge is a fixed 3B model; its labels are reproducible but not ground truth (see the Task 4 manual audit).",
            "exploitability": "The verifier is exploitable through answer-format gaming; the judge is exploitable through style and verbosity.",
            "compute": "One verifier call per response versus K(K-1)/2 judge calls per prompt, with the 4-bit judge loaded at runtime.",
        },
    }
    out = root / "feedback_comparison.json"
    save_json(out, payload)
    print(f"saved -> {out}")
    for name, entry in comparison.items():
        dom = entry["in_domain"]
        line = f"{name:6s} in-domain exact={dom['exact_accuracy']:.3f}"
        if "pairwise_vs_sft" in dom:
            line += f" win_vs_sft={dom['pairwise_vs_sft']['win_rate']:.3f}"
        if "transfer" in entry:
            line += f" | transfer exact={entry['transfer']['exact_accuracy']:.3f} drop={entry['drop_in_exact_accuracy']:.3f}"
        print(line)


if __name__ == "__main__":
    main()