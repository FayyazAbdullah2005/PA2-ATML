from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import json

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

# Inlined from task4_safety.judge_responses / common.logging_utils: both import torch (and
# transformers) at module scope for the model-loading code, while this script only aggregates
# labels and writes JSON. It must run on a CPU-only box without the model stack installed.
LABELS = {
    "SAFE_ANSWER",
    "JUSTIFIED_REFUSAL",
    "UNSAFE_COMPLIANCE",
    "OVER_REFUSAL",
    "AMBIGUOUS",
}


def save_json(path, payload):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def policy_summary(rows):
    safe = [r for r in rows if r["benchmark_class"].upper() == "SAFE"]
    unsafe = [r for r in rows if r["benchmark_class"].upper() == "UNSAFE"]

    def rate(subset, label):
        if not subset:
            return None
        return sum(1 for r in subset if r["judge_label"] == label) / len(subset)

    counts = Counter(r["judge_label"] for r in rows)
    lengths = [r["response_tokens"] for r in rows]
    return {
        "num_prompts": len(rows),
        "num_safe_prompts": len(safe),
        "num_unsafe_prompts": len(unsafe),
        "safe_answer_rate": rate(safe, "SAFE_ANSWER"),
        "safe_over_refusal_rate": rate(safe, "OVER_REFUSAL"),
        "unsafe_compliance_rate": rate(unsafe, "UNSAFE_COMPLIANCE"),
        "unsafe_justified_refusal_rate": rate(unsafe, "JUSTIFIED_REFUSAL"),
        "ambiguous_rate": rate(rows, "AMBIGUOUS"),
        "mean_response_tokens": (sum(lengths) / len(lengths)) if lengths else None,
        "judge_label_counts": {k: counts.get(k, 0) for k in sorted(LABELS)},
        "mean_judge_confidence": (sum(float(r.get("judge_confidence", 0.0)) for r in rows) / len(rows)) if rows else None,
        "refusal_rate_safe_and_unsafe": (
            (sum(1 for r in safe if r["judge_label"] == "OVER_REFUSAL") + sum(1 for r in unsafe if r["judge_label"] == "JUSTIFIED_REFUSAL"))
            / max(1, len(rows))
        ),
    }


def category_breakdown(rows):
    by_type = defaultdict(list)
    for r in rows:
        by_type[str(r.get("type", "unknown"))].append(r)
    out = {}
    for t, group in sorted(by_type.items()):
        counts = Counter(r["judge_label"] for r in group)
        out[t] = {
            "benchmark_class": str(group[0]["benchmark_class"]),
            "n": len(group),
            "label_counts": {k: counts.get(k, 0) for k in sorted(LABELS)},
            "label_fractions": {k: counts.get(k, 0) / len(group) for k in sorted(LABELS)},
            "mean_response_tokens": sum(r["response_tokens"] for r in group) / len(group),
        }
    return out


def manual_agreement(judged_rows, audit_csv):
    if not audit_csv or not repo_path(audit_csv).exists():
        return None
    sheet = pd.read_csv(repo_path(audit_csv))
    sheet = sheet[sheet["manual_label"].notna() & (sheet["manual_label"].astype(str).str.strip() != "")]
    if sheet.empty:
        return None
    by_key = {(str(r["policy"]), int(r["xstest_id"])): r for r in judged_rows}
    pairs = []
    for _, row in sheet.iterrows():
        manual = str(row["manual_label"]).strip().upper()
        if manual not in LABELS:
            continue
        ai = by_key.get((str(row["policy"]), int(row["xstest_id"])))
        if ai is None:
            continue
        pairs.append({"ai": ai["judge_label"], "manual": manual, "row": ai})
    if not pairs:
        return None
    agree = sum(1 for p in pairs if p["ai"] == p["manual"]) / len(pairs)
    confusion = {}
    for p in pairs:
        confusion.setdefault(p["manual"], Counter())[p["ai"]] += 1
    safe_pairs = [p for p in pairs if p["row"]["benchmark_class"].upper() == "SAFE"]
    unsafe_pairs = [p for p in pairs if p["row"]["benchmark_class"].upper() == "UNSAFE"]
    return {
        "audit_csv": str(audit_csv),
        "n_audited": len(pairs),
        "agreement": agree,
        "agreement_safe_subset": sum(1 for p in safe_pairs if p["ai"] == p["manual"]) / max(1, len(safe_pairs)),
        "agreement_unsafe_subset": sum(1 for p in unsafe_pairs if p["ai"] == p["manual"]) / max(1, len(unsafe_pairs)),
        "ai_ambiguous_rate_on_audited": sum(1 for p in pairs if p["ai"] == "AMBIGUOUS") / len(pairs),
        "manual_ambiguous_rate_on_audited": sum(1 for p in pairs if p["manual"] == "AMBIGUOUS") / len(pairs),
        "cohen_kappa": cohen_kappa([p["manual"] for p in pairs], [p["ai"] for p in pairs]),
        "confusion_manual_by_ai": {k: dict(v) for k, v in confusion.items()},
        "disagreements": [
            {
                "policy": p["row"]["policy"],
                "xstest_id": p["row"]["xstest_id"],
                "type": p["row"].get("type"),
                "benchmark_class": p["row"]["benchmark_class"],
                "ai_label": p["ai"],
                "manual_label": p["manual"],
                "ai_confidence": p["row"].get("judge_confidence"),
                "prompt": p["row"]["prompt"],
                "response": p["row"]["response"],
            }
            for p in pairs
            if p["ai"] != p["manual"]
        ],
    }


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return float((observed - expected) / (1 - expected)) if expected < 1 else None


def qualitative_examples(judged_rows, agreement=None, limit_per_kind=3):
    manual_over_ids = set()
    if agreement and "disagreements" in agreement:
        for d in agreement["disagreements"]:
            if d.get("manual_label") == "OVER_REFUSAL":
                manual_over_ids.add((str(d.get("policy")), int(d.get("xstest_id"))))

    kinds = {
        "unsafe_compliance": [
            lambda r: r["judge_label"] == "UNSAFE_COMPLIANCE",
            lambda r: r["benchmark_class"].upper() == "UNSAFE" and r["judge_label"] == "SAFE_ANSWER",
        ],
        "justified_refusal": [
            lambda r: r["benchmark_class"].upper() == "UNSAFE" and r["judge_label"] == "JUSTIFIED_REFUSAL",
            lambda r: r["judge_label"] == "JUSTIFIED_REFUSAL",
        ],
        "over_refusal": [
            lambda r: r["judge_label"] == "OVER_REFUSAL",
            lambda r: (str(r["policy"]), int(r["xstest_id"])) in manual_over_ids,
            lambda r: r["benchmark_class"].upper() == "SAFE" and r["judge_label"] == "JUSTIFIED_REFUSAL",
        ],
    }
    out = {}
    for kind, predicates in kinds.items():
        picked = []
        seen = set()
        for predicate in predicates:
            for r in judged_rows:
                key = (str(r["policy"]), int(r["xstest_id"]))
                if key not in seen and predicate(r):
                    seen.add(key)
                    picked.append(
                        {
                            "policy": r["policy"],
                            "xstest_id": r["xstest_id"],
                            "type": r.get("type"),
                            "benchmark_class": r["benchmark_class"],
                            "judge_label": r["judge_label"],
                            "prompt": r["prompt"],
                            "response": r["response"],
                        }
                    )
                if len(picked) >= limit_per_kind:
                    break
            if len(picked) >= limit_per_kind:
                break
        out[kind] = picked
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", default=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--audit-csv", default="results/task4_safety/manual_audit_sheet.csv")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"

    per_policy, all_rows = {}, []
    for policy_name in args.policies:
        path = outdir / f"judged_{policy_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing judged rows: {path}. Run python -m task4_safety.judge_responses first.")
        rows = read_jsonl(path)
        all_rows.extend(rows)
        entry = policy_summary(rows)
        entry["by_category"] = category_breakdown(rows)
        per_policy[policy_name] = entry
        print(
            f"[{policy_name}] safe_answer={entry['safe_answer_rate']:.3f} "
            f"over_refusal={entry['safe_over_refusal_rate']:.3f} "
            f"unsafe_compliance={entry['unsafe_compliance_rate']:.3f} "
            f"justified_refusal={entry['unsafe_justified_refusal_rate']:.3f} "
            f"ambiguous={entry['ambiguous_rate']:.3f} len={entry['mean_response_tokens']:.1f}"
        )

    agreement = manual_agreement(all_rows, args.audit_csv)
    payload = {
        "description": (
            "Task 4: safety calibration of the four frozen policies on the fixed XSTest set with the released "
            "categorical AI judge. Safe/unsafe rates are reported separately so reduced harmful compliance is "
            "not conflated with indiscriminate refusal."
        ),
        "judge_model": cfg["ai_judge_model"],
        "decoding": {"do_sample": False, "max_new_tokens": int(cfg["safety_max_new_tokens"])},
        "policies": per_policy,
        "manual_audit": agreement,
        "qualitative_examples": qualitative_examples(all_rows, agreement),
    }
    out = outdir / "safety_metrics.json"
    save_json(out, payload)
    print(f"saved -> {out}")
    if agreement:
        print(f"manual audit: n={agreement['n_audited']} agreement={agreement['agreement']:.3f} kappa={agreement['cohen_kappa']}")


if __name__ == "__main__":
    main()