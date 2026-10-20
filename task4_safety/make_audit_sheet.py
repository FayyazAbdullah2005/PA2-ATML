from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument(
        "--policy",
        default="sft",
        help="Policy whose responses are audited ('all' writes one row per policy for the same 60 prompts)",
    )
    ap.add_argument("--out-name", default="manual_audit_sheet.csv")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    # The audit sheet carries prompt and response text only; the AI label is deliberately excluded.
    policies = ["sft", "dpo", "ppo", "grpo"] if args.policy == "all" else [args.policy]
    frames = []
    for policy_name in policies:
        path = outdir / f"generated_{policy_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing generated responses: {path}")
        rows = read_jsonl(path)
        keep = {r["xstest_id"]: r for r in rows if int(r["xstest_id"]) in set(ids)}
        missing = [i for i in ids if i not in keep]
        if missing:
            raise KeyError(f"{policy_name} is missing audited prompts {missing[:5]}")
        frames.append(
            pd.DataFrame(
                [
                    {
                        "xstest_id": i,
                        "policy": policy_name,
                        "benchmark_class": keep[i]["benchmark_class"],
                        "type": keep[i]["type"],
                        "prompt": keep[i]["prompt"],
                        "response": keep[i]["response"],
                        "response_tokens": keep[i]["response_tokens"],
                        "manual_label": "",
                    }
                    for i in ids
                ]
            )
        )
    sheet = pd.concat(frames, ignore_index=True)
    out = outdir / args.out_name
    sheet.to_csv(out, index=False)
    print("Wrote audit sheet:", out, f"({len(sheet)} rows)")
    print("Join these IDs to each policy's generated responses and label without viewing AI labels first.")
    print("Fill the `manual_label` column with one of: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")
    print("Then run: python -m task4_safety.evaluate_safety")


if __name__ == "__main__":
    main()