from __future__ import annotations

import argparse

from common.data import (
    load_yaml,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.logging_utils import load_json, save_json
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import clear_gpu
from task1_dpo import evaluate as dpo_eval
from task1_dpo import protocol
from task1_dpo.train import run_training

STRATUM_KEYS = ("stratum", "length_stratum", "bucket", "group", "length_group")


def stratum_of(row):
    """Prefer the released stratum label; otherwise infer it from the length gap."""

    for key in STRATUM_KEYS:
        if key in row and row[key]:
            return str(row[key])
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if isinstance(chosen, str):
        c, r = chosen, rejected
    else:
        c, r = preference_responses(row)
    lc, lr = word_count(c), word_count(r)
    if lc > 1.5 * max(lr, 1):
        return "preferred_longer"
    if lr > 1.5 * max(lc, 1):
        return "rejected_longer"
    return "matched_length"


def canonical_stratum(label):
    s = str(label).lower().replace("-", "_").replace(" ", "_")
    if "reject" in s and "long" in s:
        return "rejected_longer"
    if "match" in s or "balanced" in s:
        return "matched_length"
    if "prefer" in s and "long" in s:
        return "preferred_longer"
    if "short" in s and "prefer" in s:
        return "preferred_longer"
    if "short" in s and "reject" in s:
        return "rejected_longer"
    return s


def word_limit_report(policy, tokenizer, rows, cfg, name, batch_size=4):
    prompts = [prompt_messages(r) for r in rows]
    rollouts = protocol.generate_with_diagnostics(policy, tokenizer, prompts, cfg, batch_size=batch_size, with_reference=False)
    records, compliance = [], []
    for row, prompt, response, n in zip(rows, prompts, rollouts["responses"], rollouts["response_lengths"]):
        prompt_text = " ".join(str(m.get("content", "")) for m in prompt)
        limit = parse_word_limit(prompt_text)
        ok = word_limit_compliance(prompt_text, response)
        if ok is not None:
            compliance.append(ok)
        records.append(
            {
                "policy": name,
                "prompt_id": row.get("prompt_id"),
                "prompt": prompt_text,
                "word_limit": limit,
                "response": response,
                "word_count": word_count(response),
                "compliant": ok,
                "response_tokens": int(n),
            }
        )
    return records, (float(sum(compliance) / len(compliance)) if compliance else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--standard-adapter", default="outputs/task1_dpo/standard")
    ap.add_argument("--train-length-balanced", action="store_true", default=True)
    ap.add_argument("--eval-prompt-limit", type=int)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    word_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    for row in stratified:
        row["_stratum"] = canonical_stratum(stratum_of(row))
    strata = sorted({r["_stratum"] for r in stratified})

    train_summary = None
    length_adapter = repo_path(cfg["length_output"])
    length_summary_path = repo_path(cfg["results_dir"]) / "length_balanced_train_summary.json"
    if args.train_length_balanced and (length_adapter / "adapter_model.safetensors").exists() and length_summary_path.exists():
        # The length-balanced model was already trained by an earlier pass whose evaluation
        # crashed; reuse it rather than spending another 20 minutes retraining.
        train_summary = load_json(length_summary_path)
        print(f"[skip] length-balanced training already complete ({length_summary_path})", flush=True)
    elif args.train_length_balanced:
        print("=== training the length-balanced DPO condition ===", flush=True)
        train_summary = run_training(
            args.config,
            run_name="length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=cfg["length_output"],
        )

    conditions = [
        ("standard", args.standard_adapter),
        ("length_balanced", cfg["length_output"]),
    ]
    per_condition = {}
    for name, adapter in conditions:
        cond_path = repo_path(cfg["results_dir"]) / f"length_{name}.json"
        if cond_path.exists():
            print(f"[skip] {name}: reusing completed condition", flush=True)
            per_condition[name] = load_json(cond_path)
            continue
        print(f"=== evaluating {name} on the length-stratified held-out set ===", flush=True)
        bundle = dpo_eval.load_evaluation_bundle(args.config, adapter)
        bundle["adapter"] = adapter
        tokenizer = bundle["tokenizer"]
        overall = protocol.evaluate_preference_pairs(bundle["policy"], tokenizer, stratified, cfg)
        by_stratum = {}
        for stratum in strata:
            subset = [r for r in stratified if r["_stratum"] == stratum]
            by_stratum[stratum] = protocol.evaluate_preference_pairs(bundle["policy"], tokenizer, subset, cfg)
            print(f"  {stratum:16s} n={by_stratum[stratum]['num_pairs']:4d} "
                  f"acc={by_stratum[stratum]['preference_accuracy']:.4f}", flush=True)

        records, compliance = word_limit_report(bundle["policy"], tokenizer, word_rows, cfg, name)
        # Task 1 Step 3 asks for generated length and explicit word-limit compliance
        # on the common word-limit prompts.  Generating the entire length-stratified
        # preference set again and assigning reward-model scores is not part of that
        # experiment, so report the requested generation statistics from this fixed
        # ten-prompt common set instead.
        lengths = protocol.length_stats([r["response_tokens"] for r in records])

        write_jsonl(repo_path(cfg["results_dir"]) / "samples" / f"{name}_word_limit.jsonl", records)
        per_condition[name] = {
            "adapter": adapter,
            "stratified_overall": overall,
            "by_stratum": by_stratum,
            "generation_prompt_set": "data/word_limit_prompts.jsonl",
            "generated_length": lengths,
            "word_limit_compliance": compliance,
        }
            # Persist each condition immediately so the second cannot cost the first.
        save_json(cond_path, per_condition[name])
        print(f"[done] {name}: compliance={per_condition[name]['word_limit_compliance']} "
              f"len={per_condition[name]['generated_length']['mean']:.1f}", flush=True)

        bundle.clear()
        clear_gpu()

    payload = {
        "description": (
            "Task 1 Step 3: length-confounding study. The standard and length-balanced DPO policies are "
            "scored on the supplied length-stratified held-out set (per-stratum preference accuracy) and on "
            "the common word-limit prompt set (generated length and explicit word-limit compliance)."
        ),
        "strata": strata,
        "stratified_eval_rows": len(stratified),
        "length_balanced_train_rows": len(balanced),
        "word_limit_prompts": len(word_rows),
        "length_balanced_train_summary": train_summary,
        "conditions": per_condition,
        "dataset_length_correlation": dataset_length_stats(stratified),
    }
    out = repo_path(cfg["results_dir"]) / "length_confounding.json"
    save_json(out, payload)
    print(f"saved -> {out}")
    for name, res in per_condition.items():
        print(f"{name:16s} compliance={res['word_limit_compliance']} "
              f"len={res['generated_length']['mean']:.1f}")


def dataset_length_stats(rows):
    """Length structure of the preference data itself (a dataset property)."""

    deltas, chosen_lens, rejected_lens = [], [], []
    for row in rows:
        c, r = preference_responses(row)
        lc, lr = word_count(c), word_count(r)
        chosen_lens.append(lc)
        rejected_lens.append(lr)
        deltas.append(lc - lr)

    import numpy as np

    d = np.asarray(deltas, dtype=float)
    return {
        "chosen_words_mean": float(np.mean(chosen_lens)),
        "rejected_words_mean": float(np.mean(rejected_lens)),
        "chosen_minus_rejected_words_mean": float(d.mean()),
        "chosen_longer_fraction": float((d > 0).mean()),
    }


if __name__ == "__main__":
    main()
