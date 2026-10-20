"""Select the qualitative examples the manual requires for Tasks 1-3.

Usage:
    python -m scripts.qualitative_examples

The manual asks for brief examples, not raw dumps:
  Task 1 (i) a preference/reward-versus-quality disagreement, (ii) length bias / compliance
  Task 2 at least one case where reward and response quality agree and one where they disagree
  Task 3 examples that help interpret the normalization and group-informativeness results

Selection is objective and reproducible, not hand-picked: every case is scored by the same
rules below and only the minimum text is quoted, as the manual requires.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from common.data import read_jsonl, repo_path, write_jsonl

# common.metrics imports torch at module scope; this script only needs two pure-Python
# helpers from it, so they are inlined to keep the script runnable without the model stack.
_WORD_LIMIT_PATTERNS = [
    r"(?:at most|no more than|under|within)\s+(\d+)\s+words?",
    r"(?:in|use)\s+(\d+)\s+words?\s+(?:or fewer|max(?:imum)?)",
    r"(?:maximum|max)\s+(?:of\s+)?(\d+)\s+words?",
]


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def parse_word_limit(prompt):
    lower = str(prompt).lower()
    for pattern in _WORD_LIMIT_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            return int(m.group(1))
    return None


def save_json(path, payload):
    """Local writer.

    common.logging_utils imports torch at module scope, and this script only reads and
    writes JSON, so it must not drag the model stack in just to emit a file.
    """

    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

RESULTS = repo_path("results")
QUOTE = 420


def repetition_score(text: str, n: int = 8) -> float:
    """Fraction of n-grams that are repeats. Near 1.0 means the model is looping."""

    words = text.split()
    if len(words) < 2 * n:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def quality_flags(record):
    """Objective quality problems that the reward model is known not to penalise."""

    text = record.get("response", "") or ""
    flags = []
    if not record.get("terminated_with_eos", True):
        flags.append("no_eos")
    if record.get("truncated"):
        flags.append("truncated")
    if repetition_score(text) > 0.35:
        flags.append("repetitive")
    prompt = record.get("prompt", "") or ""
    limit = parse_word_limit(prompt)
    if limit is not None and word_count(text) > limit:
        flags.append("exceeds_word_limit")
    if len(text.strip()) < 15:
        flags.append("empty")
    return flags


def summarise(record, reason):
    return {
        "condition": record.get("policy") or record.get("condition"),
        "prompt": (record.get("prompt") or "")[:300],
        "response": (record.get("response") or "")[:QUOTE],
        "response_tokens": record.get("response_tokens"),
        "reward_model_score": record.get("reward_model_score"),
        "flags": quality_flags(record),
        "why": reason,
    }


# --------------------------------------------------------------------------- Task 1
def task1():
    by_condition = {}
    for path in sorted((RESULTS / "task1_dpo" / "samples").glob("*_generations.jsonl")):
        name = path.stem.replace("_generations", "")
        try:
            rows = read_jsonl(path.relative_to(repo_path(".")))
        except Exception:
            continue
        by_condition[name] = rows

    word_limit_rows = []
    wl = RESULTS / "task1_dpo" / "samples" / "standard_word_limit.jsonl"
    if wl.exists():
        word_limit_rows = read_jsonl("results/task1_dpo/samples/standard_word_limit.jsonl")

    # (i) reward says good, quality checks say bad: high reward + at least one quality flag.
    disagreements = []
    for name, rows in by_condition.items():
        scored = [r for r in rows if r.get("reward_model_score") is not None]
        if not scored:
            continue
        cutoff = sorted(r["reward_model_score"] for r in scored)[int(0.7 * len(scored))]
        for r in scored:
            flags = quality_flags(r)
            if r["reward_model_score"] >= cutoff and flags:
                disagreements.append(summarise(r, f"reward in the top 30% for {name} but {'/'.join(flags)}"))
    disagreements.sort(key=lambda d: -(d["reward_model_score"] or 0))

    # (ii) length bias / instruction compliance: word-limit violations, worst first.
    compliance = []
    for r in sorted(word_limit_rows, key=lambda r: -(r.get("word_count") or 0) / max(1, r.get("word_limit") or 1)):
        compliance.append(
            {
                "condition": r.get("policy"),
                "prompt": (r.get("prompt") or "")[:300],
                "prompt_id": r.get("prompt_id"),
                "word_limit": r.get("word_limit"),
                "word_count": r.get("word_count"),
                "compliant": r.get("compliant"),
                "response": (r.get("response") or "")[:QUOTE],
                "why": ("overran the requested word limit"
                        if not r.get("compliant") else "respected the requested word limit"),
            }
        )

    # Length shift between the standard and length-balanced policies on shared prompts.
    # analyze_length.py writes this pair under its own filename, so look for both spellings.
    length_shift = []
    std_rows = by_condition.get("standard") or by_condition.get("length_standard") or []
    bal_rows = by_condition.get("length_balanced") or by_condition.get("length_length_balanced") or []
    if not bal_rows:
        bal_path = RESULTS / "task1_dpo" / "samples" / "length_balanced_length_stratified_generations.jsonl"
        if bal_path.exists():
            bal_rows = read_jsonl("results/task1_dpo/samples/length_balanced_length_stratified_generations.jsonl")
    std = {r.get("prompt"): r for r in std_rows if r.get("prompt")}
    bal = {r.get("prompt"): r for r in bal_rows if r.get("prompt")}
    for prompt in sorted(set(std) & set(bal)):
        d = (bal[prompt].get("response_tokens") or 0) - (std[prompt].get("response_tokens") or 0)
        length_shift.append(
            {
                "prompt": prompt[:300],
                "standard_tokens": std[prompt].get("response_tokens"),
                "length_balanced_tokens": bal[prompt].get("response_tokens"),
                "delta_tokens": d,
                "standard_reward": std[prompt].get("reward_model_score"),
                "length_balanced_reward": bal[prompt].get("reward_model_score"),
            }
        )
    length_shift.sort(key=lambda r: r["delta_tokens"])

    # Paired comparison on the 10 common word-limit prompts. Both conditions generated on the
    # same fixed prompt set, so this is a clean per-prompt contrast even though the released
    # runner does not retain the larger length-stratified generations per condition.
    paired = []
    std_wl = RESULTS / "task1_dpo" / "samples" / "standard_word_limit.jsonl"
    bal_wl = RESULTS / "task1_dpo" / "samples" / "length_balanced_word_limit.jsonl"
    if std_wl.exists() and bal_wl.exists():
        a = {r.get("prompt_id"): r for r in read_jsonl("results/task1_dpo/samples/standard_word_limit.jsonl")}
        b = {r.get("prompt_id"): r for r in read_jsonl("results/task1_dpo/samples/length_balanced_word_limit.jsonl")}
        for pid in sorted(set(a) & set(b), key=lambda x: str(x)):
            paired.append(
                {
                    "prompt_id": pid,
                    "prompt": (a[pid].get("prompt") or "")[:300],
                    "word_limit": a[pid].get("word_limit"),
                    "standard_tokens": a[pid].get("response_tokens"),
                    "length_balanced_tokens": b[pid].get("response_tokens"),
                    "delta_tokens": (b[pid].get("response_tokens") or 0) - (a[pid].get("response_tokens") or 0),
                    "standard_words": a[pid].get("word_count"),
                    "length_balanced_words": b[pid].get("word_count"),
                    "standard_compliant": a[pid].get("compliant"),
                    "length_balanced_compliant": b[pid].get("compliant"),
                }
            )
        paired.sort(key=lambda r: r["delta_tokens"])

    return {
        "selection_rule": (
            "Cases are chosen by fixed objective rules: (i) reward in the top 30% for that condition AND at "
            "least one quality flag (no EOS, truncated, repeated 8-grams >35%, empty); (ii) word-limit "
            "compliance on the fixed 10 prompts, plus the largest length shifts between the standard and "
            "length-balanced policies on identical prompts. Text is truncated to the minimum needed."
        ),
        "preference_vs_quality": disagreements[:6],
        "length_or_compliance": {
            "word_limit_cases": compliance[:10],
            "largest_length_shifts": length_shift[:5],
            "paired_word_limit_comparison": paired,
            "paired_note": (
                "Per-prompt contrast between the standard and length-balanced policies on the 10 "
                "common word-limit prompts. A positive delta means the length-balanced policy was longer."
            ),
        },
    }


# --------------------------------------------------------------------------- Task 2
def task2():
    dirs = {
        "standard": "results/task2_ppo/samples/standard_generations.jsonl",
        "kl_0": "results/task2_ppo/samples/kl_0_generations.jsonl",
        "kl_0.1": "results/task2_ppo/samples/kl_0.1_generations.jsonl",
        "kl_0.2": "results/task2_ppo/samples/kl_0.2_generations.jsonl",
    }
    loaded = {}
    for name, path in dirs.items():
        if (RESULTS.parent / path).exists():
            loaded[name] = {r["prompt"]: r for r in read_jsonl(path)}

    agree, disagree = [], []
    for name, rows in loaded.items():
        scored = [r for r in rows.values() if r.get("reward_model_score") is not None]
        if not scored:
            continue
        rewards = sorted(r["reward_model_score"] for r in scored)
        hi, lo = rewards[int(0.8 * len(rewards))], rewards[int(0.2 * len(rewards))]
        for r in scored:
            flags = quality_flags(r)
            if r["reward_model_score"] >= hi and not flags:
                agree.append(summarise(r, "high learned reward and no quality flag: reward and quality agree"))
            if r["reward_model_score"] >= hi and flags:
                disagree.append(summarise(r, f"high learned reward but {'/'.join(flags)}: reward and quality disagree"))
            if r["reward_model_score"] <= lo and "truncated" in flags:
                disagree.append(summarise(r, "low learned reward although the rollout was cut off by the length cap"))

    # Reward/response disagreement across the KL-pressure forks on identical prompts.
    cross = []
    if "kl_0" in loaded and "kl_0.2" in loaded:
        shared = set(loaded["kl_0"]) & set(loaded["kl_0.2"])
        for prompt in sorted(shared):
            a, b = loaded["kl_0"][prompt], loaded["kl_0.2"][prompt]
            if a.get("reward_model_score") is None or b.get("reward_model_score") is None:
                continue
            cross.append(
                {
                    "prompt": prompt[:300],
                    "kl_0": {"reward": a.get("reward_model_score"), "tokens": a.get("response_tokens"),
                             "excerpt": (a.get("response") or "")[:QUOTE]},
                    "kl_0.2": {"reward": b.get("reward_model_score"), "tokens": b.get("response_tokens"),
                               "excerpt": (b.get("response") or "")[:QUOTE]},
                    "reward_delta": b.get("reward_model_score") - a.get("reward_model_score"),
                    "token_delta": (b.get("response_tokens") or 0) - (a.get("response_tokens") or 0),
                }
            )
        cross.sort(key=lambda r: r["reward_delta"])

    return {
        "selection_rule": (
            "agree: reward in the top 20% for that fork and no quality flag. disagree: reward in the top 20% "
            "with a quality flag, or a truncated rollout that was penalised. Cross-fork rows compare "
            "beta_KL=0 with beta_KL=0.2 on identical prompts."
        ),
        "reward_quality_agree": agree[:5],
        "reward_quality_disagree": disagree[:5],
        "kl_pressure_pairs": cross[:5],
    }


# --------------------------------------------------------------------------- Task 3
def task3():
    heldout = {}
    for name in ("standard", "grpo", "dr_grpo"):
        path = RESULTS / "task3_grpo" / "samples" / f"{name}_generations.jsonl"
        if path.exists():
            rows = read_jsonl(f"results/task3_grpo/samples/{name}_generations.jsonl")
            by_prompt = defaultdict(list)
            for r in rows:
                by_prompt[r["prompt_index"]].append(r)
            heldout[name] = by_prompt

    groups = []
    for prompt_index, rows in sorted(heldout.get("standard", {}).items()):
        rewards = [r.get("reward_model_score") for r in rows if r.get("reward_model_score") is not None]
        if len(rewards) < 2:
            continue
        spread = max(rewards) - min(rewards)
        groups.append(
            {
                "prompt_index": prompt_index,
                "prompt": rows[0]["prompt"][:300],
                "reward_spread": spread,
                "informative": spread > 1e-6,
                "rewards": rewards,
                "response_tokens": [r.get("response_tokens") for r in rows],
                "best_response": max(rows, key=lambda r: r.get("reward_model_score") or -1e9)["response"][:QUOTE],
                "worst_response": min(rows, key=lambda r: r.get("reward_model_score") or 1e9)["response"][:QUOTE],
            }
        )
    uninformative = [g for g in groups if not g["informative"]]
    informative = [g for g in groups if g["informative"]]

    normalization = []
    if "grpo" in heldout and "dr_grpo" in heldout:
        shared = sorted(set(heldout["grpo"]) & set(heldout["dr_grpo"]))
        for prompt_index in shared:
            a = heldout["grpo"][prompt_index]
            b = heldout["dr_grpo"][prompt_index]
            ta = sum(r.get("response_tokens") or 0 for r in a) / max(1, len(a))
            tb = sum(r.get("response_tokens") or 0 for r in b) / max(1, len(b))
            normalization.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": a[0]["prompt"][:300],
                    "canonical_mean_tokens": ta,
                    "dr_grpo_mean_tokens": tb,
                    "token_delta": tb - ta,
                    "canonical_reward": sum(r.get("reward_model_score") or 0 for r in a) / max(1, len(a)),
                    "dr_grpo_reward": sum(r.get("reward_model_score") or 0 for r in b) / max(1, len(b)),
                    "canonical_excerpt": a[0]["response"][:QUOTE],
                    "dr_grpo_excerpt": b[0]["response"][:QUOTE],
                }
            )
        normalization.sort(key=lambda r: r["token_delta"])

    return {
        "selection_rule": (
            "Group informativeness: held-out prompts grouped by K completions, informative when the "
            "reward spread exceeds 1e-6; one fully uninformative and one fully informative group are "
            "quoted. Normalization: identical held-out prompts under canonical vs Dr. GRPO, sorted by "
            "generated-length difference."
        ),
        "normalization": {
            "largest_length_differences": normalization[:4],
            "mean_token_delta": (sum(r["token_delta"] for r in normalization) / len(normalization))
            if normalization else None,
        },
        "group_informativeness": {
            "num_groups": len(groups),
            "uninformative_fraction": (len(uninformative) / len(groups)) if groups else None,
            "example_uninformative": uninformative[:2],
            "example_informative": informative[:2],
        },
    }


def main():
    out_dir = RESULTS / "task1_dpo"
    for task, builder, name in ((1, task1, "qualitative_examples.json"),
                                (2, task2, "qualitative_examples.json"),
                                (3, task3, "qualitative_examples.json")):
        try:
            payload = builder()
        except Exception as exc:
            print(f"[skip] task{task}: {exc}")
            continue
        target = RESULTS / f"task{task}_{'ppo' if task == 2 else 'grpo' if task == 3 else 'dpo'}" / name
        save_json(target, payload)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()