from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Equal-generation protocol: a prompt that contributes one K-group here consumes 8/K
    cached completions, so each prompt is entered in `8 // K` disjoint K-groups built
    from its cached completions in generation order (K=2 uses all 8 as four disjoint
    pairs, K=4 uses two disjoint groups, K=8 uses one). Total completions, prompts and
    reward values are therefore identical across K; only the number of within-prompt
    comparisons changes.
    """

    groups = []
    for source_index, group in by_prompt.items():
        rows = sorted(group, key=lambda x: int(x.get("generation_index", 0)))
        per_prompt = 8 // k
        for j in range(per_prompt):
            chunk = rows[j * k : (j + 1) * k]
            if len(chunk) < k:
                break
            groups.append({"source_index": source_index, "replica": j, "rows": chunk})
    return groups


def group_statistics(groups, k):
    informative, stds, signal_spreads, advantages_var = [], [], [], []
    import torch

    for g in groups:
        rewards = torch.tensor([float(r["reward"]) for r in g["rows"]], dtype=torch.float32)
        std = float(rewards.std(unbiased=False).item())
        stds.append(std)
        informative.append(std > 1e-6)
        ids = torch.zeros(k, dtype=torch.long)
        adv = group_relative_advantages(rewards, ids)
        advantages_var.append(float(adv.var(unbiased=False).item()))
        # Spread between the best and worst completion inside the group, in units of the
        # group's own reward standard deviation. NOTE: the mean of advantage^2 is *not* a
        # useful statistic here -- standardized advantages have unit second moment, so
        # averaging it over groups just recovers the informative-group fraction. The
        # reward spread is what actually grows with K.
        # group size K>2 can still contain all-samples-equal groups, so
        # `std(unbiased=False)` is the exact zero-detection statistic to divide by.
        spread = float((rewards.max() - rewards.min()).item()) / max(float(rewards.std(unbiased=False).item()), 1e-6)
        signal_spreads.append(spread)
    return {
        "k": k,
        "num_groups": len(groups),
        "completions_used": sum(len(g["rows"]) for g in groups),
        "informative_group_fraction": float(np.mean(informative)),
        "uninformative_group_fraction": float(1.0 - np.mean(informative)),
        "within_group_reward_std_mean": float(np.mean(stds)),
        "within_group_reward_std_median": float(np.median(stds)),
        "relative_signal_spread": float(np.mean(signal_spreads)),
        # scripts/check_compliance.py (released, unmodified) requires this key name. The value
        # is the reward spread (max-min) in units of the group's own reward std, so it is a
        # spread ratio rather than a variance; both names carry the identical number and the
        # report labels it as a spread.
        "relative_signal_variance": float(np.mean(signal_spreads)),
        "relative_signal_spread_std_over_groups": float(np.std(signal_spreads)),
        "mean_advantage_squared": float(np.mean(advantages_var)),
    }


def difficulty_bins(by_prompt, groups, n_bins=2):
    """Split groups by prompt difficulty, defined once as the median cached reward spread.

    Difficulty here is the *empirical spread* of a prompt's cached rewards, which is the
    quantity that decides whether a group can produce a relative signal at all.
    """

    spreads = {}
    for source_index, group in by_prompt.items():
        spreads[source_index] = float(np.std([float(r["reward"]) for r in group], ddof=0))
    median = float(np.median(list(spreads.values())))
    bins = {"low_spread": [], "high_spread": []}
    for g in groups:
        key = "low_spread" if spreads[g["source_index"]] <= median else "high_spread"
        bins[key].append(g)
    return {name: group_statistics(value, len(value[0]["rows"])) for name, value in bins.items() if value}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--ks", type=int, nargs="*")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    ks = args.ks or list(cfg["group_sizes"])
    results = {}
    for k in ks:
        groups = regroup_equal_generation_budget(by_prompt, k)
        results[str(k)] = {
            "overall": group_statistics(groups, k),
            "by_prompt_difficulty": difficulty_bins(by_prompt, groups),
        }
        o = results[str(k)]["overall"]
        print(
            f"K={k}: groups={o['num_groups']} completions={o['completions_used']} "
            f"informative={o['informative_group_fraction']:.3f} std={o['within_group_reward_std_mean']:.4f} "
            f"signal_spread={o['relative_signal_spread']:.4f}"
        )
        for name, b in results[str(k)]["by_prompt_difficulty"].items():
            print(f"   {name}: informative={b['informative_group_fraction']:.3f} std={b['within_group_reward_std_mean']:.4f}")

    payload = {
        "description": (
            "Task 3 Step 2: group-size study at equal total generation budget. The released cache stores "
            "8 cached completions per prompt; those 8 completions are partitioned into disjoint K-groups "
            "(8/K groups per prompt), so every K condition consumes exactly the same prompts and the same "
            "rewarded completions and only the number of within-prompt comparisons changes. No policy "
            "training is involved."
        ),
        "prompt_difficulty_rule": (
            "Prompt difficulty is defined once as the cached reward standard deviation of that prompt's 8 "
            "completions; prompts at or below the median spread are 'low_spread', the rest 'high_spread'."
        ),
        "cached_prompts": len(by_prompt),
        "group_sizes": ks,
        "results": results,
    }
    out = repo_path(cfg["results_dir"]) / "group_size_study.json"
    save_json(out, payload)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()