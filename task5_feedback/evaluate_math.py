from __future__ import annotations

import argparse
from itertools import combinations

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def problem_text(row):
    for key in ("question", "problem", "prompt"):
        if isinstance(row.get(key), str):
            return row[key]
    messages = prompt_messages(row)
    return " ".join(str(m.get("content", "")) for m in messages)


def generate_for_policy(cfg, tokenizer, rows, name, batch_size=8, max_new_tokens=None, greedy=False):
    """Matched decoding across policies: same prompts, same seed, same length cap."""

    policy = load_frozen_policy(cfg, name)
    gen = cfg.get("generation", {})
    max_new_tokens = int(max_new_tokens or cfg["math_max_new_tokens"])
    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [prompt_messages(r) for r in chunk]
        set_seed(int(cfg["seed"]))
        out = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg.get("math_max_prompt_length", 512)),
            max_new_tokens=max_new_tokens,
            temperature=float(gen.get("temperature", 0.7)),
            top_p=float(gen.get("top_p", 0.9)),
            do_sample=False if greedy else bool(gen.get("do_sample", True)),
        )
        for row, response, n in zip(chunk, out["responses"], out["response_lengths"]):
            records.append(
                {
                    "policy": name,
                    "problem_index": str(row.get("source_index", row.get("problem_id", len(records)))),
                    "problem_id": str(row.get("problem_id", row.get("source_index", len(records)))),
                    "problem": problem_text(row),
                    "gold_final": str(row.get("gold_final", "")),
                    "response": response,
                    "response_tokens": int(n),
                }
            )
    del policy
    clear_gpu()
    return records


def verifier_stats(records):
    n = max(1, len(records))
    exact = [exact_reward(r["response"], r["gold_final"]) for r in records]
    parsed = [1.0 if extract_designated_final(r["response"]) is not None else 0.0 for r in records]
    lengths = [r["response_tokens"] for r in records]
    return {
        "exact_accuracy": sum(exact) / n,
        "format_compliance": sum(parsed) / n,
        "mean_response_tokens": sum(lengths) / n,
        "median_response_tokens": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "num_problems": len(records),
        "per_problem_exact": {r["problem_id"]: r["_exact"] for r in records},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--policies", nargs="*", default=["sft", "rlvr", "rlaif"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    responses = {}
    for name in args.policies:
        print(f"=== generating {name} responses ===", flush=True)
        records = generate_for_policy(
            cfg, tokenizer, rows, name, batch_size=args.batch_size, greedy=args.greedy
        )
        for r in records:
            r["_exact"] = exact_reward(r["response"], r["gold_final"])
        responses[name] = records
        write_jsonl(outdir / f"{args.dataset}_responses_{name}.jsonl", [
            {k: v for k, v in r.items() if k != "_exact"} for r in records
        ])
        s = verifier_stats(records)
        print(f"  exact={s['exact_accuracy']:.3f} format={s['format_compliance']:.3f} len={s['mean_response_tokens']:.1f}", flush=True)

    payload = {
        "description": (
            "Task 5: matched evaluation of the SFT, RLVR and RLAIF policies. The policy family, prompt set, "
            "decoding configuration and length cap are identical across conditions; only the feedback source "
            "used for training differs."
        ),
        "dataset": args.dataset,
        "decoding": {"greedy": bool(args.greedy), "max_new_tokens": int(cfg["math_max_new_tokens"]), "seed": int(cfg["seed"])},
        "policies": {},
    }
    for name, records in responses.items():
        stats = verifier_stats(records)
        stats.pop("per_problem_exact")
        payload["policies"][name] = stats

    if not args.skip_judge:
        judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
        by_problem = {}
        for name, records in responses.items():
            for r in records:
                by_problem.setdefault(r["problem_id"], {})[name] = r

        for name in args.policies:
            if name == "sft" or "sft" not in args.policies:
                continue
            wins = ties = losses = 0
            for pid, entry in by_problem.items():
                if name not in entry or "sft" not in entry:
                    continue
                pref = judge.compare(entry["sft"]["problem"], entry[name]["response"], entry["sft"]["response"])
                wins += pref == "A"
                losses += pref == "B"
                ties += pref == "TIE"
            n = max(1, wins + ties + losses)
            payload["policies"].setdefault(name, {})["pairwise_vs_sft"] = {
                "win": wins, "tie": ties, "loss": losses,
                "win_rate": (wins + 0.5 * ties) / n,
                "judge_tie_rate": ties / n,
                "num_comparisons": wins + ties + losses,
            }
            print(f"  [{name}] pairwise win rate vs sft = {payload['policies'][name]['pairwise_vs_sft']['win_rate']:.3f}")

        # Verifier vs judge agreement on pairs the verifier *separates*.
        agree = disagree = 0
        disagreements = []
        for pid, entry in by_problem.items():
            for (na, ra), (nb, rb) in combinations(entry.items(), 2):
                if ra["_exact"] == rb["_exact"]:
                    continue
                better, worse = (ra, rb) if ra["_exact"] > rb["_exact"] else (rb, ra)
                pref = judge.compare(better["problem"], better["response"], worse["response"])
                hit = pref == "A"
                agree += hit
                disagree += (not hit)
                if not hit:
                    disagreements.append(
                        {
                            "problem_id": pid,
                            "pair": [na, nb],
                            "judge_preference": pref,
                            "verifier_correct": [na if ra["_exact"] > rb["_exact"] else nb],
                            "sft": ra["response"][:400] if na == "sft" or nb == "sft" else None,
                        }
                    )
        total = agree + disagree
        payload["verifier_judge_agreement"] = {
            "definition": (
                "Over response pairs for the same problem where the exact verifier assigns different rewards, "
                "the fraction of pairs the AI pairwise judge prefers the verifier-correct response."
            ),
            "num_pairs": total,
            "agreement": agree / max(1, total),
            "disagreements": disagreements[:20],
        }
        print(f"  verifier/judge agreement on separating pairs = {agree}/{total}")

    out = outdir / f"{args.dataset}_eval.json"
    save_json(out, payload)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()