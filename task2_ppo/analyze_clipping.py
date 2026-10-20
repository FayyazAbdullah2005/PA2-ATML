from __future__ import annotations

import argparse

import torch

from common.data import encode_prompt_response, load_yaml, pad_batch, prompt_messages_from_preference, repo_path
from common.logging_utils import load_json, save_json
from common.generation import chunked_response_token_logprobs
from common.models import clear_gpu, load_policy, load_tokenizer, load_value_model
from task2_ppo.continue_train import run_ppo, value_slice
from task2_ppo.evaluate import evaluate_policy, load_evaluation_bundle
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards

VALUE_KEYS = ("values", "old_values", "value")
# `effective_terminal_reward` is the name the released rollout cache actually uses. It is the
# reward the PPO surrogate was optimised against: `continue_train.py` subtracts the missing-EOS
# penalty before caching, and that column differs from `raw_terminal_reward` by exactly the
# penalty on precisely the non-terminated rollouts. Prefer it over `raw_...` so the reconstructed
# advantages match training.
REWARD_KEYS = (
    "effective_terminal_reward",
    "task_reward",
    "learned_reward",
    "reward",
    "scores",
)


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def first_present(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def as_vector(value):
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def cached_prompt(row):
    return row.get("prompt_messages") or row.get("messages") or prompt_messages_from_preference(row)


@torch.no_grad()
def reconstruct_batch(cfg, rows, tokenizer, policy, value_model):
    """Rebuild (current logp, old logp, ref logp, mask, values) for the fixed cached batch."""

    device = next(policy.parameters()).device
    out = []
    for row in rows:
        cached_ids = first_present(row, ("input_ids", "sequences", "prompt_response_ids"))
        if cached_ids is not None:
            ids = as_vector(cached_ids).long()
            prompt_width = int(row["prompt_width"])
            response = ids[prompt_width:]
        else:
            ids_list, rmask = encode_prompt_response(
                tokenizer, cached_prompt(row), row["response"],
                int(cfg["max_prompt_length"]) + int(cfg["max_response_length"]),
            )
            ids = torch.tensor(ids_list, dtype=torch.long)
            prompt_width = len(ids) - int(sum(rmask))
            response = torch.tensor(ids[prompt_width:], dtype=torch.long)

        mask = first_present(row, ("response_mask", "mask"))
        if mask is None:
            ref = as_vector(row["ref_logprobs"])
            mask = torch.zeros(max(ref.numel(), response.numel()), dtype=torch.float32)
            mask[: ref.numel()] = 1.0
        else:
            mask = as_vector(mask)

        batch = pad_batch(tokenizer, [(ids.tolist(), mask[: ids.numel()].tolist())])
        batch = {k: v.to(device) for k, v in batch.items()}
        # Avoid materialising a full [sequence, vocabulary] logit tensor.  At the
        # 768-token cap it is needlessly large on the 8 GB target GPU; the shared
        # helper computes exactly the sampled response-token log-probabilities in
        # position chunks.
        current = chunked_response_token_logprobs(
            policy, batch["input_ids"], batch["attention_mask"], prompt_width
        ).squeeze(0)

        n = min(current.numel(), as_vector(row["old_logprobs"]).numel(), mask.numel())
        current, old, ref, mask_t = current[:n].cpu(), as_vector(row["old_logprobs"])[:n], as_vector(row["ref_logprobs"])[:n], mask[:n]

        values = first_present(row, VALUE_KEYS)
        if values is None:
            if value_model is None:
                raise KeyError(
                    "The cached batch stores neither token values nor sequences; pass --value-checkpoint "
                    "so V(s_t) can be reconstructed with the released critic."
                )
            values_full = value_slice(value_model, batch["input_ids"], batch["attention_mask"], prompt_width, current.numel())
            values = values_full.squeeze(0)[:n].cpu()

        reward = first_present(row, REWARD_KEYS)
        entry = {
            "source_index": row.get("source_index"),
            "current": current,
            "old": old,
            "ref": ref,
            "mask": mask_t,
            "values": as_vector(values)[:n],
            "task_reward": None if reward is None else float(reward),
            "terminated": row.get("terminated_with_eos", row.get("terminated")),
        }
        out.append(entry)

    if any(e["task_reward"] is None for e in out):
        missing = [k for k in REWARD_KEYS if k in rows[0]]
        raise KeyError(
            "The cached batch has no terminal learned reward, so the surrogate cannot be reconstructed. "
            f"Reward-like keys found: {missing or 'none'}"
        )
    return out


def cache_geometry(cfg, batch):
    """Immediate effect of epsilon on the importance ratio (advantages held fixed)."""

    gamma = float(cfg["gamma"])
    lam = float(cfg["gae_lambda"])
    beta_kl = float(cfg["kl_beta"])
    shaped, advantages, returns = [], [], []
    for e in batch:
        mask = e["mask"]
        n = mask.numel()
        task_reward = torch.full((1,), e["task_reward"], dtype=torch.float32)
        token_rewards = shaped_rewards(task_reward, e["old"].unsqueeze(0), e["ref"].unsqueeze(0), mask.unsqueeze(0), beta_kl)
        adv, ret = compute_gae(token_rewards, e["values"].unsqueeze(0), mask.unsqueeze(0), gamma=gamma, lam=lam)
        shaped.append(token_rewards.squeeze(0))
        advantages.append(normalize_advantages(adv, mask.unsqueeze(0)).squeeze(0))
        returns.append(ret.squeeze(0))

    results = []
    for eps in cfg["clip_values"]:
        eps = float(eps)
        totals = {"surrogate": 0.0, "unclipped": 0.0, "affected": 0.0, "tokens": 0.0, "abs_log_ratio": 0.0, "max_log_ratio": 0.0}
        for e, adv in zip(batch, advantages):
            n = e["mask"].numel()
            mask = e["mask"]
            if mask.sum() == 0:
                continue
            ratio = torch.exp(e["current"] - e["old"])
            clipped = ratio.clamp(1 - eps, 1 + eps)
            affected = ((ratio < (1 - eps)) | (ratio > (1 + eps))).float() * mask
            total = float(mask.sum())
            totals["surrogate"] += float((torch.minimum(ratio * adv, clipped * adv) * mask).sum())
            totals["unclipped"] += float((ratio * adv * mask).sum())
            totals["affected"] += float(affected.sum())
            totals["tokens"] += total
            totals["abs_log_ratio"] += float(((e["current"] - e["old"]).abs() * mask).sum())
            totals["max_log_ratio"] = max(totals["max_log_ratio"], float((e["current"] - e["old"]).abs().max()))
        results.append(
            {
                "epsilon": eps,
                "clipped_surrogate_mean": totals["surrogate"] / max(1.0, totals["tokens"]),
                "unclipped_surrogate_mean": totals["unclipped"] / max(1.0, totals["tokens"]),
                # By the manual's definition, the clip fraction and affected-token
                # fraction are the same masked set; expose both labels explicitly.
                "clip_fraction": totals["affected"] / max(1.0, totals["tokens"]),
                "affected_token_fraction": totals["affected"] / max(1.0, totals["tokens"]),
                "ratio_mean_abs_log_ratio": totals["abs_log_ratio"] / max(1.0, totals["tokens"]),
                "ratio_max_abs_log_ratio": totals["max_log_ratio"],
                "response_tokens": int(totals["tokens"]),
            }
        )
    return results, advantages


def stability(log):
    """Explicit stability statistics over a fork's update trajectory."""

    if not log:
        return {}
    rewards = [r["learned_reward_mean"] for r in log]
    grads = [r["policy_grad_norm"] for r in log]
    losses = [r["policy_loss"] for r in log]
    diffs = [abs(rewards[i + 1] - rewards[i]) for i in range(len(rewards) - 1)]
    tail = log[-max(1, len(log) // 3) :]
    return {
        "reward_volatility_mean_abs_delta": sum(diffs) / max(1, len(diffs)),
        "policy_loss_range": max(losses) - min(losses),
        "policy_grad_norm_max": max(grads),
        "policy_grad_norm_mean": sum(grads) / len(grads),
        "clip_fraction_mean": sum(r["clip_fraction"] for r in log) / len(log),
        "kl_last_third_mean": sum(r["kl_token_mean"] for r in tail) / len(tail),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--epsilons", type=float, nargs="*")
    ap.add_argument("--fork-updates", type=int)
    ap.add_argument("--eval-limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--value-checkpoint", default=None)
    ap.add_argument("--skip-forks", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.epsilons:
        cfg["clip_values"] = list(args.epsilons)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Cache keys:", sorted(rows[0].keys()))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    needs_value = first_present(rows[0], VALUE_KEYS) is None
    value_model = load_value_model(cfg, args.value_checkpoint or cfg["paths"]["ppo_midpoint_value"], train_mode="frozen") if needs_value else None
    batch = reconstruct_batch(cfg, rows, tokenizer, policy, value_model)
    geometry, _ = cache_geometry(cfg, batch)
    del policy, value_model
    clear_gpu()

    for g in geometry:
        print(
            f"eps={g['epsilon']:<5} surrogate={g['clipped_surrogate_mean']:.4f} "
            f"affected={g['affected_token_fraction']:.4f} mean|log ratio|={g['ratio_mean_abs_log_ratio']:.4f}"
        )

    forks = []
    if not args.skip_forks:
        fork_updates = int(args.fork_updates or cfg["fork_updates"])
        for eps in cfg["clip_values"]:
            tag = f"clip_{float(eps):g}"
            adapter = f"outputs/task2_ppo/{tag}"
            fork_path = repo_path(cfg["results_dir"]) / f"{tag}_fork.json"
            existing = load_json(fork_path) if fork_path.exists() else None
            if existing and existing.get("heldout", {}).get("num_prompts") == 200:
                print(f"[skip] {tag}: reusing completed full-set fork", flush=True)
                forks.append(existing)
                continue

            summary_path = repo_path(cfg["results_dir"]) / f"{tag}_summary.json"
            if summary_path.exists():
                print(f"[reuse] {tag}: training is complete; re-running full held-out evaluation", flush=True)
                summary = load_json(summary_path)
            else:
                print(f"=== epsilon={eps}: {fork_updates}-update matched fork ===", flush=True)
                summary = run_ppo(args.config, output=adapter, updates=fork_updates, clip_epsilon=float(eps), run_name=tag, resume=args.resume)
            bundle = load_evaluation_bundle(args.config, adapter)
            bundle["adapter"] = adapter
            heldout = evaluate_policy(bundle, tag, adapter=adapter, limit=args.eval_limit)
            bundle.clear()
            clear_gpu()
            fork = {
                "condition": tag,
                "clip_epsilon": float(eps),
                "adapter": adapter,
                "heldout": heldout,
                "stability": stability(summary["updates_log"]),
            }
            # Persist each fork immediately: a later crash must not discard completed conditions.
            save_json(fork_path, fork)
            save_json(repo_path(cfg["results_dir"]) / "clipping_partial.json",
                      {"cache_geometry": geometry, "forks": forks + [fork]})
            forks.append(fork)
            print(f"[done] {tag}: heldout_reward={heldout['reward_model_mean']:.3f}", flush=True)

    payload = {
        "description": (
            "Task 2 Step 2: clipping study. The cached rollout batch isolates the immediate geometric effect "
            "of epsilon on the importance ratio; matched short forks from the same midpoint test whether that "
            "translates into different optimization behaviour under identical reward/KL settings."
        ),
        "cached_rows": len(rows),
        "cache_geometry": geometry,
        "fork_updates": int(args.fork_updates or cfg["fork_updates"]),
        "kl_beta_held_fixed": float(cfg["kl_beta"]),
        "forks": forks,
    }
    out = repo_path(cfg["results_dir"]) / "clipping_study.json"
    save_json(out, payload)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
