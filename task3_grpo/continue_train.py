from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    chunked_response_token_logprobs as response_token_logprobs_chunked,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for position, i in enumerate(order):
            r[i] = float(position)
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return float(num / den) if den else 0.0


def length_gradient_statistics(new_logp, old_logp, mask, advantages, lengths, loss_type, max_completion_length, eps):
    """How much gradient magnitude each completion receives under the chosen normalization.

    The per-completion gradient magnitude is |w_k * sum_t objective_k,t| with the real
    importance ratio rho = exp(log pi_new - log pi_old) and the clipped objective, where w_k is
    the sequence normalization (1/T_k for canonical GRPO, 1/T_max for Dr. GRPO). Its share of
    the group total is what decides how short versus long completions compete.

    Must be called with the *post-update* log-probs. Passing the pre-update logits for both
    arguments collapses rho to exactly 1 and the statistic degenerates to |A| alone.
    """

    with torch.no_grad():
        new_logp = torch.nan_to_num(new_logp, nan=0.0, posinf=0.0, neginf=0.0)
        old_logp = torch.nan_to_num(old_logp, nan=0.0, posinf=0.0, neginf=0.0)
        ratio = torch.exp(new_logp - old_logp)
        adv = advantages[:, None]
        objective = torch.minimum(ratio * adv, ratio.clamp(1.0 - eps, 1.0 + eps) * adv)
        if loss_type == "grpo":
            weights = 1.0 / lengths.clamp_min(1.0)
        else:
            weights = torch.full_like(lengths, 1.0 / float(max_completion_length))
        seq_grad = (objective * mask).sum(-1).abs() * weights
        if float(seq_grad.sum()) <= 0:
            return {}
        share = (seq_grad / seq_grad.sum()).tolist()
        order = torch.argsort(lengths).tolist()
        longest, shortest = share[order[-1]], share[order[0]]
        return {
            "gradient_share_longest": longest,
            "gradient_share_shortest": shortest,
            # None, not a sentinel: a zero advantage makes the ratio undefined, and a huge
            # finite number would poison the first-third/last-third averages.
            "gradient_share_ratio_longest_over_shortest": (longest / shortest) if shortest > 1e-9 else None,
            "spearman_length_vs_gradient_share": spearman(lengths.tolist(), share),
            "response_length_spread": float(lengths[order[-1]] - lengths[order[0]]),
            "per_token_weight_ratio_long_over_short": (
                float(lengths[order[0]] / lengths[order[-1]]) if loss_type == "grpo" else 1.0
            ),
        }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", clip_epsilon: float | None = None, kl_beta: float | None = None):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    optimizer = bundle["optimizer"]
    pool = bundle["prompt_rows"]
    device = next(policy.parameters()).device
    K = int(cfg["num_generations"])
    eps = float(cfg["clip_epsilon"])
    beta = float(cfg["kl_beta"])
    max_completion = int(cfg["max_completion_length"])
    per_update = int(cfg["prompts_per_update"])
    gen = cfg.get("generation", {})
    log_path = repo_path(cfg["results_dir"]) / f"{run_name}_updates.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = []

    for update in range(int(cfg["updates"])):
        update_start = time.perf_counter()
        rows = [pool[(update * per_update + i) % len(pool)] for i in range(per_update)]
        group_prompts = []
        group_ids = []
        for i, r in enumerate(rows):
            group_prompts.extend([prompt_messages(r)] * K)
            group_ids.extend([i] * K)

        set_seed(int(cfg["seed"]) + update)
        rollouts = batch_generate(
            policy,
            tokenizer,
            group_prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_completion,
            temperature=float(gen.get("temperature", 0.7)),
            top_p=float(gen.get("top_p", 0.9)),
            do_sample=bool(gen.get("do_sample", True)),
        )
        # batch_generate runs under torch.inference_mode; clone into normal tensors before
        # the update passes reuse this rollout in autograd-tracked computations.
        sequences = rollouts["sequences"].clone()
        attention_mask = rollouts["attention_mask"].clone()
        prompt_width = rollouts["prompt_width"]
        response_ids = rollouts["response_ids"].clone()
        mask = rollouts["response_mask"].clone()
        if cfg.get("mask_truncated_completions", True):
            mask = mask_truncated_sequences(mask, rollouts["truncated"])

        rewards = score_reward_pairs(
            bundle["reward_model"],
            bundle["reward_tokenizer"],
            group_prompts,
            rollouts["responses"],
            max_length=int(cfg.get("reward_max_length", max_completion + int(cfg["max_prompt_length"]))),
        ).detach().to(device=device, dtype=torch.float32)
        group_ids_t = torch.tensor(group_ids, dtype=torch.long, device=device)
        advantages = group_relative_advantages(rewards, group_ids_t)

        old_logp = response_token_logprobs_chunked(
            policy, sequences, attention_mask, prompt_width, with_grad=False
        )
        with torch.no_grad(), reference_mode(policy):
            ref_logp = response_token_logprobs_chunked(
                policy, sequences, attention_mask, prompt_width, with_grad=False
            )

        lengths = mask.sum(-1)
        grad_stats = {}

        losses, diags, grad_norms = [], [], []
        for epoch_index in range(int(cfg.get("policy_epochs", 1))):
            new_logp = response_token_logprobs_chunked(
                policy, sequences, attention_mask, prompt_width, with_grad=True
            )
            if epoch_index == 0:
                # The gradient-share statistic needs the real post-update importance ratio,
                # so take it from the first optimization pass before the weights move.
                grad_stats = length_gradient_statistics(
                    new_logp.detach(), old_logp, mask, advantages, lengths, loss_type, max_completion, eps
                )
            loss, diag = grpo_policy_loss(
                new_logp,
                old_logp,
                advantages,
                mask,
                ref_logp,
                eps,
                beta,
                loss_type=loss_type,
                max_completion_length=max_completion,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            optimizer.step()
            losses.append(float(loss.item()))
            diags.append({k: float(v.item()) for k, v in diag.items()})
            grad_norms.append(float(gnorm.item()))

        lengths = mask.sum(-1)
        within_group_std = []
        for gid in range(per_update):
            g = rewards[group_ids_t == gid]
            within_group_std.append(float(g.std(unbiased=False).item()))
        informative = float(sum(1 for s in within_group_std if s > 1e-6) / max(1, len(within_group_std)))

        record = {
            "update": update + 1,
            "run_name": run_name,
            "loss_type": loss_type,
            "prompt_source_indices": [r.get("source_index", r.get("prompt_id", i)) for i, r in enumerate(rows)],
            "reward_mean": float(rewards.mean().item()),
            "reward_max": float(rewards.max().item()),
            "reward_min": float(rewards.min().item()),
            "kl_token_mean": float(sampled_kl(old_logp, ref_logp, mask).item()),
            "kl_sequence_sum_mean": float(((old_logp - ref_logp) * mask).sum(-1).mean().item()),
            "within_group_reward_std_mean": sum(within_group_std) / max(1, len(within_group_std)),
            "uninformative_group_fraction": 1.0 - informative,
            "informative_group_fraction": informative,
            "policy_loss": sum(losses) / len(losses),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            "gradient_norm": sum(grad_norms) / len(grad_norms),
            "response_length_mean": float(lengths.float().mean().item()),
            "response_length_max": float(lengths.float().max().item()),
            "num_generations": K,
            "clip_epsilon": eps,
            "kl_beta": beta,
            "truncated_fraction": float(sum(rollouts["truncated"]) / max(1, len(rollouts["truncated"]))),
            "eos_termination_rate": float(sum(rollouts["terminated_with_eos"]) / max(1, len(rollouts["terminated_with_eos"]))),
            "update_wall_time_s": round(time.perf_counter() - update_start, 3),
            "wall_time_s": round(time.perf_counter() - start, 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4) if torch.cuda.is_available() else 0.0,
            **{f"diag_{k}": v for k, v in diags[-1].items()},
            **{f"grad_{k}": v for k, v in grad_stats.items()},
        }
        history.append(record)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[{run_name}] update {record['update']} reward={record['reward_mean']:.3f} "
            f"kl={record['kl_token_mean']:.4f} group_std={record['within_group_reward_std_mean']:.4f} "
            f"uninf={record['uninformative_group_fraction']:.2f} loss={record['policy_loss']:.4f} "
            f"len={record['response_length_mean']:.1f} t={record['wall_time_s']:.1f}s",
            flush=True,
        )

    policy.save_pretrained(str(out))
    summary = {
        "run_name": run_name,
        "config": str(config_path),
        "adapter": str(out.relative_to(repo_path("."))) if out.is_relative_to(repo_path(".")) else str(out),
        "loss_type": loss_type,
        "updates": int(cfg["updates"]),
        "num_generations": K,
        "prompts_per_update": per_update,
        "policy_epochs": int(cfg.get("policy_epochs", 1)),
        "learning_rate": float(cfg["learning_rate"]),
        "clip_epsilon": eps,
        "kl_beta": beta,
        "max_completion_length": max_completion,
        "mask_truncated_completions": bool(cfg.get("mask_truncated_completions", True)),
        "seed": int(cfg["seed"]),
        "log": str(log_path.relative_to(repo_path("."))),
        "updates_log": history,
        "wall_time_s": round(time.perf_counter() - start, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4) if torch.cuda.is_available() else 0.0,
    }
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_summary.json", summary)
    print(f"saved adapter -> {out}")

    del policy, optimizer, bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()