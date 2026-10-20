from __future__ import annotations

import argparse
import gc
import json
import time

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    chunked_response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def report_vram(stage: str) -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    used = total - free
    print(
        f"[vram] {stage}: {used / 2**30:.2f} / {total / 2**30:.2f} GB used by this process and everything else "
        f"({free / 2**30:.2f} GB free). Anything else holding VRAM (a game, a browser) will starve the 8 GB card.",
        flush=True,
    )


def prepare_ppo_continuation(config_path: str, checkpoint_critic: bool = True):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    # The frozen reward model loads with device_map="auto", which silently falls back to CPU
    # when the card is already full and then raises a confusing quantization error. Load it
    # first, while the GPU is empty, so it always lands on the GPU.
    report_vram("before loading the reward model")
    reward_model, reward_tokenizer = load_reward_model(cfg)

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    upcast_trainable_(value_model)
    if checkpoint_critic and hasattr(value_model, "gradient_checkpointing_enable"):
        # The critic is only 0.5B but its un-checkpointed backward over a 768-token rollout
        # costs ~1.4 GB of activations, which on an 8 GB card is the difference between a
        # 46-minute run and one that dies in the allocator after 14 updates.
        try:
            value_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            value_model.gradient_checkpointing_enable()
        if hasattr(value_model, "enable_input_require_grads"):
            value_model.enable_input_require_grads()
    report_vram("after loading policy + critic")
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def upcast_trainable_(value_model):
    """Put every trainable critic parameter in fp32 before the optimizer sees it.

    PEFT's autocast_adapter_dtype upcasts the LoRA weights but leaves the
    `modules_to_save` scalar head in the base dtype. With the released config that
    head is fp16, and a single AdamW step takes it to NaN (its squared-gradient
    accumulator leaves the usable fp16 range), which then produces NaN values at
    every position on the next forward. The loss was finite only because the
    preceding `sanitize` call hid it.
    """

    for module in value_model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()
    # The ModulesToSaveWrapper also keeps a frozen `original_module` copy; it has to
    # move too or the fp32 input hits an fp16 matmul.
    for attr in ("score", "classifier"):
        head = getattr(value_model, attr, None)
        if isinstance(head, torch.nn.Module):
            head.float()
    return value_model


def critic_values(value_model, sequences, attention_mask, prompt_width, n_response):
    """Same computation as common.models.token_values, but with an fp32 value head.

    `base_model_prefix` resolves to the whole Qwen2ForSequenceClassification, whose own
    forward already applies `self.score` internally; calling the inner transformer instead
    avoids that duplicate matmul and lets the head stay fp32 while the backbone stays fp16.
    """

    classifier = getattr(value_model, value_model.base_model_prefix)
    transformer = getattr(classifier, "model", classifier)
    outputs = transformer(
        input_ids=sequences,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        hidden = outputs.hidden_states[-1]
    head = value_model.score if hasattr(value_model, "score") else value_model.classifier
    values = head(hidden[:, prompt_width - 1 : prompt_width - 1 + n_response, :].float())
    return values.squeeze(-1).float()


def finite_stats(*tensors):
    """Per-tensor finiteness and range, logged so a NaN is attributable to a stage."""

    out = {}
    for name, t in tensors:
        t = t.detach().float()
        bad = int((~torch.isfinite(t)).sum().item())
        finite = t[torch.isfinite(t)]
        out[f"{name}_nonfinite"] = bad
        if bad == 0 and finite.numel():
            out[f"{name}_absmax"] = float(finite.abs().max().item())
    return out


def value_slice(value_model, sequences, attention_mask, prompt_width, n_response):
    """V(s_t) for each response step: the value of the prefix that produced it."""

    return critic_values(value_model, sequences, attention_mask, prompt_width, n_response)


def sanitize(values, mask, fill=0.0):
    """Zero out positions that do not participate in the loss and drop any non-finite entry."""

    clean = torch.where(mask.bool(), values, torch.full_like(values, fill))
    return torch.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard", resume: bool = False):
    bundle = prepare_ppo_continuation(config_path)
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
    value_model = bundle["value_model"]
    tokenizer = bundle["tokenizer"]
    policy_opt = bundle["policy_optimizer"]
    value_opt = bundle["value_optimizer"]
    pool = bundle["prompt_rows"]
    device = next(policy.parameters()).device
    eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])
    gamma = float(cfg["gamma"])
    lam = float(cfg["gae_lambda"])
    value_coef = float(cfg["value_coef"])
    eos_penalty = float(cfg["missing_eos_penalty"])
    max_resp = int(cfg["max_response_length"])
    gen = cfg.get("generation", {})
    log_path = repo_path(cfg["results_dir"]) / f"{run_name}_updates.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_update = 0
    state_path = out / "trainer_state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        missing = value_model.load_state_dict(state["critic"], strict=False)
        value_opt.load_state_dict(state["value_optimizer"])
        policy_opt.load_state_dict(state["policy_optimizer"])
        start_update = int(state["update"])
        print(f"resuming from update {start_update} ({state_path})", flush=True)

    if start_update == 0:
        log_path.write_text("", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = []

    for update in range(start_update, int(cfg["updates"])):
        update_start = time.perf_counter()
        rows = [pool[(update * int(cfg["prompts_per_update"]) + i) % len(pool)] for i in range(int(cfg["prompts_per_update"]))]
        prompts = [prompt_messages(r) for r in rows]

        # ---- rollout collection (on-policy, from the current policy) ----
        set_seed(int(cfg["seed"]) + update)
        rollouts = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_resp,
            temperature=float(gen.get("temperature", 0.7)),
            top_p=float(gen.get("top_p", 0.9)),
            do_sample=bool(gen.get("do_sample", True)),
        )
        # batch_generate runs under torch.inference_mode, so its tensors are inference tensors
        # that autograd refuses to record. Clone them into normal tensors before the
        # optimization passes reuse the same rollout.
        sequences = rollouts["sequences"].clone()
        attention_mask = rollouts["attention_mask"].clone()
        prompt_width = rollouts["prompt_width"]
        response_ids = rollouts["response_ids"].clone()
        mask = rollouts["response_mask"].clone()

        # ---- learned task reward, with the missing-EOS penalty ----
        task_rewards = score_reward_pairs(
            bundle["reward_model"],
            bundle["reward_tokenizer"],
            prompts,
            rollouts["responses"],
            max_length=int(cfg["reward_max_length"]),
        )
        task_rewards = task_rewards.detach().to(device=device, dtype=torch.float32)
        terminated = torch.tensor(rollouts["terminated_with_eos"], dtype=torch.bool, device=device)
        task_rewards = task_rewards - eos_penalty * (~terminated).float()

        # ---- old-policy and reference log-probs on the sampled tokens ----
        with torch.no_grad():
            old_logp = chunked_response_token_logprobs(policy, sequences, attention_mask, prompt_width)
            with reference_mode(policy):
                ref_logp = chunked_response_token_logprobs(policy, sequences, attention_mask, prompt_width)
            old_values_raw = value_slice(value_model, sequences, attention_mask, prompt_width, response_ids.shape[1])

        token_rewards = shaped_rewards(task_rewards, old_logp, ref_logp, mask, beta_kl)
        old_values = sanitize(old_values_raw, mask)
        advantages, returns = compute_gae(token_rewards, old_values, mask, gamma=gamma, lam=lam)
        returns = sanitize(returns, mask)
        adv_norm = sanitize(normalize_advantages(advantages, mask), mask)
        diagnostics = finite_stats(
            ("task_reward", task_rewards),
            ("old_logp", old_logp),
            ("ref_logp", ref_logp),
            ("token_rewards", token_rewards),
            ("old_values_raw", old_values_raw),
            ("advantages", advantages),
            ("returns", returns),
        )

        # ---- PPO optimization over `ppo_epochs` passes on the same rollout batch ----
        policy_losses, value_losses, clip_fracs, entropies = [], [], [], []
        policy_grad_norms, value_grad_norms = [], []
        critic_skipped = 0
        for _ in range(int(cfg["ppo_epochs"])):
            new_logp = chunked_response_token_logprobs(policy, sequences, attention_mask, prompt_width, with_grad=True)
            new_values = value_slice(value_model, sequences, attention_mask, prompt_width, response_ids.shape[1])
            policy_loss, ratio, clip_fraction = ppo_policy_loss(new_logp, old_logp, adv_norm, mask, eps)
            v_loss = value_mse_loss(sanitize(new_values, mask), returns, mask)
            diagnostics.update(
                finite_stats(("new_values", new_values), ("new_logp", new_logp), ("policy_loss", policy_loss))
            )

            # One backward pass over the summed objective, as in the released design. The critic
            # term is dropped when it is non-finite: a NaN in value_mse_loss would otherwise
            # flow into the shared graph and turn every actor gradient into NaN, which is
            # exactly how the first smoke run destroyed the policy adapter on update 2.
            v_loss_finite = bool(torch.isfinite(v_loss))
            loss = policy_loss + value_coef * v_loss if v_loss_finite else policy_loss

            policy_opt.zero_grad(set_to_none=True)
            value_opt.zero_grad(set_to_none=True)
            loss.backward()
            p_gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            v_params = [p for g in value_opt.param_groups for p in g["params"]]
            v_gn = torch.nn.utils.clip_grad_norm_(v_params, float(cfg["max_grad_norm"]))
            policy_opt.step()
            if v_loss_finite:
                value_opt.step()
            else:
                critic_skipped += 1

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(v_loss.item()))
            clip_fracs.append(float(clip_fraction.item()))
            entropies.append(float(sample_entropy(new_logp.detach(), mask).item()))
            policy_grad_norms.append(float(p_gn.item()))
            value_grad_norms.append(float(v_gn.item()))

            # A non-finite actor loss or gradient invalidates the run, so abort rather than
            # saving a poisoned adapter. A non-finite critic value loss is only skipped
            # (its step is dropped above) and reported, since the supplied midpoint critic
            # is documented as imperfectly calibrated.
            if not (torch.isfinite(policy_loss) and torch.isfinite(p_gn)):
                raise SystemExit(
                    f"[FATAL] non-finite PPO policy update at update {update + 1} "
                    f"(policy_loss={float(policy_loss.item())}, policy_grad_norm={float(p_gn.item())}).\n"
                    f"Stage diagnostics: {json.dumps(diagnostics, indent=2)}\n"
                    "Adapter NOT saved."
                )
            if not v_loss_finite:
                print(
                    f"[WARN] update {update + 1}: value_loss was {float(v_loss.item())}; critic step skipped.\n"
                    f"  diagnostics: {json.dumps(diagnostics)}",
                    flush=True,
                )

        record = {
            "update": update + 1,
            "run_name": run_name,
            "prompt_source_indices": [r.get("source_index", r.get("prompt_id", i)) for i, r in enumerate(rows)],
            "learned_reward_mean": float(task_rewards.mean().item()),
            "learned_reward_raw_mean": float((task_rewards + eos_penalty * (~terminated).float()).mean().item()),
            "kl_token_mean": float(sampled_kl(old_logp, ref_logp, mask).item()),
            "kl_sequence_sum_mean": float(((old_logp - ref_logp) * mask).sum(-1).mean().item()),
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "clip_fraction": sum(clip_fracs) / len(clip_fracs),
            "entropy_token_mean": sum(entropies) / len(entropies),
            "policy_grad_norm": sum(policy_grad_norms) / len(policy_grad_norms),
            "value_grad_norm": sum(value_grad_norms) / len(value_grad_norms),
            "advantage_mean": float(adv_norm[mask.bool()].mean().item()) if mask.sum() else 0.0,
            "advantage_std": float(adv_norm[mask.bool()].std(unbiased=False).item()) if mask.sum() > 1 else 0.0,
            "return_mean": float(returns[mask.bool()].mean().item()) if mask.sum() else 0.0,
            "value_mean": float(old_values[mask.bool()].mean().item()) if mask.sum() else 0.0,
            "response_length_mean": mean_response_length(mask),
            "eos_termination_rate": float(terminated.float().mean().item()),
            "clip_epsilon": eps,
            "kl_beta": beta_kl,
            "update_wall_time_s": round(time.perf_counter() - update_start, 3),
            "wall_time_s": round(time.perf_counter() - start, 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4) if torch.cuda.is_available() else 0.0,
            "critic_steps_skipped": critic_skipped,
            **diagnostics,
        }
        history.append(record)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[{run_name}] update {record['update']} reward={record['learned_reward_mean']:.3f} "
            f"kl={record['kl_token_mean']:.4f} pi_loss={record['policy_loss']:.4f} "
            f"v_loss={record['value_loss']:.4f} clip={record['clip_fraction']:.3f} "
            f"len={record['response_length_mean']:.1f} t={record['wall_time_s']:.1f}s",
            flush=True,
        )

        # Checkpoint every update so a late allocator failure costs one update, not the run.
        policy.save_pretrained(str(out))
        torch.save(
            {
                "update": update + 1,
                "critic": {k: v.detach().cpu() for k, v in value_model.state_dict().items()
                           if ("lora_" in k or "score" in k)},
                "policy_optimizer": policy_opt.state_dict(),
                "value_optimizer": value_opt.state_dict(),
            },
            out / "trainer_state.pt",
        )

        del new_logp, new_values, policy_loss, v_loss, rollouts
        del sequences, attention_mask, response_ids, mask, token_rewards
        del old_logp, ref_logp, old_values, old_values_raw, advantages, returns, adv_norm, task_rewards
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy.save_pretrained(str(out))
    for name, p in policy.named_parameters():
        if p.requires_grad and not torch.isfinite(p).all():
            raise SystemExit(f"[FATAL] non-finite trained parameter {name}; refusing to save the adapter.")
    summary = {
        "run_name": run_name,
        "config": str(config_path),
        "adapter": str(out.relative_to(repo_path("."))) if out.is_relative_to(repo_path(".")) else str(out),
        "updates": int(cfg["updates"]),
        "ppo_epochs": int(cfg["ppo_epochs"]),
        "prompts_per_update": int(cfg["prompts_per_update"]),
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "gamma": gamma,
        "gae_lambda": lam,
        "value_coef": value_coef,
        "missing_eos_penalty": eos_penalty,
        "policy_learning_rate": float(cfg["policy_learning_rate"]),
        "value_lora_learning_rate": float(cfg["value_lora_learning_rate"]),
        "value_head_learning_rate": float(cfg["value_head_learning_rate"]),
        "value_train_mode": cfg.get("value_train_mode"),
        "max_response_length": max_resp,
        "seed": int(cfg["seed"]),
        "log": str(log_path.relative_to(repo_path("."))),
        "updates_log": history,
        "wall_time_s": round(time.perf_counter() - start, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4) if torch.cuda.is_available() else 0.0,
    }
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_summary.json", summary)
    print(f"saved adapter -> {out}")

    del policy, value_model, policy_opt, value_opt, bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--resume", action="store_true", help="continue from output/trainer_state.pt if present")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name, args.resume)


if __name__ == "__main__":
    main()