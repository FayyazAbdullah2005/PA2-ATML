from __future__ import annotations

import argparse
import json
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import save_json, set_seed
from common.models import (
    clear_gpu,
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def fits_max_length(tokenizer, row, max_length, response_budget=8):
    """The released encoder raises when a prompt alone exceeds max_length.

    Drop such rows up front instead of crashing mid-epoch, and record the count.
    """

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
    )
    return len(prompt_ids) + response_budget <= max_length


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None, output: str | None = None, resume: bool = False):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)

    tokenizer = load_tokenizer(cfg["base_model"])
    max_length = int(cfg["max_sequence_length"])
    before = len(rows)
    rows = [r for r in rows if fits_max_length(tokenizer, r, max_length)]
    dropped = before - len(rows)
    if dropped:
        print(f"dropped {dropped} rows whose prompt does not fit max_length={max_length}")

    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, max_length),
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    state = torch.load(repo_path(output) / "trainer_state.pt", map_location="cpu", weights_only=False) if (
        resume and (repo_path(output) / "trainer_state.pt").exists()
    ) else None
    if state is not None:
        model.load_state_dict(state["policy"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        print(f"resuming from micro-batch {state['micro_batches']} ({repo_path(output) / 'trainer_state.pt'})", flush=True)
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped_rows": dropped,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
        "resume_micro": 0 if state is None else int(state["micro_batches"]),
    }


def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _micro_step(model, chosen, rejected, beta):
    """One DPO forward/backward micro-batch; returns (loss, diagnostics, chosen_len, rejected_len)."""

    with torch.no_grad():
        with reference_mode(model):
            ref_chosen_logp, _, _ = response_sequence_logprobs(model, chosen)
            ref_rejected_logp, _, _ = response_sequence_logprobs(model, rejected)

    policy_chosen_logp, chosen_tok, chosen_mask = response_sequence_logprobs(model, chosen)
    policy_rejected_logp, rejected_tok, rejected_mask = response_sequence_logprobs(model, rejected)

    loss, diag = dpo_loss(
        policy_chosen_logp,
        policy_rejected_logp,
        ref_chosen_logp,
        ref_rejected_logp,
        beta,
    )
    diag = {k: float(v.item()) for k, v in diag.items()}
    diag["implied_reward_margin"] = float(
        beta * ((policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp)).mean().item()
    )
    diag["raw_preference_accuracy"] = float(
        (policy_chosen_logp > policy_rejected_logp).float().mean().item()
    )
    diag["chosen_logp_mean"] = float(policy_chosen_logp.mean().item())
    diag["rejected_logp_mean"] = float(policy_rejected_logp.mean().item())
    return loss, diag, chosen_mask.sum(-1).mean(), rejected_mask.sum(-1).mean()


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
    results_dir: str | None = None,
    resume: bool = False,
):
    cfg = load_yaml(config_path)
    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples, str(output), resume)
    cfg = bundle["cfg"]
    model = bundle["model"]
    optimizer = bundle["optimizer"]
    loader = bundle["loader"]
    device = next(model.parameters()).device
    beta = bundle["beta"]

    log_path = repo_path(results_dir or cfg["results_dir"]) / f"{run_name}_train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    grad_accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg["epochs"])
    max_grad_norm = float(cfg["max_grad_norm"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    micro = 0
    opt_step = 0
    window = []
    history = []
    start = time.perf_counter()
    skip = int(bundle.get("resume_micro", 0))
    total_micro = len(loader) * epochs

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        last = len(loader) - 1
        for batch_idx, (chosen_raw, rejected_raw) in enumerate(loader):
            micro += 1
            if micro <= skip:
                # Already applied before the crash: redo it only to rebuild the RNG/loader
                # position, but do not accumulate gradients for it.
                continue

            chosen = _move(chosen_raw, device)
            rejected = _move(rejected_raw, device)

            loss, diag, chosen_len, rejected_len = _micro_step(model, chosen, rejected, beta)
            (loss / grad_accum).backward()
            acc = {
                "loss": float(loss.item()),
                "chosen_tokens": float(chosen_len.item()),
                "rejected_tokens": float(rejected_len.item()),
                **{k: diag[k] for k in diag},
            }

            is_boundary = (micro % grad_accum == 0) or micro >= total_micro or batch_idx == last
            if not is_boundary:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters(model), max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            window.append(acc)
            record = {k: _mean([w[k] for w in window]) for k in acc}
            record.update(
                {
                    "step": opt_step,
                    "epoch": epoch + 1,
                    "micro_batches": micro,
                    "grad_norm": float(grad_norm.item()),
                    "learning_rate": float(cfg["learning_rate"]),
                    "beta": beta,
                    "wall_time_s": round(time.perf_counter() - start, 3),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4) if torch.cuda.is_available() else 0.0,
                }
            )
            history.append(record)
            window.clear()
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"[{run_name}] step {opt_step} "
                + " ".join(f"{k}={record[k]:.4f}" for k in ("loss", "preference_accuracy", "implied_reward_margin"))
                + f" grad_norm={record['grad_norm']:.3f} t={record['wall_time_s']:.1f}s",
                flush=True,
            )

            # Checkpoint periodically so an allocator failure costs one interval, not the epoch.
            if opt_step % 10 == 0:
                torch.save(
                    {
                        "micro_batches": micro,
                        "policy": {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k},
                        "optimizer": optimizer.state_dict(),
                    },
                    output / "trainer_state.pt",
                )

    model.save_pretrained(str(output))
    summary = {
        "run_name": run_name,
        "config": str(config_path),
        "dataset": str(dataset_path or cfg["paths"]["dpo_standard_train"]),
        "adapter": str(output.relative_to(repo_path("."))) if output.is_relative_to(repo_path(".")) else str(output),
        "beta": beta,
        "num_examples": len(bundle["rows"]),
        "dropped_long_prompt_rows": bundle["dropped_rows"],
        "epochs": epochs,
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": grad_accum,
        "learning_rate": float(cfg["learning_rate"]),
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "seed": int(cfg["seed"]),
        "optimizer_steps": opt_step,
        "micro_batches": micro,
        "final": history[-1] if history else None,
        "wall_time_s": round(time.perf_counter() - start, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4) if torch.cuda.is_available() else 0.0,
    }
    save_json(log_path.parent / f"{run_name}_train_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("run_name", "adapter", "optimizer_steps", "wall_time_s", "peak_vram_gb")}, indent=2))

    # The released clear_gpu helper only drops its own argument name, so drop the
    # caller's references explicitly before emptying the cache: subsequent forks
    # must start from a clean allocator state on a single 8 GB GPU.
    del model, optimizer, loader, bundle
    clear_gpu()
    return summary


def _mean(values):
    return float(sum(values) / max(1, len(values)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--resume", action="store_true", help="continue from output/trainer_state.pt if present")
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples, resume=args.resume)


if __name__ == "__main__":
    main()