"""Shared evaluation protocol for the Task 1 DPO conditions.

Every DPO condition (standard, beta forks, length-balanced) is scored with the
same code path so that held-out pairs, decoding settings and averaging
conventions are identical across comparisons.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
)
from common.generation import (
    batch_generate,
    chunked_response_token_logprobs,
    response_sequence_logprobs,
    score_reward_pairs,
)
from common.logging_utils import set_seed
from common.metrics import sample_entropy, sampled_kl
from common.models import reference_mode
from task1_dpo.dpo import dpo_loss


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def fits_max_length(tokenizer, row, max_length, response_budget=8):
    """The released encoder raises when a prompt alone exceeds max_length."""

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
    )
    return len(prompt_ids) + response_budget <= max_length


def filter_fits(tokenizer, rows, max_length):
    """Drop pairs whose prompt cannot fit; return (kept, dropped_count)."""

    kept = [r for r in rows if fits_max_length(tokenizer, r, max_length)]
    return kept, len(rows) - len(kept)


def _sequence_logps(model, tokenizer, rows, max_length, device):
    chosen, rejected = encode_pairs(tokenizer, rows, max_length)
    chosen = _to_device(chosen, device)
    rejected = _to_device(rejected, device)
    return {
        "chosen_logp": _masked_sequence_logp(model, chosen),
        "rejected_logp": _masked_sequence_logp(model, rejected),
    }


def encode_pair_examples(tokenizer, rows, max_length):
    """Raw (ids, response_mask) examples for both responses of every pair."""

    chosen, rejected = [], []
    for row in rows:
        prompt = prompt_messages_from_preference(row)
        yc, yr = preference_responses(row)
        chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
        rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
    return chosen, rejected


def encode_pairs(tokenizer, rows, max_length):
    chosen, rejected = encode_pair_examples(tokenizer, rows, max_length)
    return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)


def chunk_response_tokens(tokenizer, rows, max_length):
    chosen, rejected = encode_pair_examples(tokenizer, rows, max_length)
    return (
        torch.tensor([sum(m) for _, m in chosen], dtype=torch.float32),
        torch.tensor([sum(m) for _, m in rejected], dtype=torch.float32),
    )


def _masked_sequence_logp(model, batch):
    """Sum of response-token log-probabilities under teacher forcing.

    `pad_batch` left-pads, so every response ends at the final position and the whole batch
    shares one right edge. Slicing to the longest response before the fp32 log-softmax avoids
    materialising a [batch, full_length, |V|] fp32 tensor.
    """

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    response_len = int(batch["response_mask"].sum(-1).max().item())
    logits = outputs.logits[:, -(response_len + 1) : -1, :]
    labels = batch["input_ids"][:, -response_len:]
    mask = batch["response_mask"][:, -response_len:]
    logp = F.log_softmax(logits.float(), dim=-1)
    tok = torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)
    del outputs, logits, logp
    return (tok * mask).sum(-1)


@torch.no_grad()
def evaluate_preference_pairs(policy, tokenizer, rows, cfg, batch_size=2, beta=None):
    """Held-out DPO loss / preference accuracy on fixed preference pairs.

    preference_accuracy follows the manual definition m_theta > 0 with
    m_theta = [log pi(y+) - log pi_ref(y+)] - [log pi(y-) - log pi_ref(y-)].
    """

    beta = float(cfg["beta"] if beta is None else beta)
    device = next(policy.parameters()).device
    max_length = int(cfg["max_sequence_length"])

    # The held-out files contain prompts longer than max_sequence_length; encode_prompt_response
    # raises on those, so drop them here and report the count rather than silently changing
    # the evaluated set.
    rows, dropped = filter_fits(tokenizer, rows, max_length)
    if not rows:
        raise ValueError(f"No pair fits max_length={max_length}; raise the config value.")

    totals = {
        "dpo_loss": 0.0,
        "logit_mean": 0.0,
        "policy_margin_mean": 0.0,
        "preference_accuracy": 0.0,
        "raw_preference_accuracy": 0.0,
        "implied_reward_margin_mean": 0.0,
        "chosen_logp_mean": 0.0,
        "rejected_logp_mean": 0.0,
        "chosen_tokens_mean": 0.0,
        "rejected_tokens_mean": 0.0,
        "chosen_longer_fraction": 0.0,
    }
    n = 0
    margins = []
    for chunk in _chunks(rows, batch_size):
        with reference_mode(policy):
            ref = _sequence_logps(policy, tokenizer, chunk, max_length, device)
        pol = _sequence_logps(policy, tokenizer, chunk, max_length, device)
        chosen_tokens = chunk_response_tokens(tokenizer, chunk, max_length)

        loss, diag = dpo_loss(
            pol["chosen_logp"],
            pol["rejected_logp"],
            ref["chosen_logp"],
            ref["rejected_logp"],
            beta,
        )
        implied = (pol["chosen_logp"] - ref["chosen_logp"]) - (
            pol["rejected_logp"] - ref["rejected_logp"]
        )
        totals["dpo_loss"] += float(loss.item()) * len(chunk)
        totals["logit_mean"] += float(diag["logit_mean"].item()) * len(chunk)
        totals["policy_margin_mean"] += float(diag["policy_margin_mean"].item()) * len(chunk)
        totals["preference_accuracy"] += float((implied > 0).float().sum().item())
        totals["raw_preference_accuracy"] += float(
            (pol["chosen_logp"] > pol["rejected_logp"]).float().sum().item()
        )
        totals["implied_reward_margin_mean"] += float(implied.mean().item()) * len(chunk)
        totals["chosen_logp_mean"] += float(pol["chosen_logp"].mean().item()) * len(chunk)
        totals["rejected_logp_mean"] += float(pol["rejected_logp"].mean().item()) * len(chunk)
        totals["chosen_tokens_mean"] += float(chosen_tokens[0].mean().item()) * len(chunk)
        totals["rejected_tokens_mean"] += float(chosen_tokens[1].mean().item()) * len(chunk)
        totals["chosen_longer_fraction"] += float(
            (chosen_tokens[0] > chosen_tokens[1]).float().sum().item()
        )
        margins.extend(implied.tolist())
        n += len(chunk)

    out = {k: v / max(1, n) for k, v in totals.items()}
    out["num_pairs"] = n
    out["num_dropped_too_long"] = dropped
    # Binomial standard error: at n=40 this is about +/-0.08, which bounds how finely the
    # beta conditions can be separated in the report.
    p = out["preference_accuracy"]
    out["preference_accuracy_stderr"] = float((max(p * (1.0 - p), 0.0) / max(1, n)) ** 0.5)
    out["beta"] = beta
    return out


def generate_with_diagnostics(policy, tokenizer, prompts, cfg, batch_size=8, seed=None, max_new_tokens=None, with_reference=True):
    """Generate responses and, per sequence, the KL to the frozen reference and the entropy.

    Decoding is re-seeded per chunk so every DPO condition sees the same samples. The KL and
    entropy are computed inside the chunk loop rather than after merging, because
    `batch_generate` left-pads each chunk to its own longest prompt: `prompt_width` therefore
    differs between chunks and a single merged offset would misalign the logits.
    """

    set_seed(int(cfg["seed"] if seed is None else seed))
    gen = cfg.get("generation", {})
    max_new_tokens = int(max_new_tokens or cfg.get("max_generation_tokens", 256))
    keys = ["responses", "response_lengths", "terminated_with_eos", "truncated", "kl_token_mean", "entropy_token_mean"]
    out = {k: [] for k in keys}

    for chunk in _chunks(prompts, batch_size):
        rollouts = batch_generate(
            policy,
            tokenizer,
            chunk,
            max_prompt_length=int(cfg["max_sequence_length"]),
            max_new_tokens=max_new_tokens,
            temperature=float(gen.get("temperature", 0.7)),
            top_p=float(gen.get("top_p", 0.9)),
            do_sample=bool(gen.get("do_sample", True)),
        )
        if with_reference:
            # batch_generate returns inference-mode tensors, so this pass must be no_grad or
            # autograd refuses to record them.
            with torch.no_grad():
                pol_logp = chunked_response_token_logprobs(
                    policy, rollouts["sequences"], rollouts["attention_mask"], rollouts["prompt_width"]
                )
                with reference_mode(policy):
                    ref_logp = chunked_response_token_logprobs(
                        policy, rollouts["sequences"], rollouts["attention_mask"], rollouts["prompt_width"]
                    )
                mask = rollouts["response_mask"]
                for i in range(len(chunk)):
                    out["kl_token_mean"].append(float(sampled_kl(pol_logp[i:i+1], ref_logp[i:i+1], mask[i:i+1]).item()))
                    out["entropy_token_mean"].append(float(sample_entropy(pol_logp[i:i+1], mask[i:i+1]).item()))
                del pol_logp, ref_logp, mask
        out["responses"].extend(rollouts["responses"])
        out["response_lengths"].extend(rollouts["response_lengths"])
        out["terminated_with_eos"].extend(rollouts["terminated_with_eos"])
        out["truncated"].extend(rollouts["truncated"])
    return out


def aggregate_diagnostics(diag):
    """Mean KL / entropy over the generated responses."""

    def mean(key):
        values = diag.get(key) or [0.0]
        return float(sum(values) / len(values))

    return {"kl_token_mean": mean("kl_token_mean"), "entropy_token_mean": mean("entropy_token_mean")}


def reward_scores(reward_model, reward_tokenizer, prompts, responses):
    return [float(x) for x in score_reward_pairs(reward_model, reward_tokenizer, prompts, responses)]


def mean_std(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def length_stats(lengths):
    import numpy as np

    arr = np.asarray(lengths, dtype=float)
    q25, q50, q75 = (float(x) for x in np.percentile(arr, [25, 50, 75]))
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p25": q25,
        "median": q50,
        "p75": q75,
        "iqr": q75 - q25,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }