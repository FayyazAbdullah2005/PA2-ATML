# ATML PA2 - LLM Post-Training


## Quick start

```bash
git clone https://github.com/FayyazAbdullah2005/PA2-ATML.git
cd PA2-ATML
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/FayyazAbdullah2005/PA2-ATML.git
cd PA2-ATML
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Validate the released objectives

The release ships one deliberate defect in each core objective (Tasks 1–3). Corrected
before any training run, and checked against the manual equations:

```bash
python -m scripts.verify_objectives
```

| Task | Released form | Correct form | Why the released form is wrong |
| --- | --- | --- | --- |
| 1 — DPO | `beta * (policy_margin + ref_margin)` | `beta * (policy_margin - ref_margin)` | DPO models the *change* in relative likelihood induced by the policy, so the reference log-ratio is subtracted. With `+` the loss is `-log sigma(2*beta*ref_margin)` at initialisation instead of `log 2`, i.e. the objective does not vanish at `theta = theta_ref` and it rewards moving away from the reference. |
| 2 — PPO | `torch.maximum(surr1, surr2)` | `torch.minimum(surr1, surr2)` | Schulman et al. require a pessimistic lower bound, `min(rho A, clip(rho) A) <= rho A`. `max` is an optimistic upper bound: an out-of-trust-region move with negative advantage saturates at `clip(rho) A` and stops being penalised, so the policy can keep taking steps the trust region was meant to forbid. |
| 3 — GRPO | batch-wide `rewards.mean()/std()` | per-`group_ids` mean/std | The group-relative baseline is defined within a prompt group. A batch-wide statistic (a) lets one prompt's reward level shift another prompt's advantage and (b) gives *non-zero* advantages to groups whose rewards are all identical whenever some other prompt varies — exactly the uninformative groups the critic-free estimator is meant to discard. |

## 6. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 7. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

## 8. Implementation notes

- **Single-GPU memory.** Runs were produced on one 8 GB RTX 5060 Laptop GPU. Frozen
  reward/judge models load in 8-bit/4-bit (`quantize_frozen_models`), policies use fp16
  with gradient checkpointing, and runs are executed sequentially with an explicit
  `gc.collect()` + `torch.cuda.empty_cache()` between conditions. `task3_grpo` computes
  response-token log-probabilities by applying the LM head in position chunks with
  `torch.utils.checkpoint` (`response_token_logprobs_chunked`), because a K=4 x 768-token
  rollout otherwise materialises a full `[4, 768, 151936]` logit tensor on top of the
  policy and the frozen reward model. This is numerically identical to the full-sequence
  computation, just not simultaneous in memory.
- **Prompt selection.** PPO/GRPO draw prompts from `rl_prompt_pool_train` in fixed file
  order (update `u` uses row `u mod len(pool)`), and the RNG is re-seeded as
  `seed + u` before each rollout, so every matched fork sees the same prompts and the same
  samples at the same update index.
- **Critic dtype (Task 2).** `load_value_model(..., train_mode="lora_head")` wraps the
  released merged critic with a fresh LoRA adapter and puts the scalar head in PEFT's
  `modules_to_save`. PEFT's `autocast_adapter_dtype` upcasts the LoRA weights to fp32 but
  leaves that head fp16, so `AdamW` updated one fp16 parameter: a single step took
  `score.modules_to_save.default.weight` to NaN, after which every critic value was NaN and
  the value loss was NaN from update 2 onward. `upcast_trainable_()` moves the trainable
  critic parameters (and the head's frozen `original_module` copy) to fp32, and
  `critic_values()` applies the head to an fp32 slice of the last hidden state while the
  backbone stays fp16. It also calls the inner transformer rather than
  `base_model_prefix`, which resolves to the whole `Qwen2ForSequenceClassification` and
  would apply the score head internally a second time. This is a dtype fix, not a change
  to the released PPO objective.
- **KL convention.** Task 1 and Task 3 report the released sampled-response estimator
  (`common.metrics.sampled_kl`, a masked token-level mean of `log pi - log pi_ref`)
  everywhere, plus a sequence-sum variant alongside it; the convention is never switched
  between conditions.

## 9. Attribution

- The course starter repository `ATML-PA2-LLM-PostTraining` supplies `common/`, `configs/`,
  `scripts/`, the AI safety judge, the exact RLVR verifier, and the pairwise AI judge. The
  released helpers (`common.generation`, `common.metrics`, `common.models`,
  `task2_ppo.ppo.compute_gae/shaped_rewards`, `task3_grpo.grpo.grpo_policy_loss`,
  `task4_safety.judge_responses.judge_one`, `task5_feedback.rlvr.exact_reward`,
  `task5_feedback.rlaif.PairwiseAIJudge`) are used unmodified where they were correct.
- `task1_dpo/protocol.py` is new: it holds the shared Task 1 evaluation protocol so that the
  standard run, the beta forks and the length-balanced condition are scored by one code path.
- The training loops, ablation orchestration, aggregation and analysis in `task1_dpo/`,
  `task2_ppo/`, `task3_grpo/`, `task4_safety/` and `task5_feedback/`, plus
  `scripts/verify_objectives.py`, are written for this assignment.
- Libraries: PyTorch, Hugging Face `transformers` / `peft` / `bitsandbytes` (model loading,
  LoRA, quantization), `numpy`, `pandas`, `pyyaml`. No external code was copied from other
  repositories; the objective implementations follow the equations in the course readings
  (Rafailov et al. 2023; Schulman et al. 2017; Shao et al. 2024; Liu et al. 2025).

See the assignment manual for the required experiments, metrics, and report questions.
