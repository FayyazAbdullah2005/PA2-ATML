"""Validate the three released objectives against the equations in the assignment manual.

Run:  python -m scripts.verify_objectives

The release ships one deliberate defect in each core objective. These checks fail on
the released code and pass on the corrected implementation, so they double as the
unit tests for the fix.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((PASS if condition else FAIL, name, detail))
    print(f"[{PASS if condition else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def test_dpo_objective() -> None:
    torch.manual_seed(0)
    pc = torch.tensor([-2.0, -1.0], requires_grad=True)
    pr = torch.tensor([-3.0, -4.0], requires_grad=True)
    rc = torch.tensor([-2.5, -1.5])
    rr = torch.tensor([-3.5, -4.5])
    beta = 0.1

    loss, diag = dpo_loss(pc, pr, rc, rr, beta)
    margin = (pc - rc) - (pr - rr)
    expected = -F.logsigmoid(beta * margin).mean()
    check(
        "Task 1 DPO: loss equals -log sigmoid(beta * ([pi-rf](y+) - [pi-rf](y-)))",
        torch.allclose(loss, expected, atol=1e-6),
        f"got {loss.item():.6f}, expected {expected.item():.6f}",
    )

    # At initialisation the policy equals the reference, so the margin is 0 and the loss is log 2.
    loss0, _ = dpo_loss(rc.clone(), rr.clone(), rc, rr, beta)
    check(
        "Task 1 DPO: loss is log 2 at theta = theta_ref (the additive form would not be)",
        abs(loss0.item() - math.log(2)) < 1e-6,
        f"got {loss0.item():.6f}",
    )

    grad = torch.autograd.grad(loss, pc, retain_graph=True)[0]
    sigmoid = torch.sigmoid(beta * margin)
    # dpo_loss averages over the batch, so the per-element gradient carries a 1/B factor.
    check(
        "Task 1 DPO: dL/d log pi(y+) equals -beta * sigma(-beta m) / B",
        torch.allclose(grad, -beta * (1 - sigmoid) / margin.numel(), atol=1e-6),
        f"got {grad.tolist()}",
    )


def test_ppo_objective() -> None:
    new = torch.log(torch.tensor([1.0, 3.0, 0.2, 1.0, 1.0]))
    old = torch.log(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
    adv = torch.tensor([1.0, -1.0, -2.0, 1.0, -1.0])
    mask = torch.ones(5)
    eps = 0.2

    loss, ratio, clip_fraction = ppo_policy_loss(new, old, adv, mask, eps)
    r = torch.exp(new - old)
    expected = -(torch.minimum(r * adv, r.clamp(1 - eps, 1 + eps) * adv)).mean()
    check(
        "Task 2 PPO: clipped surrogate uses min(rho A, clip(rho) A)",
        torch.allclose(loss, expected, atol=1e-6),
        f"got {loss.item():.6f}, expected {expected.item():.6f}",
    )

    # Schulman et al. require the clipped objective to be a *lower bound* on the unclipped
    # surrogate, min(rho A, clip(rho) A) <= rho A, so that the value the optimiser reports
    # never over-estimates the improvement. max() gives the opposite (optimistic) bound.
    pessimistic = torch.minimum(r * adv, r.clamp(1 - eps, 1 + eps) * adv)
    optimistic = torch.maximum(r * adv, r.clamp(1 - eps, 1 + eps) * adv)
    check(
        "Task 2 PPO: the clipped objective is a pointwise lower bound on rho * A",
        bool((pessimistic <= r * adv + 1e-6).all()) and bool((optimistic >= r * adv - 1e-6).all()),
    )
    # Consequence for a negative-advantage token whose ratio blew past 1+eps: min() keeps the
    # penalty growing with rho, while max() saturates it at (1-eps... )*A and stops penalising.
    bad_move = torch.tensor([5.0]) * torch.tensor([-1.0])
    pen_min = float(torch.minimum(bad_move, bad_move.clamp(1 - eps, 1 + eps)))
    pen_max = float(torch.maximum(bad_move, bad_move.clamp(1 - eps, 1 + eps)))
    check(
        "Task 2 PPO: an out-of-trust-region move with negative advantage keeps being penalised "
        "(min), whereas max saturates the penalty",
        pen_min < pen_max - 1e-6,
        f"rho=5, A=-1: min={pen_min:.2f}, max={pen_max:.2f}",
    )
    affected = ((r < 1 - eps) | (r > 1 + eps)).float()
    check(
        "Task 2 PPO: clip fraction equals the affected-token fraction under the response mask",
        abs(clip_fraction.item() - affected.mean().item()) < 1e-6,
        f"clip_fraction={clip_fraction.item():.4f}",
    )


def test_gae_and_shaping() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    mask = torch.ones(1, 3)
    advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    # delta = [0, 0, 0.5]; A_t = sum_k (gamma*lam)^k delta_{t+k}
    delta = torch.tensor([0.0, 0.0, 0.5])
    expected = torch.tensor(
        [
            [
                delta[0] + 0.95 * delta[1] + 0.95**2 * delta[2],
                delta[1] + 0.95 * delta[2],
                delta[2],
            ]
        ]
    )
    check(
        "Task 2 PPO: GAE recursion matches the manual definition",
        torch.allclose(advantages, expected, atol=1e-6),
        f"got {advantages.tolist()}",
    )
    check(
        "Task 2 PPO: returns equal advantages + values",
        torch.allclose(returns, advantages + values, atol=1e-6),
    )

    shaped = shaped_rewards(torch.tensor([2.0]), torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1, 3), 0.1)
    check(
        "Task 2 PPO: KL shaping puts the terminal task reward on the last valid response token",
        abs(float(shaped[0, 2]) - 2.0) < 1e-6 and float(shaped[0, 0]) == 0.0,
        f"shaped={shaped.tolist()}",
    )


def test_grpo_advantages() -> None:
    # Group 0 varies, groups 1 and 2 are uninformative (identical rewards inside the group).
    rewards = torch.tensor([1.0, 3.0, 2.0, 10.0, 10.0, 5.0, 5.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    adv = group_relative_advantages(rewards, groups)

    check(
        "Task 3 GRPO: group 0 advantages are normalized within the group (mean 0, unit std)",
        abs(float(adv[groups == 0].mean())) < 1e-6
        and abs(float(adv[groups == 0].std(unbiased=False)) - 1.0) < 1e-3,
        f"got {adv[groups == 0].tolist()}",
    )
    check(
        "Task 3 GRPO: an uninformative group receives exactly zero advantage "
        "(a batch-wide mean/std would give it non-zero values)",
        bool(adv[groups == 1].abs().sum() == 0 and adv[groups == 2].abs().sum() == 0),
        f"group 1 = {adv[groups == 1].tolist()}, group 2 = {adv[groups == 2].tolist()}",
    )
    batch_wide = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
    check(
        "Task 3 GRPO: the released batch-wide normalization would have perturbed the uninformative groups",
        bool(batch_wide[groups == 1].abs().sum() > 0),
        f"batch-wide group-1 advantages = {batch_wide[groups == 1].tolist()}",
    )

    shifted = rewards.clone()
    shifted[groups == 0] += 1000.0
    adv2 = group_relative_advantages(shifted, groups)
    check(
        "Task 3 GRPO: advantages are invariant to a constant shift of another group's rewards",
        torch.allclose(adv2[groups != 0], adv[groups != 0], atol=1e-6),
    )
    widened = rewards.clone()
    widened[groups == 0] = torch.tensor([1.0, 10.0, 2.0])  # a 3-element group exposes a real spread change
    adv3 = group_relative_advantages(widened, groups)
    check(
        "Task 3 GRPO: widening one group's reward spread changes only that group",
        not torch.allclose(adv3[groups == 0], adv[groups == 0], atol=1e-6)
        and torch.allclose(adv3[groups != 0], adv[groups != 0], atol=1e-6),
        f"group 0 = {adv3[groups == 0].tolist()}, others unchanged",
    )
    for gid in range(3):
        r = rewards[groups == gid]
        expected = (r - r.mean()) / r.std(unbiased=False).clamp_min(1e-6)
        check(
            f"Task 3 GRPO: group {gid} matches A_k = (r_k - mu_r) / (sigma_r + eps)",
            torch.allclose(adv[groups == gid], expected, atol=1e-6),
            f"got {adv[groups == gid].tolist()}, expected {expected.tolist()}",
        )


def test_grpo_loss() -> None:
    new = torch.zeros(2, 4)
    old = torch.zeros(2, 4)
    ref = torch.zeros(2, 4)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    eps, beta = 0.2, 0.1

    loss_grpo, diag = grpo_policy_loss(new, old, adv, mask, ref, eps, beta, loss_type="grpo", max_completion_length=4)
    ratio = torch.exp(new - old)
    s1, s2 = ratio * adv[:, None], ratio.clamp(1 - eps, 1 + eps) * adv[:, None]
    obj = torch.minimum(s1, s2)
    expected = -(((obj * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)).mean())
    check(
        "Task 3 GRPO: canonical loss divides each sequence by its own length T_k",
        torch.allclose(loss_grpo, expected, atol=1e-6),
        f"got {loss_grpo.item():.6f}, expected {expected.item():.6f}",
    )

    loss_dr, _ = grpo_policy_loss(new, old, adv, mask, ref, eps, beta, loss_type="dr_grpo", max_completion_length=4)
    expected_dr = -(((obj * mask).sum(-1) / 4.0).mean())
    check(
        "Task 3 GRPO: Dr. GRPO divides every sequence by the constant T_max",
        torch.allclose(loss_dr, expected_dr, atol=1e-6),
        f"got {loss_dr.item():.6f}, expected {expected_dr.item():.6f}",
    )
    check(
        "Task 3 GRPO: the two normalizations differ on a masked short sequence",
        abs(loss_grpo.item() - loss_dr.item()) > 1e-6,
        f"grpo={loss_grpo.item():.4f} dr_grpo={loss_dr.item():.4f}",
    )


def main() -> None:
    test_dpo_objective()
    test_ppo_objective()
    test_gae_and_shaping()
    test_grpo_advantages()
    test_grpo_loss()

    failed = [r for r in _results if r[0] == FAIL]
    print()
    print(f"{len(_results) - len(failed)}/{len(_results)} objective checks passed.")
    if failed:
        raise SystemExit("Objective validation FAILED:\n" + "\n".join(f"  - {n}: {d}" for _, n, d in failed))
    print("All released objectives match the assignment equations.")


if __name__ == "__main__":
    main()