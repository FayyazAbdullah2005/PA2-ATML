from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Validated against the manual objective

        L_DPO = -E[ log sigma( beta * ( [log pi(y+|x) - log pi_ref(y+|x)]
                                        -[log pi(y-|x) - log pi_ref(y-|x)] ) ) ]

    The release shipped ``beta * (policy_margin + ref_margin)``, which is a
    defect: the reference log-ratio must be *subtracted*, because DPO models the
    *change* in relative likelihood induced by the policy. With the additive
    form the objective is not a Bradley-Terry loss on the implicit reward
    ``beta * log pi/pi_ref``; at theta = theta_ref it would return
    ``-log sigma(2 * beta * ref_margin)`` instead of ``log 2``, so the released
    version pushes the policy away from the reference and does not vanish at
    initialisation.
    """

    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    ref_margin = (
        ref_chosen_logp
        - ref_rejected_logp
    )

    # Implicit reward margin: beta * [ (log pi - log pi_ref)(y+) - (log pi - log pi_ref)(y-) ]
    logits = beta * (
        policy_margin - ref_margin
    )

    loss = -F.logsigmoid(
        logits
    ).mean()

    return loss, {
        "logit_mean":
            logits.detach().mean(),

        "policy_margin_mean":
            policy_margin.detach().mean(),

        "preference_accuracy": (
            (policy_margin - ref_margin) > 0
        ).float().mean().detach(),
    }
