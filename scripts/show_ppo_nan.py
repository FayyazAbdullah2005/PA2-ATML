"""Print the per-stage finiteness diagnostics from a PPO continuation log.

Run:  python -m scripts.show_ppo_nan results/task2_ppo/smoke2_updates.jsonl

Any `*_nonfinite` above zero identifies the tensor that poisoned the update. The log
fields are written by task2_ppo.continue_train.run_ppo.
"""

from __future__ import annotations

import json
import sys

from common.data import repo_path


def main() -> None:
    path = repo_path(sys.argv[1] if len(sys.argv) > 1 else "results/task2_ppo/standard_updates.jsonl")
    if not path.exists():
        raise SystemExit(f"No such log: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        diag = {k: v for k, v in row.items() if k.endswith("_nonfinite")}
        print(
            f"update {row['update']}: policy_loss={row['policy_loss']:.5g} value_loss={row['value_loss']:.5g} "
            f"policy_grad_norm={row['policy_grad_norm']:.5g} value_grad_norm={row['value_grad_norm']:.5g}"
        )
        bad = {k: v for k, v in diag.items() if v}
        if bad:
            print(f"  NON-FINITE at: {bad}")
            print(f"  ranges: {json.dumps({k: v for k, v in row.items() if k.endswith('_absmax')})}")
        else:
            print(f"  all stages finite; ranges: {json.dumps({k: round(v, 3) for k, v in row.items() if k.endswith('_absmax')})}")


if __name__ == "__main__":
    main()