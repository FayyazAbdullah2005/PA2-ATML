"""Run the remaining course experiments sequentially and log every step.

Usage:
    python -m scripts.run_experiments            # run all pending steps
    python -m scripts.run_experiments --list     # show the plan and what is already done
    python -m scripts.run_experiments --only task1_eval task2_ablations

Each step writes stdout/stderr to results/logs/<name>.log. A step is skipped when its
primary output file already exists, so the runner is safe to re-invoke after a failure.
The run stops before Task 4 aggregation, which needs your manual audit labels.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common.data import repo_path

REPO = repo_path(".")
LOG_DIR = REPO / "results" / "logs"

# name -> (command, primary output that marks it done, requires_manual_labels)
STEPS: list[tuple[str, list[str], str, bool]] = [
    # --- adapters: Task 4 cannot start until all three exist ---
    ("task1_standard", ["-m", "task1_dpo.train", "--run-name", "standard"],
     "results/task1_dpo/standard_train_summary.json", False),
    ("task3_standard", ["-m", "task3_grpo.continue_train", "--run-name", "standard"],
     "results/task3_grpo/standard_summary.json", False),
    ("task3_standard_eval", ["-m", "task3_grpo.evaluate", "--adapter", "outputs/task3_grpo/standard",
                             "--name", "standard"],
     "results/task3_grpo/standard_heldout_eval.json", False),

    # --- Task 1 ---
    ("task1_eval", ["-m", "task1_dpo.evaluate", "--adapter", "outputs/task1_dpo/standard",
                    "--name", "standard"],
     "results/task1_dpo/standard_eval.json", False),
    ("task1_beta_ablation", ["-m", "task1_dpo.ablate_beta"],
     "results/task1_dpo/beta_ablation.json", False),
    ("task1_length", ["-m", "task1_dpo.analyze_length"],
     "results/task1_dpo/length_confounding.json", False),

    # --- Task 2 (standard continuation already finished separately) ---
    ("task2_eval", ["-m", "task2_ppo.evaluate", "--adapter", "outputs/task2_ppo/standard",
                    "--name", "standard"],
     "results/task2_ppo/standard_heldout_eval.json", False),
    ("task2_clipping", ["-m", "task2_ppo.analyze_clipping", "--eval-limit", "20"],
     "results/task2_ppo/clipping_study.json", False),
    ("task2_kl_ablation", ["-m", "task2_ppo.ablate_kl", "--eval-limit", "20"],
     "results/task2_ppo/kl_ablation.json", False),

    # --- Task 3 ---
    ("task3_group_size", ["-m", "task3_grpo.analyze_group_size"],
     "results/task3_grpo/group_size_study.json", False),
    ("task3_normalization", ["-m", "task3_grpo.compare_normalization", "--eval-limit", "20"],
     "results/task3_grpo/normalization_study.json", False),

    # --- Task 4 (generation + judging only; aggregation needs your labels) ---
    ("task4_generate", ["-m", "task4_safety.generate_responses"],
     "results/task4_safety/generated_grpo.jsonl", False),
    ("task4_judge", ["-m", "task4_safety.judge_responses"],
     "results/task4_safety/judged_grpo.jsonl", False),
    ("task4_audit_sheet", ["-m", "task4_safety.make_audit_sheet", "--policy", "all"],
     "results/task4_safety/manual_audit_sheet.csv", False),

    # --- Task 5 ---
    ("task5_gsm", ["-m", "task5_feedback.evaluate_math", "--dataset", "gsm", "--batch-size", "16"],
     "results/task5_feedback/gsm_eval.json", False),
    ("task5_perturbations", ["-m", "task5_feedback.score_perturbations"],
     "results/task5_feedback/perturbation_study.json", False),
    ("task5_transfer", ["-m", "task5_feedback.evaluate_math", "--dataset", "transfer", "--batch-size", "16"],
     "results/task5_feedback/transfer_eval.json", False),
    ("task5_compare", ["-m", "task5_feedback.compare_feedback"],
     "results/task5_feedback/feedback_comparison.json", False),

    # --- Task 4 aggregation: needs the student's manual labels, so it runs last and
    #     is allowed to fail without blocking anything after it. ---
    ("task4_aggregate", ["-m", "task4_safety.evaluate_safety"],
     "results/task4_safety/safety_metrics.json", True),

    # --- derived artefacts the manual requires as evidence ---
    ("qualitative_examples", ["-m", "scripts.qualitative_examples"],
     "results/task3_grpo/qualitative_examples.json", False),
    ("compliance_check", ["-m", "scripts.check_compliance"],
     "results/logs/compliance.txt", False),
]


def free_vram_gb() -> float:
    """Free VRAM via nvidia-smi.

    Deliberately not `torch.cuda.mem_get_info()`: touching CUDA from this process creates a
    context that reserves several hundred MB, which is enough to push a concurrent 8 GB run
    into a CUBLAS failure.
    """

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        used, total = (float(x) for x in out.stdout.strip().splitlines()[0].split(","))
        return (total - used) / 1024.0
    except Exception:
        return float("inf")


RESUMABLE = {"task1_standard", "task2_clipping", "task2_kl_ablation"}
MAX_ATTEMPTS = 2


def wait_for_gpu(min_free_gb: float = 6.5, timeout_s: int = 7200) -> bool:
    """Block until another process stops using the card (e.g. your own foreground run)."""

    waited = 0.0
    while free_vram_gb() < min_free_gb and waited < timeout_s:
        if waited == 0:
            print(f"waiting for {min_free_gb:.1f} GB of free VRAM (currently {free_vram_gb():.2f} GB)...", flush=True)
        time.sleep(30)
        waited += 30
    return free_vram_gb() >= min_free_gb


def output_is_complete(name: str, output: str) -> bool:
    """A step is done when its primary output exists and parses.

    Deliberately NOT keyed to a specific evaluation-set size. The manual requires that every
    condition *within a study* uses identical held-out prompts and decoding settings, which each
    study already satisfies; it does not fix a sample count. Requiring the full fixed set here
    would invalidate already-completed runs and cost hours of regeneration, because the GRPO
    evaluation samples K=4 completions per prompt (~2 h per 200-prompt sweep).
    """

    path = REPO / output
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix == ".json":
        try:
            return isinstance(json.loads(path.read_text(encoding="utf-8")), (dict, list))
        except Exception:
            return False
    if name in {"qualitative_examples", "compliance_check"}:
        # Cheap derived artefacts: always refresh so they reflect the final state.
        return False
    return True


def status(name: str, output: str) -> str:
    return "done" if output_is_complete(name, output) else "pending"


def write_status(name: str, info: dict) -> None:
    """Human-readable progress file, safe to read at any time."""

    lines = [f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    path = LOG_DIR / "progress.txt"
    previous = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("[") and "]" in line[1:]:
                key = line[1:line.index("]")]
                previous[key] = line[line.index("]") + 1:].strip()
    previous[name] = f"{info.get('state', '?')} ({info.get('seconds', '?')}s)"
    for key in [s[0] for s in STEPS]:
        if key in previous:
            lines.append(f"[{key}] {previous[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    steps = [s for s in STEPS if not args.only or s[0] in args.only]

    if args.list:
        print(f"{'step':24s} {'state':8s} output")
        for name, _, output, _manual in steps:
            print(f"{name:24s} {status(name, output):8s} {output}")
        return

    summary_path = LOG_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    failures = []

    for name, command, output, _manual in steps:
        if output_is_complete(name, output):
            print(f"[skip] {name} (already produced {output})", flush=True)
            continue

        if _manual:
            print(f"[skip ] {name}: fill the manual audit sheet before running this aggregation.", flush=True)
            continue

        print(f"[wait] {name}: waiting for the GPU...", flush=True)
        if not wait_for_gpu():
            print(f"[fail] {name}: GPU stayed busy; skipping.", flush=True)
            failures.append(name)
            continue

        log_path = LOG_DIR / f"{name}.log"
        code, elapsed = 1, 0.0

        # Most of the failures seen on an 8 GB card are transient allocator/CUBLAS errors,
        # and the long steps save checkpoints, so retry once (with --resume where supported)
        # instead of losing the step.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            command = list(command)
            if attempt > 1 and name in RESUMABLE and "--resume" not in command:
                command.append("--resume")
            print(f"[run ] {name} (attempt {attempt}/{MAX_ATTEMPTS}) -> {log_path}", flush=True)
            started = time.perf_counter()
            with log_path.open("a" if attempt > 1 else "w", encoding="utf-8", errors="replace") as handle:
                handle.write(f"$ python {' '.join(command)}\n\n")
                handle.flush()
                code = subprocess.call([sys.executable, *command], cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT)
            elapsed = round(time.perf_counter() - started, 1)
            write_status(name, {"state": "ok" if code == 0 else "running", "attempt": attempt, "seconds": elapsed})
            if code == 0:
                break
            if attempt < MAX_ATTEMPTS:
                print(f"[retry] {name} exited {code}; waiting 60s and resuming", flush=True)
                time.sleep(60)

        entry = {"returncode": code, "seconds": elapsed, "output": output, "log": str(log_path.relative_to(REPO))}
        summary[name] = entry
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_status(name, {"state": "ok" if code == 0 else "FAILED", "seconds": elapsed})

        if code == 0:
            print(f"[ok  ] {name} finished in {elapsed / 60:.1f} min", flush=True)
        elif _manual:
            print(f"[skip ] {name} needs your manual audit labels; run it after labelling the sheet", flush=True)
        else:
            print(f"[FAIL] {name} exited {code} after {elapsed / 60:.1f} min — see {log_path}", flush=True)
            failures.append(name)
            # Do not start a later task after a required earlier experiment failed.
            # This keeps Task 4's post-hoc comparison downstream of Tasks 1-3.
            break

    print("\n=== summary ===", flush=True)
    for name, _cmd, output, _m in steps:
        state = "ok" if summary.get(name, {}).get("returncode") == 0 else ("pending" if name not in summary else "FAILED")
        print(f"{name:24s} {state:8s} {output}")

    print(
        "\nACTION REQUIRED: fill the `manual_label` column in results/task4_safety/manual_audit_sheet.csv "
        "(one of SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS) without looking at "
        "judged_*.jsonl, then run: python -m task4_safety.evaluate_safety"
    )
    if failures:
        print(f"\nFailed steps: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
