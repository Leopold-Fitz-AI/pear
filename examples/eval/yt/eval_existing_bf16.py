"""Five fresh xhigh LIBERO trials against an already supervised BF16 server.

No server lifecycle changes here: the owning TP4 supervisor restores GPU 0 serving.
"""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from run_bf16_validation import MODEL, TASKS, terminate

ROOT = Path("/root/pear-yt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    run = ROOT / "runs" / ("bf16-tp4-evaluation-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir()
    (ROOT / "latest-tp4-evaluation.txt").write_text(str(run) + "\n")
    state = {
        "status": "preflight",
        "pid": os.getpid(),
        "run": str(run),
        "attempts": [],
        "resume_run": str(args.resume_run) if args.resume_run else None,
        "model": MODEL,
        "case_seconds": 7200,
        "max_turns": 300,
        "trial_start": 0,
        "context_limit": 260096,
        "episode_response_budget": 256000,
        "protocol": "Original prompts, xhigh, temperature=1, top_p=.95, no per-turn thinking cap; "
        "4096 initial prompt budget + 256000 episode response/observation budget; "
        "no guidance, feedback, overlays or object truth; five fresh tasks, no retries; "
        "original native-step budgets. Final PEAR checkpoint, not LIBERO-only paper reproduction.",
        "server_supervisor_run": (ROOT / "latest-scaling.txt").read_text().strip(),
    }
    child = None

    def save():
        tmp = run / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(run / "status.json")
        lines = [
            "# BF16 TP4 robot evaluation",
            "",
            f"Status: {state['status']}",
            "",
            state["protocol"],
            "",
            "| Task | Status | Environment success | Turns | Native steps | Seconds |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in state["attempts"]:
            r = row.get("result", {})
            lines.append(
                f"| {row['suite']}/{row['task_id']} | {row['status']} | {r.get('success', 'unresolved')} | "
                f"{r.get('turns', '—')} | {r.get('native_steps', '—')} | {row['seconds']:.1f} |"
            )
        lines += [
            "",
            "Wall-time interruptions, model errors and incomplete attempts are not complete environment failures.",
            "Shared GPUs 0,1,4,6, BF16 TP=4, compilation/CUDA Graph, 45% per-GPU memory budget.",
        ]
        (run / "REPORT.md").write_text("\n".join(lines) + "\n")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        with urllib.request.urlopen("http://127.0.0.1:8015/v1/models", timeout=5) as response:
            models = json.load(response)
        assert any(m["id"] == MODEL and m.get("max_model_len", 0) >= 260096 for m in models["data"])
        (run / "server-models.json").write_text(json.dumps(models, indent=2))
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="0",
            MUJOCO_GL="egl",
            MUJOCO_EGL_DEVICE_ID="0",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        common = [
            str(ROOT / "envs/sim/bin/python"),
            str(ROOT / "pear/examples/eval/eval_libero_splash.py"),
            "--libero-root",
            str(ROOT / "LIBERO"),
            "--config-dir",
            str(ROOT / "config"),
            "--server-kind",
            "vllm",
            "--base-url",
            "http://127.0.0.1:8015/v1",
            "--model",
            MODEL,
            "--trials",
            "1",
            "--trial-start",
            "0",
            "--effort",
            "xhigh",
            "--thinking-budget",
            "-1",
            "--temperature",
            "1",
            "--top-p",
            "0.95",
            "--max-tokens",
            "256000",
            "--max-turns",
            "300",
            "--episode-response-budget",
            "256000",
            "--required-context",
            "260096",
            "--server-image-limit",
            "1016",
            "--timeout",
            "7200",
        ]
        state.update(status="evaluating", command_common=common)
        save()
        for suite, task in TASKS:
            folder = run / f"{suite}-{task:02d}"
            row = {"suite": suite, "task_id": task, "directory": str(folder)}
            resume_options = []
            if args.resume_run:
                source = args.resume_run / folder.name
                if list((source / "trial-00").glob("*-transition.json")):
                    summary = json.loads((source / "trial-00/summary.json").read_text())
                    if summary["status"] != "token_accounting_error":
                        raise ValueError("Refusing to retry a completed or different interrupted episode")
                    resume_options = ["--resume-from", str(source)]
                    row["resume_from"] = str(source)
                elif source.exists():
                    row["prior_unexecuted_attempt"] = str(source)
            state["active"] = row
            save()
            started = time.monotonic()
            with (run / f"{suite}-{task:02d}.log").open("w") as log:
                child = subprocess.Popen(
                    [*common, *resume_options, "--suite", suite, "--task-id", str(task), "--output", str(folder)],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                row["pid"] = child.pid
                save()
                try:
                    row["returncode"] = child.wait(timeout=7200)
                    p = folder / "trial-00/summary.json"
                    if p.exists():
                        row["result"] = json.loads(p.read_text())
                        row["status"] = row["result"]["status"]
                    else:
                        row["status"] = "process_error"
                except subprocess.TimeoutExpired:
                    terminate(child)
                    row["status"] = "wall_limit"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "STOPPED.md").write_text(
                        "Interrupted at 7200 seconds; not a completed environment failure.\n"
                    )
            row["seconds"] = time.monotonic() - started
            state["attempts"].append(row)
            state.pop("active", None)
            save()
            if row["status"] in ("token_accounting_error", "process_error", "environment_error"):
                raise RuntimeError(f"Infrastructure error: {row['status']}; remaining tasks not attempted")
            # Fail closed if the model service died; do not produce repeated connection-error trials.
            with urllib.request.urlopen("http://127.0.0.1:8015/health", timeout=5):
                pass
        state["status"] = "finished"
    except (Exception, KeyboardInterrupt) as exc:
        state.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
            error=f"{type(exc).__name__}: {exc}",
        )
        terminate(child)
        if state.get("active"):
            folder = Path(state["active"]["directory"])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "STOPPED.md").write_text(
                state["error"] + "\nIncomplete attempt, not a completed environment failure.\n"
            )
    finally:
        save()
    if state["status"] != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
