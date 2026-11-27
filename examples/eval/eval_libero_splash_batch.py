"""Five distinct LIBERO tasks, one assisted attempt each; never retry episodes.

Runs sequentially against an already-started local Splash server. Wall-time
interruptions are recorded separately from environment failures. Optionally stop
only the server PID explicitly assigned to this batch when it finishes/exits.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TASKS = [
    ("libero_goal", 1, "put_the_bowl_on_the_stove"),
    ("libero_goal", 5, "push_the_plate_to_the_front_of_the_stove"),
    ("libero_goal", 7, "turn_on_the_stove"),
    ("libero_object", 0, "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
    ("libero_spatial", 2, "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"),
]


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def save(root, report):
    temp = root / "batch.json.tmp"
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    temp.replace(root / "batch.json")
    lines = [
        "# Assisted LIBERO diagnostic batch",
        "",
        f"Batch status: {report['status']}; attempts with final records: {len(report['attempts'])}/{len(report['requested_tasks'])}.",
        "One initial state (0) per task; no retries. Not the original policy or paper benchmark.",
        "No privileged object positions or contacts are supplied to the model.",
        "",
        "| Task | Status | Environment success | Turns | Native steps |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["attempts"]:
        result = row.get("result", {})
        lines.append(
            f"| {row['task']} | {row['status']} | {result.get('success', 'unresolved')} | "
            f"{result.get('turns', '—')} | {result.get('native_steps', '—')} |"
        )
    successes = sum(r.get("result", {}).get("success") is True for r in report["attempts"])
    lines += [
        "",
        f"Observed environment successes: {successes}. No whole-suite success rate is inferred.",
        "wall_limit/interrupted/turn_limit/output or inference errors are not completed environment failures.",
        "For interrupted processes, last_saved_progress is only a snapshot, not a final result.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-seconds", type=float, default=900)
    parser.add_argument("--stop-server-pid", type=int, default=0)
    available = {f"{suite}:{task_id}": (suite, task_id, name) for suite, task_id, name in TASKS}
    parser.add_argument("--tasks", nargs="+", choices=list(available), default=list(available))
    parser.add_argument("--calibrated-overlays", action="store_true")
    args = parser.parse_args()
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Duplicate tasks would violate the single-attempt protocol")
    tasks = [available[key] for key in args.tasks]
    if not 0 < args.case_seconds < float("inf"):
        parser.error("case-seconds must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("eval_libero_splash.py")
    common = [
        "--trials",
        "1",
        "--trial-start",
        "0",
        "--max-turns",
        "30",
        "--effort",
        "low",
        "--max-tokens",
        "8192",
        "--thinking-budget",
        "-1",
        "--temperature",
        "1.0",
        "--top-p",
        "0.95",
        "--control-guidance",
        "--measured-feedback",
        "--timeout",
        "600",
    ]
    if args.calibrated_overlays:
        common.append("--calibrated-overlays")
    report = {
        "status": "waiting_for_server",
        "pid": os.getpid(),
        "requested_tasks": tasks,
        "case_seconds": args.case_seconds,
        "common_arguments": common,
        "protocol_mode": "assisted",
        "attempts": [],
    }
    child = None
    active = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    save(args.output, report)
    try:
        deadline = time.monotonic() + 300
        while True:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
                    models = json.load(response)
                if not any(m["id"] == "mit-han-lab/Qwen3.8-27B-PEAR-NVFP4" for m in models["data"]):
                    raise RuntimeError("Expected PEAR model not available")
                break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server startup deadline exceeded") from None
                time.sleep(3)
        report["status"] = "running"
        for suite, task_id, name in tasks:
            folder = args.output / f"{suite}-{task_id:02d}"
            command = [
                sys.executable,
                str(script),
                "--suite",
                suite,
                "--task-id",
                str(task_id),
                "--output",
                str(folder),
                *common,
            ]
            active = {"task": name, "suite": suite, "task_id": task_id, "directory": str(folder), "command": command}
            report["active"] = active
            save(args.output, report)
            print(f"START {suite}/{task_id}: {name}", flush=True)
            started = time.monotonic()
            with (args.output / f"{suite}-{task_id:02d}.log").open("w") as log:
                child = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
                )
                active["pid"] = child.pid
                save(args.output, report)
                try:
                    code = child.wait(timeout=args.case_seconds)
                    active["returncode"] = code
                    summary_path = folder / "trial-00/summary.json"
                    if summary_path.exists():
                        active["result"] = json.loads(summary_path.read_text())
                        active["status"] = active["result"]["status"]
                    else:
                        active["status"] = "process_error"
                except subprocess.TimeoutExpired:
                    stop_child(child)
                    active["status"] = "wall_limit"
                    snapshot = folder / "progress.json"
                    if snapshot.exists():
                        active["last_saved_progress"] = json.loads(snapshot.read_text())
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "STOPPED.md").write_text(
                        "Stopped by the batch wall-time limit. Not a completed environment failure.\n"
                    )
            active["wall_seconds"] = time.monotonic() - started
            report["attempts"].append(active)
            print(f"END {name}: {active['status']}", flush=True)
            active = None
            report.pop("active", None)
            save(args.output, report)
        report["status"] = "finished"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        report["status"] = "batch_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_child(child)
        if active is not None:
            active["status"] = "interrupted"
            report["attempts"].append(active)
        report.pop("active", None)
        if args.stop_server_pid:
            pid_path = Path.home() / "Library/Logs/pear-splash/launcher.pid"
            if pid_path.exists() and pid_path.read_text().strip() == str(args.stop_server_pid):
                result = subprocess.run(
                    [str(Path.home() / ".local/bin/pear-splash"), "stop"], capture_output=True, text=True, timeout=30
                )
                report["server_stop"] = {"returncode": result.returncode, "output": result.stdout + result.stderr}
            else:
                report["server_stop"] = "Skipped: server ownership changed"
        save(args.output, report)
        print(f"BATCH {report['status']}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
