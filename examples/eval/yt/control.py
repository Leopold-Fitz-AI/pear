"""Inspect/stop only processes owned by this deployment (no system service changes)."""

import argparse
import json
import os
import signal
import urllib.request
from pathlib import Path

ROOT = Path("/root/pear-yt")
SIGNATURES = {
    "scaling": [str(ROOT / "scripts/benchmark_scaling.py")],
    "supervisor": ["run_bf16_validation.py", str(ROOT)],
    "server": ["vllm.entrypoints.openai.api_server", str(ROOT / "models/Qwen3.8-27B-PEAR")],
    "download": ["download_pear.py", str(ROOT)],
}


def owned(name):
    try:
        pid = int((ROOT / f"{name}.pid").read_text())
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return pid if all(part in command for part in SIGNATURES[name]) else None
    except (OSError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["status", "stop", "stop-all"])
    args = p.parse_args()
    if args.action != "status":
        names = ["scaling", "supervisor", "server"]
        if args.action == "stop-all":
            names.append("download")
        for name in names:
            pid = owned(name)
            if pid is not None:
                try:
                    if name == "server":
                        os.killpg(pid, signal.SIGTERM)
                    else:
                        os.kill(pid, signal.SIGTERM)
                    print(f"Stop requested: {name} PID {pid}")
                except ProcessLookupError:
                    print(f"Already exited: {name} PID {pid}")
        return
    report = {"owned_processes": {name: owned(name) for name in SIGNATURES}}
    for name in ("download-status.json", "libero-verified.json"):
        try:
            report[name] = json.loads((ROOT / name).read_text())
        except (OSError, json.JSONDecodeError):
            pass
    try:
        run = Path((ROOT / "latest-validation.txt").read_text().strip())
        report["latest_run"] = str(run)
        report["validation"] = json.loads((run / "status.json").read_text())
        active = report["validation"].get("active")
        if active:
            try:
                report["last_evaluator_snapshot"] = json.loads(
                    (Path(active["directory"]) / "progress.json").read_text()
                )
            except (OSError, json.JSONDecodeError):
                pass
    except (OSError, json.JSONDecodeError):
        pass
    try:
        scaling = Path((ROOT / "latest-scaling.txt").read_text().strip())
        report["scaling"] = json.loads((scaling / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    try:
        evaluation = Path((ROOT / "latest-tp4-evaluation.txt").read_text().strip())
        report["tp4_evaluation"] = json.loads((evaluation / "status.json").read_text())
        active = report["tp4_evaluation"].get("active")
        if active:
            report["tp4_progress_snapshot"] = json.loads((Path(active["directory"]) / "progress.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    try:
        with urllib.request.urlopen("http://127.0.0.1:8015/v1/models", timeout=3) as response:
            report["http_models"] = json.load(response)
    except OSError:
        report["http_models"] = "not ready or stopped"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
