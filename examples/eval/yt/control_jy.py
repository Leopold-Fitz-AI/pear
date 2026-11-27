"""Inspect/stop only this jy PEAR deployment. Does not touch yt's services."""

import argparse
import json
import os
import signal
import urllib.request
from pathlib import Path

ROOT = Path("/root/pear-jy")
SIGNATURES = {
    "evaluation": ["/root/pear-jy/scripts/run_jy_libero_smoke.py"],
    "deploy": ["/root/pear-jy/scripts/deploy_jy_tp8.py"],
    "server": ["vllm.entrypoints.openai.api_server", "pear-jy-deployment/models/Qwen3.8-27B-PEAR"],
}


def owned(name):
    try:
        pid = int((ROOT / f"{name}.pid").read_text())
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return pid if all(s in command for s in SIGNATURES[name]) else None
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "stop", "stop-evaluation"])
    args = parser.parse_args()
    if args.action != "status":
        for name in ["evaluation"] if args.action == "stop-evaluation" else SIGNATURES:
            pid = owned(name)
            if pid is not None:
                try:
                    if name == "server":
                        os.killpg(pid, signal.SIGTERM)
                    else:
                        os.kill(pid, signal.SIGTERM)
                    print(f"Stop requested: jy PEAR {name} PID {pid}")
                except ProcessLookupError:
                    pass
        return
    result = {"owned_processes": {name: owned(name) for name in SIGNATURES}}
    try:
        run = Path((ROOT / "latest-deployment.txt").read_text().strip())
        result["deployment"] = json.loads((run / "status.json").read_text())
        result["transfer"] = json.loads(Path("/vla-cd/pear-jy-deployment/transfer-status.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    try:
        evaluation = Path((ROOT / "latest-libero-evaluation.txt").read_text().strip())
        result["evaluation"] = json.loads((evaluation / "status.json").read_text())
        active = result["evaluation"].get("active")
        if active:
            result["evaluation_progress_snapshot"] = json.loads(
                (Path(active["directory"]) / "progress.json").read_text()
            )
    except (OSError, json.JSONDecodeError):
        pass
    try:
        with urllib.request.urlopen("http://127.0.0.1:8015/v1/models", timeout=3) as f:
            result["models"] = json.load(f)
    except OSError:
        result["models"] = "not ready or stopped"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
