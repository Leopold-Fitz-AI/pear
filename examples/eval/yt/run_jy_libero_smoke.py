"""Install the isolated, verified LIBERO runtime and test one episode per suite.

Use the existing jy TP4 server, original prompts, xhigh, official token budgets.
No retries, object truth, assisted feedback, training, or server lifecycle changes.
"""

import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/root/pear-jy")
SHARED = Path("/vla-cd/pear-jy-deployment")
MODEL = "mit-han-lab/Qwen3.8-27B-PEAR"
TASKS = [("libero_goal", 0), ("libero_spatial", 0), ("libero_object", 0), ("libero_10", 0)]


def stop(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main():
    run = ROOT / "runs" / ("libero-paper-settings-smoke-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir()
    (ROOT / "latest-libero-evaluation.txt").write_text(str(run) + "\n")
    (ROOT / "evaluation.pid").write_text(str(os.getpid()) + "\n")
    state = {
        "status": "waiting_for_runtime",
        "pid": os.getpid(),
        "run": str(run),
        "attempts": [],
        "tasks": TASKS,
        "trial_id": 0,
        "case_seconds": 7200,
        "scope": "Four-suite smoke subset, not full 2000-episode LIBERO or paper-checkpoint reproduction",
        "protocol": "original prompts; BF16, xhigh, 4096 initial / 256000 continuation budget, 260096 server context; "
        "temperature=1, top_p=.95, top_k=-1; no extra thinking cap, no object truth or assistance",
        "rendering": "Local GLFW/EGL adapter, NVIDIA EGL rather than paper CPU Mesa; original physics/controller/success predicate",
    }
    child = None

    def save():
        p = run / "status.tmp"
        p.write_text(json.dumps(state, indent=2))
        p.replace(run / "status.json")
        lines = [
            "# jy LIBERO four-suite smoke",
            "",
            f"Status: {state['status']}",
            "",
            state["scope"],
            state["protocol"],
            "",
            "| Suite/task | Status | Environment success | Turns | Native steps | Seconds |",
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
            "Infrastructure errors and wall-time interruptions are not completed task failures. No automatic retries.",
        ]
        (run / "REPORT.md").write_text("\n".join(lines) + "\n")

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    save()
    try:
        deadline = time.monotonic() + 3600
        while True:
            try:
                transfer = json.loads((SHARED / "jy-sim-transfer.json").read_text())
            except (OSError, json.JSONDecodeError):
                transfer = {}
            if transfer.get("status") == "error":
                raise RuntimeError("Runtime transfer failed: " + str(transfer.get("error")))
            if transfer.get("status") == "verified":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Runtime preparation deadline exceeded")
            time.sleep(10)
        state.update(status="installing_runtime", runtime_archive=transfer)
        save()
        marker = ROOT / "sim-environment-ready.json"
        if not marker.exists():
            h = hashlib.sha256()
            with (SHARED / "jy-sim-bundle.tar").open("rb") as f:
                for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != transfer["sha256"]:
                raise RuntimeError("Simulation archive checksum mismatch")
            subprocess.run(
                ["/root/miniconda3/bin/python", "-m", "venv", "--without-pip", str(ROOT / "envs/sim")],
                check=True,
                timeout=120,
            )
            subprocess.run(["tar", "-xf", str(SHARED / "jy-sim-bundle.tar"), "-C", str(ROOT)], check=True, timeout=900)
            site = ROOT / "envs/sim/lib/python3.10/site-packages"
            finder = site / "__editable___pear_0_1_0_finder.py"
            text = finder.read_text()
            if "MAPPING: dict[str, str] = {'pear': '/root/pear-yt/pear/pear'}" not in text:
                raise RuntimeError("Unexpected editable package mapping")
            # Relocate the top-level package AND setuptools namespace-resource paths.
            finder.write_text(text.replace("/root/pear-yt/pear/pear", str(ROOT / "pear/pear")))
            direct = site / "pear-0.1.0.dist-info/direct_url.json"
            direct.write_text(json.dumps({"dir_info": {"editable": True}, "url": (ROOT / "pear").as_uri()}))
            m = json.loads((ROOT / "libero-file-manifest.json").read_text())
            for name, expected in m["files"].items():
                if hashlib.sha256((ROOT / "LIBERO" / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("LIBERO pinned-file mismatch: " + name)
            config = ROOT / "config/libero"
            config.mkdir(parents=True, exist_ok=True)
            lib = ROOT / "LIBERO/libero/libero"
            datasets = ROOT / "datasets/libero"
            datasets.mkdir(parents=True, exist_ok=True)
            mapping = {
                "benchmark_root": lib,
                "bddl_files": lib / "bddl_files",
                "init_states": lib / "init_files",
                "assets": lib / "assets",
                "datasets": datasets,
            }
            (config / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in mapping.items()) + "\n")
            marker.write_text(
                json.dumps(
                    {
                        "archive_sha256": transfer["sha256"],
                        "libero_revision": m["revision"],
                        "verified_files": len(m["files"]),
                    },
                    indent=2,
                )
            )
        with urllib.request.urlopen("http://127.0.0.1:8015/v1/models", timeout=10) as f:
            models = json.load(f)
        if not any(m["id"] == MODEL and m.get("max_model_len", 0) >= 260096 for m in models["data"]):
            raise RuntimeError("Expected BF16 server with full context not available")
        (run / "server-models.json").write_text(json.dumps(models, indent=2))
        env = dict(
            os.environ,
            PYTHONPATH=str(ROOT / "pear"),
            CUDA_VISIBLE_DEVICES="0",
            MUJOCO_GL="egl",
            MUJOCO_EGL_DEVICE_ID="0",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        common = [
            str(ROOT / "envs/sim/bin/python"),
            str(ROOT / "pear/examples/eval/eval_libero_splash.py"),
            "--libero-root",
            str(ROOT / "LIBERO"),
            "--config-dir",
            str(ROOT / "config/libero"),
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
            "508",
            "--episode-response-budget",
            "256000",
            "--required-context",
            "260096",
            "--server-image-limit",
            "1016",
            "--timeout",
            "7200",
        ]
        state.update(status="render_preflight", command_common=common)
        save()
        with (run / "render-preflight.log").open("w") as log:
            child = subprocess.Popen(
                [
                    *common,
                    "--suite",
                    "libero_goal",
                    "--task-id",
                    "0",
                    "--smoke",
                    "--output",
                    str(run / "render-preflight"),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if child.wait(timeout=180):
                raise RuntimeError("Renderer preflight process failed")
        summary = json.loads((run / "render-preflight/trial-00/summary.json").read_text())
        if summary["status"] != "render_only":
            raise RuntimeError("Renderer preflight failed: " + str(summary))
        state["status"] = "evaluating"
        save()
        for suite, task in TASKS:
            folder = run / f"{suite}-{task:02d}"
            row = {"suite": suite, "task_id": task, "directory": str(folder)}
            state["active"] = row
            save()
            started = time.monotonic()
            with (run / f"{suite}-{task:02d}.log").open("w") as log:
                child = subprocess.Popen(
                    [*common, "--suite", suite, "--task-id", str(task), "--output", str(folder)],
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
                    path = folder / "trial-00/summary.json"
                    if path.exists():
                        row["result"] = json.loads(path.read_text())
                        row["status"] = row["result"]["status"]
                    else:
                        row["status"] = "process_error"
                except subprocess.TimeoutExpired:
                    stop(child)
                    row["status"] = "wall_limit"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "STOPPED.md").write_text(
                        "7200-second safety limit; not a completed environment failure.\n"
                    )
            row["seconds"] = time.monotonic() - started
            state["attempts"].append(row)
            state.pop("active", None)
            save()
            if row["status"] in ("process_error", "environment_error", "token_accounting_error"):
                raise RuntimeError("Infrastructure error; remaining tasks not attempted: " + row["status"])
            with urllib.request.urlopen("http://127.0.0.1:8015/health", timeout=5):
                pass
        state["status"] = "finished"
    except (Exception, KeyboardInterrupt) as exc:
        state.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
            error=f"{type(exc).__name__}: {exc}",
        )
        stop(child)
        if state.get("active"):
            folder = Path(state["active"]["directory"])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "STOPPED.md").write_text(state["error"] + "\nIncomplete attempt, not a completed failure.\n")
    finally:
        save()


if __name__ == "__main__":
    main()
