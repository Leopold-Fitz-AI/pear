"""yt-only deployment supervisor: verified BF16 weights -> GPU 0 -> five trials.

Never stops unrelated processes, never changes global CUDA/driver configuration.
Only localhost is served. Successful deployment remains running after validation.
"""

import argparse
import base64
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL = "mit-han-lab/Qwen3.8-27B-PEAR"
TASKS = [("libero_goal", 1), ("libero_goal", 5), ("libero_goal", 7), ("libero_object", 0), ("libero_spatial", 2)]


def terminate(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    except ProcessLookupError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/root/pear-yt"))
    p.add_argument("--case-seconds", type=float, default=900)
    args = p.parse_args()
    root = args.root
    run = root / "runs" / ("bf16-matched-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "waiting_for_download_and_environments",
        "pid": os.getpid(),
        "run": str(run),
        "model": MODEL,
        "tasks": TASKS,
        "attempts": [],
        "scope": "BF16/vLLM assisted diagnostic; backend and precision change together; not paper replication",
    }
    (root / "latest-validation.txt").write_text(str(run) + "\n")
    server = child = None

    def save():
        temp = run / "status.json.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(run / "status.json")
        lines = [
            "# yt BF16 matched diagnostic",
            "",
            f"Status: {status['status']}",
            "One initial state per task, no retries. No object truth or calibrated overlays in model inputs.",
            "",
            "| Suite/task | Status | Success | Turns | Native steps | Seconds |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in status["attempts"]:
            r = row.get("result", {})
            lines.append(
                f"| {row['suite']}/{row['task_id']} | {row['status']} | {r.get('success', 'unresolved')} | "
                f"{r.get('turns', '—')} | {r.get('native_steps', '—')} | {row.get('seconds', 0):.1f} |"
            )
        lines += [
            "",
            status["scope"],
            "Interrupted/error attempts are not completed environment failures.",
            "Weights: authored non-quantized checkpoint, BF16 execution, no draft model, no FP8 KV cache.",
            "Sampling: low effort, temperature 1, top_p .95, top_k disabled; 8192 response tokens; 30 turns.",
            "The original per-task native-step budgets apply; each case has a separate wall-time limit.",
        ]
        (run / "REPORT.md").write_text("\n".join(lines) + "\n")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        deadline = time.monotonic() + 7200
        while True:
            download_path = root / "download-status.json"
            try:
                download = json.loads(download_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                download = {}
            if download.get("status") == "error":
                raise RuntimeError("Model download or checksum verification failed")
            server_ok = (root / "envs/server/lib/python3.12/site-packages/vllm/__init__.py").exists()
            sim_ok = (root / "envs/sim/lib/python3.10/site-packages/torch/__init__.py").exists()
            if (
                download.get("status") == "verified"
                and server_ok
                and sim_ok
                and (root / "libero-verified.json").exists()
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Preparation deadline exceeded")
            time.sleep(10)
        status["weights_revision"] = download["revision"]
        status["status"] = "gpu_preflight"
        save()
        gpu = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip()
        uuid, free = [x.strip() for x in gpu.split(",")]
        status["gpu0"] = {"uuid": uuid, "free_mib_before_start": int(free)}
        if int(free) < 76000:
            raise RuntimeError("GPU 0 is no longer sufficiently free; existing jobs left untouched")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", 8015)) == 0:
                raise RuntimeError("Port 8015 already occupied; will not stop its owner")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=uuid,
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4",
            VLLM_CACHE_ROOT=str(root / "cache/vllm"),
        )
        env["LD_LIBRARY_PATH"] = (
            str(root / "cuda-compat-13/usr/local/cuda-13.0/compat") + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        command = [
            str(root / "envs/server/bin/python"),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(root / "models/Qwen3.8-27B-PEAR"),
            "--served-model-name",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            "8015",
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "131072",
            "--gpu-memory-utilization",
            "0.90",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "4096",
            "--limit-mm-per-prompt",
            '{"image":64,"video":0}',
            "--mm-processor-kwargs",
            '{"max_pixels":262144}',
            "--reasoning-parser",
            "qwen3",
            "--generation-config",
            "vllm",
            "--enforce-eager",
        ]
        status.update(status="starting_server", server_command=command)
        save()
        with (root / "logs/server.log").open("a") as log:
            server = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True
            )
        (root / "server.pid").write_text(str(server.pid) + "\n")
        status["server_pid"] = server.pid
        save()
        deadline = time.monotonic() + 1800
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited with {server.returncode}; see logs/server.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8015/v1/models", timeout=5) as response:
                    models = json.load(response)
                if any(m["id"] == MODEL for m in models["data"]):
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Server readiness deadline exceeded")
            time.sleep(5)
        (run / "server-models.json").write_text(json.dumps(models, indent=2))
        status["status"] = "vision_preflight"
        save()
        simenv = dict(
            os.environ,
            # robosuite 1.4.1 validates EGL's numeric physical index against
            # CUDA_VISIBLE_DEVICES as a string; it rejects UUID masks. This
            # simulator uses CPU-only torch and EGL device 0, never CUDA kernels.
            CUDA_VISIBLE_DEVICES="0",
            MUJOCO_GL="egl",
            MUJOCO_EGL_DEVICE_ID="0",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        # Separate graphics libraries: do NOT inject CUDA compatibility libs into the GL simulator.
        script = root / "pear/examples/eval/eval_libero_splash.py"
        common = [
            str(root / "envs/sim/bin/python"),
            str(script),
            "--libero-root",
            str(root / "LIBERO"),
            "--config-dir",
            str(root / "config"),
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
            "low",
            "--thinking-budget",
            "-1",
            "--temperature",
            "1",
            "--top-p",
            "0.95",
            "--max-tokens",
            "8192",
            "--max-turns",
            "30",
            "--control-guidance",
            "--measured-feedback",
            "--timeout",
            "600",
        ]
        smoke = run / "render-smoke"
        with (run / "render-smoke.log").open("w") as log:
            subprocess.run(
                [*common, "--suite", "libero_goal", "--task-id", "7", "--smoke", "--output", str(smoke)],
                env=simenv,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        image = (smoke / "trial-00/000-agentview.png").read_bytes()
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Briefly describe the visible robot and objects. Do not claim a task has succeeded.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()},
                        },
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:8015/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.load(response)
        (run / "vision-smoke-response.json").write_text(json.dumps(answer, indent=2))
        if answer["choices"][0]["finish_reason"] != "stop" or not answer["choices"][0]["message"].get("content"):
            raise RuntimeError("Vision preflight did not produce a complete response")
        status["status"] = "evaluating"
        save()
        for suite, task_id in TASKS:
            folder = run / f"{suite}-{task_id:02d}"
            row = {"suite": suite, "task_id": task_id, "directory": str(folder)}
            status["active"] = row
            save()
            started = time.monotonic()
            with (run / f"{suite}-{task_id:02d}.log").open("w") as log:
                child = subprocess.Popen(
                    [*common, "--suite", suite, "--task-id", str(task_id), "--output", str(folder)],
                    env=simenv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                row["pid"] = child.pid
                save()
                try:
                    row["returncode"] = child.wait(timeout=args.case_seconds)
                    path = folder / "trial-00/summary.json"
                    if path.exists():
                        row["result"] = json.loads(path.read_text())
                        row["status"] = row["result"]["status"]
                    else:
                        row["status"] = "process_error"
                except subprocess.TimeoutExpired:
                    terminate(child)
                    row["status"] = "wall_limit"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "STOPPED.md").write_text("Wall-time interruption, not a completed environment failure.\n")
            row["seconds"] = time.monotonic() - started
            status["attempts"].append(row)
            status.pop("active", None)
            save()
        status["status"] = "finished"
        status["server_left_running"] = True
    except (Exception, KeyboardInterrupt) as exc:
        status["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        status["error"] = f"{type(exc).__name__}: {exc}"
        terminate(child)
        terminate(server)
    finally:
        save()
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
