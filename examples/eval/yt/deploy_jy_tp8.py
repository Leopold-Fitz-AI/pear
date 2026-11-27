"""jy-only shared PEAR deployment (defaults to TP=8; GPUs are configurable). Never stops other GPU jobs.

Weights/runtime are verified copies from yt, staged on the shared filesystem.
Successful serving remains running. No robot evaluation is started automatically.
"""

import argparse
import base64
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/root/pear-jy")
SHARED = Path("/vla-cd/pear-jy-deployment")
MODEL = "mit-han-lab/Qwen3.8-27B-PEAR"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        dict(
            zip(
                ["index", "uuid", "used_mib", "total_mib", "utilization"],
                [s.strip() if i == 1 else int(s.strip()) for i, s in enumerate(line.split(","))],
            )
        )
        for line in text.splitlines()
    ]


def stop_child(p):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=60)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if (
        len(args.gpus) not in (1, 2, 4, 8)
        or len(set(args.gpus)) != len(args.gpus)
        or any(i not in range(8) for i in args.gpus)
        or not 0 < args.memory_fraction < 1
    ):
        parser.error("Select 1/2/4/8 unique GPU indices 0..7 and a memory fraction between 0 and 1")
    run = ROOT / "runs" / ("deployment-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True)
    (ROOT / "latest-deployment.txt").write_text(str(run) + "\n")
    (ROOT / "deploy.pid").write_text(str(os.getpid()) + "\n")
    state = {
        "status": "waiting_for_verified_transfer",
        "pid": os.getpid(),
        "run": str(run),
        "model": MODEL,
        "tp": len(args.gpus),
        "gpu_indices": args.gpus,
        "context_limit": 260096,
        "memory_fraction_per_gpu": args.memory_fraction,
        "effort": "xhigh supplied by request/template; explicit in smoke test",
        "scope": f"Shared {len(args.gpus)} A100s, existing jobs retained; deployment only, no robot trials",
    }
    server = child = None

    def save():
        p = run / "status.tmp"
        p.write_text(json.dumps(state, indent=2))
        p.replace(run / "status.json")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        deadline = time.monotonic() + 21600
        while True:
            try:
                transfer = json.loads((SHARED / "transfer-status.json").read_text())
            except (OSError, json.JSONDecodeError):
                transfer = {}
            if transfer.get("status") == "error":
                raise RuntimeError("Transfer failed: " + str(transfer.get("error")))
            if transfer.get("status") == "verified":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Shared transfer readiness deadline exceeded")
            time.sleep(10)
        state.update(status="installing_isolated_environment", revision=transfer["revision"])
        save()
        (run / "transfer-manifest.json").write_text(json.dumps(transfer, indent=2))
        marker = ROOT / "environment-ready.json"
        if not marker.exists():
            for name, meta in transfer["bundles"].items():
                if sha256(SHARED / name) != meta["sha256"]:
                    raise RuntimeError("Bundle checksum mismatch: " + name)
            runtime = ROOT / "runtime"
            runtime.mkdir(exist_ok=True)
            subprocess.run(
                ["tar", "-xf", str(SHARED / "python-runtime.tar"), "-C", str(runtime)], check=True, timeout=900
            )
            python = runtime / "cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
            subprocess.run(
                [str(python), "-m", "venv", "--without-pip", str(ROOT / "envs/server")], check=True, timeout=120
            )
            subprocess.run(
                ["tar", "-xf", str(SHARED / "server-packages.tar"), "-C", str(ROOT / "envs/server")],
                check=True,
                timeout=1800,
            )
            subprocess.run(["tar", "-xf", str(SHARED / "cuda-compat.tar"), "-C", str(ROOT)], check=True, timeout=300)
            marker.write_text(
                json.dumps({"bundles": transfer["bundles"], "source": "verified yt runtime copy"}, indent=2)
            )
        state["status"] = "staging_local_weights"
        save()
        with (run / "local-weights.log").open("w") as log:
            child = subprocess.Popen(
                ["/root/miniconda3/bin/python", str(ROOT / "scripts/cache_jy_weights.py")],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if child.wait(timeout=7200):
                raise RuntimeError("Local model-cache verification failed")
        local_weights = json.loads((ROOT / "local-weights-status.json").read_text())
        if local_weights["status"] != "verified" or local_weights["revision"] != transfer["revision"]:
            raise RuntimeError("Local model-cache manifest mismatch")
        state["local_weights"] = local_weights
        gpus = inventory()
        if [g["index"] for g in gpus] != list(range(8)):
            raise RuntimeError("Expected exactly eight physical GPUs")
        state["inventory_before"] = gpus
        state["status"] = "cuda_preflight"
        save()
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=next(g["uuid"] for g in gpus if g["index"] == args.gpus[0]),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            VLLM_CACHE_ROOT=str(ROOT / "cache/vllm"),
            TMPDIR=str(ROOT / "cache/tmp"),
        )
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        env["LD_LIBRARY_PATH"] = (
            str(ROOT / "cuda-compat-13/usr/local/cuda-13.0/compat") + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        python = ROOT / "envs/server/bin/python"
        with (run / "gpu-preflight.log").open("w") as log:
            child = subprocess.Popen(
                [str(python), str(ROOT / "scripts/gpu_preflight.py")],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if child.wait(timeout=180):
                raise RuntimeError("Isolated CUDA/BF16/Triton preflight failed")
        current = inventory()
        gpus = [next(g for g in current if g["index"] == index) for index in args.gpus]
        # The user explicitly accepted sharing. Require ample headroom beyond our
        # fixed allocation; do not interpret another process's idle snapshot as ownership.
        for g in gpus:
            remaining = g["total_mib"] * (1 - args.memory_fraction) - g["used_mib"]
            if remaining < 16384:
                raise RuntimeError(f"GPU {g['index']} has insufficient headroom; other jobs untouched")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", 8015)) == 0:
                raise RuntimeError("Port 8015 occupied; will not stop its owner")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(g["uuid"] for g in gpus)
        command = [
            str(python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            local_weights["directory"],
            "--safetensors-load-strategy",
            "prefetch",
            "--served-model-name",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            "8015",
            "--tensor-parallel-size",
            str(len(args.gpus)),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "260096",
            "--gpu-memory-utilization",
            str(args.memory_fraction),
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "4096",
            "--limit-mm-per-prompt",
            '{"image":1016,"video":0}',
            "--mm-processor-kwargs",
            '{"max_pixels":262144}',
            "--skip-mm-profiling",
            "--reasoning-parser",
            "qwen3",
            "--generation-config",
            "vllm",
        ]
        state.update(
            status=f"starting_tp{len(args.gpus)}", server_command=command, visible_gpu_uuids=env["CUDA_VISIBLE_DEVICES"]
        )
        save()
        with (run / "server.log").open("w") as log:
            server = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
            )
        (ROOT / "server.pid").write_text(str(server.pid) + "\n")
        state["server_pid"] = server.pid
        save()
        deadline = time.monotonic() + 3600
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"TP{len(args.gpus)} server exited: {server.returncode}; see server.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8015/v1/models", timeout=5) as response:
                    models = json.load(response)
                if any(m["id"] == MODEL and m.get("max_model_len", 0) >= 260096 for m in models["data"]):
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Server readiness deadline exceeded")
            time.sleep(5)
        (run / "models.json").write_text(json.dumps(models, indent=2))
        state["status"] = "vision_xhigh_smoke"
        save()
        image = (SHARED / "scripts/vision-smoke-agentview.png").read_bytes()
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Briefly describe the visible robot and main objects in two sentences. Do not claim task success or execute actions.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()},
                        },
                    ],
                }
            ],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"},
        }
        (run / "vision-smoke-request.json").write_text(json.dumps(payload))
        req = urllib.request.Request(
            "http://127.0.0.1:8015/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=1200) as response:
            answer = json.load(response)
        (run / "vision-smoke-response.json").write_text(json.dumps(answer, indent=2))
        state["smoke_seconds"] = time.monotonic() - start
        state["smoke_usage"] = answer.get("usage")
        if answer["choices"][0]["finish_reason"] != "stop" or not answer["choices"][0]["message"].get("content"):
            raise RuntimeError("Vision smoke incomplete; not a robot evaluation")
        state.update(status="ready", inventory_after=inventory(), server_left_running=True)
    except (Exception, KeyboardInterrupt) as exc:
        state.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
            error=f"{type(exc).__name__}: {exc}",
        )
        stop_child(child)
        stop_child(server)
    finally:
        save()


if __name__ == "__main__":
    main()
