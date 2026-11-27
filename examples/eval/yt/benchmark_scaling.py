"""Serial, fixed-output latency benchmark. No robot actions, no model substitution.

Benchmark-only ignore_eos ensures equal output counts; this is NOT a quality test.
By default use otherwise idle GPUs. Explicit --shared-tp4 allows bounded sharing
on 0,1,4,6 without stopping existing jobs. Restore optimized GPU-0-only serving afterwards.
"""

import argparse
import base64
import hashlib
import json
import os
import signal
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/root/pear-yt")
MODEL = "mit-han-lab/Qwen3.8-27B-PEAR"
URL = "http://127.0.0.1:8015"


def gpu_inventory():
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
                ["index", "uuid", "used", "total", "util"],
                [int(v.strip()) if i != 1 else v.strip() for i, v in enumerate(line.split(","))],
            )
        )
        for line in text.splitlines()
    ]


def request(payload, output):
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    first = last = None
    usage = None
    pieces = []
    events = []
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            line = line[6:].strip()
            if line == b"[DONE]":
                break
            event = json.loads(line)
            now = time.perf_counter()
            events.append({"seconds": now - start, "event": event})
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if text:
                    if first is None:
                        first = now
                    last = now
                    pieces.append(text)
    end = time.perf_counter()
    output.write_text(json.dumps(events, ensure_ascii=False))
    if not usage or usage["completion_tokens"] != 256 or first is None or last <= first:
        raise RuntimeError(f"Invalid fixed-token response: {usage}")
    return {
        "ttft_seconds": first - start,
        "seconds": end - start,
        "usage": usage,
        "end_to_end_tokens_per_second": usage["completion_tokens"] / (end - start),
        "approx_decode_tokens_per_second": (usage["completion_tokens"] - 1) / (last - first),
        "output_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-tp4",
        action="store_true",
        help="Explicitly test only TP=4 on 0,1,4,6 with a 45%% per-GPU memory budget; retain existing jobs",
    )
    parser.add_argument("--evaluate-tp4", action="store_true", help="Run five xhigh robot tasks instead of speed tests")
    parser.add_argument("--resume-evaluation-run", type=Path)
    args = parser.parse_args()
    if args.resume_evaluation_run and not args.evaluate_tp4:
        parser.error("--resume-evaluation-run requires --evaluate-tp4")
    if args.evaluate_tp4:
        args.shared_tp4 = True
    run = (
        ROOT
        / "runs"
        / (("tp4-evaluation-service-" if args.evaluate_tp4 else "scaling-") + time.strftime("%Y%m%d-%H%M%S"))
    )
    run.mkdir()
    (ROOT / "latest-scaling.txt").write_text(str(run) + "\n")
    (ROOT / "scaling.pid").write_text(str(os.getpid()) + "\n")
    prior = Path((ROOT / "latest-validation.txt").read_text().strip())
    original = json.loads((prior / "status.json").read_text())["server_command"]
    state = {
        "status": "baseline",
        "run": str(run),
        "pid": os.getpid(),
        "variants": [],
        "inventory_before": gpu_inventory(),
        "shared_tp4": args.shared_tp4,
        "evaluate_tp4": args.evaluate_tp4,
        "method": "2 fixed image prompts, short and synthetic long text; one warmup + three measured requests each; "
        "256 forced output tokens, greedy, no thinking, no actions, warm repeated prefix/image caches. "
        "Decode rate estimated from first/last nonempty streaming chunks. This is not task success or thinking speed.",
    }
    child = None

    def save():
        p = run / "status.tmp"
        p.write_text(json.dumps(state, indent=2))
        p.replace(run / "status.json")

    def stop_owned():
        try:
            pid = int((ROOT / "server.pid").read_text())
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            return
        # A terminated, not-yet-reaped child has an empty /proc cmdline.
        if not cmd:
            if child is not None and child.pid == pid:
                child.poll()
            return
        if b"vllm.entrypoints.openai.api_server" not in cmd or str(ROOT / "models").encode() not in cmd:
            raise RuntimeError("Refusing to stop unknown server PID")
        os.killpg(pid, signal.SIGTERM)
        for _ in range(60):
            try:
                live = Path(f"/proc/{pid}/cmdline").read_bytes()
            except FileNotFoundError:
                live = b""
            if not live:
                break
            time.sleep(1)
        else:
            raise TimeoutError("Owned server did not stop cleanly; no forced kill")
        time.sleep(5)

    def start(indices, label, optimized=True, shared=False):
        nonlocal child
        inventory = gpu_inventory()
        selected = [next(g for g in inventory if g["index"] == i) for i in indices]
        for g in selected:
            allowance = 1536 if g["index"] == 0 else 256
            if shared and g["index"] in (1, 6):
                allowance = 12288
            reserve = g["total"] - g["used"] - 0.45 * g["total"]
            if g["used"] > allowance or g["util"] > 0 or (shared and reserve < 16384):
                raise RuntimeError(
                    f"GPU {g['index']} exceeds the agreed idle/memory bounds; existing jobs left untouched"
                )
        command = list(original)
        command[command.index("--tensor-parallel-size") + 1] = str(len(indices))
        if shared:
            command[command.index("--gpu-memory-utilization") + 1] = "0.45"
        if shared and args.evaluate_tp4:
            command[command.index("--max-model-len") + 1] = "260096"
            command[command.index("--limit-mm-per-prompt") + 1] = '{"image":1016,"video":0}'
            # Avoid profiling 1016 dummy images; substantial physical GPU headroom
            # is retained by the 45% cache/model budget on these shared cards.
            command.append("--skip-mm-profiling")
        if optimized:
            command.remove("--enforce-eager")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in selected),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            OMP_NUM_THREADS="1" if optimized else "4",
            MKL_NUM_THREADS="1" if optimized else "4",
            OPENBLAS_NUM_THREADS="1" if optimized else "4",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            VLLM_CACHE_ROOT=str(ROOT / "cache/vllm"),
        )
        env["LD_LIBRARY_PATH"] = (
            str(ROOT / "cuda-compat-13/usr/local/cuda-13.0/compat") + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        state.update(status="starting_" + label, active_command=command, active_gpus=indices)
        save()
        with (run / (label + "-server.log")).open("w") as log:
            child = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
            )
        (ROOT / "server.pid").write_text(str(child.pid) + "\n")
        state["server_pid"] = child.pid
        save()
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"{label} server exited: {child.returncode}")
            try:
                with urllib.request.urlopen(URL + "/v1/models", timeout=3) as response:
                    data = json.load(response)
                if any(m["id"] == MODEL for m in data["data"]):
                    return
            except OSError:
                pass
            time.sleep(5)
        raise TimeoutError(f"{label} startup timed out")

    scene = prior / "render-smoke/trial-00"
    images = []
    for view in ["agentview", "wrist"]:
        data = (scene / f"000-{view}.png").read_bytes()
        images.append(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}
        )
    prompts = {}
    for label, repeats in [("short", 0), ("long_synthetic", 600)]:
        text = (
            "Describe only the visible robot and scene in detail. Do not assert task success. "
            "This is a throughput test, not an instruction to move the robot.\n"
        )
        text += (
            "Historical placeholder: the robot observed the scene; no object coordinates or actions were provided.\n"
            * repeats
        )
        prompts[label] = {
            "model": MODEL,
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}, *images]}],
            "temperature": 0,
            "top_k": -1,
            "seed": 123,
            "max_tokens": 256,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    (run / "payloads.json").write_text(json.dumps(prompts))

    def bench(label, indices):
        state["status"] = "benchmarking_" + label
        row = {
            "name": label,
            "gpus": indices,
            "requests": [],
            "metrics": {},
            "inventory": gpu_inventory(),
            "server_command": list(state["active_command"]),
        }
        state["variants"].append(row)
        save()
        for name, payload in prompts.items():
            request(payload, run / f"{label}-{name}-warmup.json")
            results = []
            for i in range(3):
                result = request(payload, run / f"{label}-{name}-{i}.json")
                result["prompt"] = name
                result["inventory_after"] = gpu_inventory()
                row["requests"].append(result)
                results.append(result)
                save()
            row["metrics"][name] = {
                key: statistics.median(r[key] for r in results)
                for key in [
                    "ttft_seconds",
                    "seconds",
                    "end_to_end_tokens_per_second",
                    "approx_decode_tokens_per_second",
                ]
            }
            row["metrics"][name]["prompt_tokens"] = results[0]["usage"]["prompt_tokens"]
            save()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    save()
    interrupted = False
    try:
        # Do not label a previously optimized live server as an eager baseline.
        stop_owned()
        if args.shared_tp4:
            indices = [0, 1, 4, 6]
            start(indices, "tp4_shared_optimized", shared=True)
            if args.evaluate_tp4:
                from run_bf16_validation import terminate

                state["status"] = "evaluating_xhigh"
                state["method"] = "Five original-protocol LIBERO trials, xhigh, 260096 context; see evaluation status"
                save()
                evaluation_command = [str(ROOT / "envs/sim/bin/python"), str(ROOT / "scripts/eval_existing_bf16.py")]
                if args.resume_evaluation_run:
                    evaluation_command += ["--resume-run", str(args.resume_evaluation_run)]
                with (run / "evaluation-supervisor.log").open("w") as log:
                    evaluation = subprocess.Popen(
                        evaluation_command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    state["evaluation_pid"] = evaluation.pid
                    save()
                    try:
                        rc = evaluation.wait(timeout=5 * 7200 + 300)
                        if rc:
                            raise RuntimeError(f"Evaluation supervisor exited with {rc}; see its preserved status")
                    finally:
                        terminate(evaluation)
                state["status"] = "completed_evaluation"
            else:
                bench("tp4_shared_optimized", indices)
                state["status"] = "completed_benchmarks"
            save()
            return  # The finally block restores optimized GPU-0-only serving.
        start([0], "tp1_eager_baseline", optimized=False)
        bench("tp1_eager_baseline", [0])
        stop_owned()
        start([0], "tp1_optimized")
        bench("tp1_optimized", [0])
        stop_owned()
        start([0, 4], "tp2_optimized")
        bench("tp2_optimized", [0, 4])
        stop_owned()
        available = [g["index"] for g in gpu_inventory() if g["index"] != 0 and g["used"] <= 256 and g["util"] == 0]
        if len(available) >= 3:
            indices = [0, *available[:3]]
            start(indices, "tp4_optimized")
            bench("tp4_optimized", indices)
            stop_owned()
        else:
            state["tp4"] = {
                "status": "blocked_other_jobs",
                "inventory": gpu_inventory(),
                "reason": "Need three otherwise idle GPUs in addition to GPU 0; will not stop or share existing jobs.",
            }
            save()
        state["status"] = "completed_benchmarks"
        save()
    except KeyboardInterrupt:
        interrupted = True
        state.update(status="interrupted")
        save()
    except Exception as exc:
        state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        save()
    finally:
        previous_status = state["status"]
        try:
            stop_owned()
            if not interrupted:
                # Release every extra GPU and leave a single-GPU service available.
                try:
                    start([0], "restored_tp1_optimized")
                except Exception as exc:
                    state["optimized_restore_error"] = str(exc)
                    stop_owned()
                    start([0], "restored_tp1_eager", optimized=False)
                state["serving_gpu_indices"] = [0]
            state["status"] = previous_status
        except Exception as exc:
            state["status"] = "restore_error"
            state["restore_error"] = str(exc)
        save()


if __name__ == "__main__":
    main()
