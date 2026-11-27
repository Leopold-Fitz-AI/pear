"""Measure natural-EOS xhigh vision responses on the existing jy server.

No actions or service changes. The finite smoke-test cap is not an episode budget.
"""

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

from latency_helper import gpu_inventory


def measure(payload, path):
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        "http://127.0.0.1:8015/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = last = None
    usage = None
    finish = None
    events = []
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            line = line[6:].strip()
            if line == b"[DONE]":
                break
            event = json.loads(line)
            now = time.perf_counter()
            events.append({"seconds": now - started, "event": event})
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta", {})
                if any(delta.get(key) for key in ("content", "reasoning", "reasoning_content")):
                    if first is None:
                        first = now
                    last = now
    ended = time.perf_counter()
    path.write_text(json.dumps(events, ensure_ascii=False))
    if usage is None or first is None or last <= first:
        raise RuntimeError("Missing usage or streamed output")
    tokens = usage["completion_tokens"]
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return {
        "usage": usage,
        "reasoning_tokens": reasoning,
        "non_reasoning_tokens": tokens - reasoning if reasoning is not None else None,
        "finish_reason": finish,
        "ttft_seconds": first - started,
        "seconds": ended - started,
        "end_to_end_tokens_per_second": tokens / (ended - started),
        "approx_decode_tokens_per_second": (tokens - 1) / (last - first),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tp", type=int, choices=[1, 2, 4, 8])
    args = parser.parse_args()
    root = Path("/root/pear-jy")
    deployment_dir = Path((root / "latest-deployment.txt").read_text().strip())
    deployment = json.loads((deployment_dir / "status.json").read_text())
    if deployment["status"] != "ready" or (args.expected_tp and deployment["tp"] != args.expected_tp):
        raise RuntimeError("Expected deployment is not ready; refusing a mislabeled speed test")
    tp = deployment["tp"]
    run = root / "runs" / (f"tp{tp}-xhigh-speed-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir()
    (root / "latest-thinking-speed.txt").write_text(str(run) + "\n")
    source = Path("/vla-cd/pear-jy-deployment/scripts/scaling-payloads.json")
    prompts = json.loads(source.read_text())
    for payload in prompts.values():
        payload.pop("ignore_eos", None)
        payload.update(
            max_tokens=8192,
            chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"},
        )
    (run / "payloads.json").write_text(json.dumps(prompts))
    state = {
        "status": "running",
        "run": str(run),
        "tp": tp,
        "deployment": str(deployment_dir),
        "gpu_indices": deployment.get("gpu_indices"),
        "memory_fraction_per_gpu": deployment["memory_fraction_per_gpu"],
        "source_payload_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "inventory_before": gpu_inventory(),
        "results": {},
        "warmups": {},
        "metrics": {},
        "method": "Same two images and texts as prior test; greedy, xhigh, natural EOS, 8192 safety cap; "
        "one warmup then three measured requests per prompt; warm caches; no actions.",
    }

    def save():
        p = run / "status.tmp"
        p.write_text(json.dumps(state, indent=2))
        p.replace(run / "status.json")

    save()
    try:
        for name, payload in prompts.items():
            state["active_prompt"] = name
            save()
            warmup = measure(payload, run / f"{name}-warmup.json")
            state["warmups"][name] = warmup
            save()
            if warmup["finish_reason"] != "stop":
                raise RuntimeError("Warmup reached output cap; not a natural-EOS result")
            results = []
            for i in range(3):
                result = measure(payload, run / f"{name}-{i}.json")
                results.append(result)
                state["results"][name] = results
                save()
                if result["finish_reason"] != "stop":
                    raise RuntimeError("Measured request did not end naturally")
            state["metrics"][name] = {
                k: statistics.median(x[k] for x in results)
                for k in ["ttft_seconds", "seconds", "end_to_end_tokens_per_second", "approx_decode_tokens_per_second"]
            }
            state["metrics"][name]["completion_token_counts"] = [x["usage"]["completion_tokens"] for x in results]
            state["metrics"][name]["reasoning_token_counts"] = [x["reasoning_tokens"] for x in results]
            save()
        state.update(status="finished", inventory_after=gpu_inventory())
        state.pop("active_prompt", None)
    except Exception as exc:
        state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    print(json.dumps({"run": str(run), "status": state["status"], "metrics": state["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
