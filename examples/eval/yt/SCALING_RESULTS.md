# A100 BF16 single-/dual-GPU speed comparison

Original run: `/root/pear-yt/runs/scaling-20261116-005738/` (server timezone).
Four-GPU follow-up, explicitly requested by the user: `/root/pear-yt/runs/scaling-20261116-012327/`.

## Results

Median of three measured requests per prompt and configuration, after one warmup:

| Configuration | Short input: 256-token response | Short input end-to-end tok/s | Long input: 256-token response | Long input end-to-end tok/s |
|---|---:|---:|---:|---:|
| Original single GPU, eager, OMP=4 | 21.828 s | 11.728 | 21.867 s | 11.707 |
| Optimized single GPU, compilation/CUDA Graph, CPU threads=1 | 9.035 s | 28.333 | 9.162 s | 27.942 |
| Optimized TP=2, GPU 0+4, NVLink | 5.326 s | 48.064 | 5.430 s | 47.146 |
| Optimized TP=4, GPU 0+1+4+6, **shared environment** | 3.237 s | 79.092 | 3.285 s | 77.926 |

The single-GPU change gives about **2.4x** throughput. TP=2 gives another **1.7x** versus optimized single GPU, or approximately **4.0–4.1x** versus the original eager configuration. The shared TP=4 follow-up achieved about **1.65x** TP=2, **2.8x** optimized single GPU, or **6.7x** the original eager configuration. These are observed benchmark ratios, not expected task-success improvements; TP=4 is not a controlled exclusive-GPU comparison.

Warm-cache median TTFT was 0.245/0.278 s (original), 0.173/0.155 s (optimized single), and 0.140/0.165 s (dual), for short/long prompts respectively. Streaming estimates of post-first-token decode rates were approximately 11.8, 28.3–28.8, and 48.4–49.2 tok/s. End-to-end throughput in the table is the primary metric; streaming chunk boundaries introduce small decode-estimation error.

## Conditions and limitations

- Same authored non-quantized PEAR checkpoint, BF16, no draft, no FP8 KV cache; same 128K context setting.
- Two identical LIBERO images in both prompts. Short input is 559 tokens; long input is 11,959 tokens with **synthetic repeated text**, not real multi-turn robot history.
- Greedy, thinking disabled, exactly 256 output tokens enforced with benchmark-only `ignore_eos`. This must not be copied into action evaluation.
- One warmup plus three sequential timed requests per prompt. Prefix and image caches are **warm**. These TTFT measurements are not cold-prefill latency for new history or new images.
- No robot action was executed; no new LIBERO success-rate result was produced.
- CPU threads and compilation/graph settings changed together, so this does not isolate each optimization's individual contribution.
- GPU 0 retained the pre-existing approximately 844 MiB of external contexts throughout. Other jobs were not stopped. GPU 4 was otherwise empty. This is a shared server, not a dedicated hardware laboratory.
- All eight GPUs have NVLink connectivity. Initially TP=4 was blocked by existing jobs. After the user explicitly requested trying four cards, the follow-up used 0,1,4,6, retaining other jobs. GPUs 1 and 6 had approximately 7.8 GiB and 2.4 GiB resident allocations and zero sampled utilization before startup. PEAR's memory budget was reduced from 0.90 to **0.45 per GPU** for this shared TP=4 test; a preflight required at least 16 GiB projected headroom per card. This is not process isolation or a guarantee against interference from other jobs.
- The follow-up `payloads.json` is SHA-256 identical to the original run. Same two prompts, fixed 256 outputs, warmup and three measurements. TP=4 median TTFT was 0.138/0.168 s; estimated post-first-token decode was 82.3/81.8 tok/s for short/long prompts.
- GPU allocation snapshots were saved after each measured request. At the end, the additional cards returned to their pre-existing allocations; no other process was stopped.

## Current service and audit artifacts

After the original measurement, GPU 4 was released and **optimized GPU-0-only serving restored** on port 8015. A fresh generation check produced all 256 tokens in 9.175 s. After the TP=4 follow-up, GPUs 1,4,6 were also released, optimized GPU-0-only serving was restored automatically, and `/v1/models` was verified. The follow-up status is `completed_benchmarks`; other GPU allocations returned to their initial values.

The first automatic restore attempt hit a process-ownership guard because a terminated, unreaped child has an empty `/proc/<pid>/cmdline`. This did not invalidate the already completed measurements. The guard was fixed and the single-GPU service restored separately; no benchmark repetitions were substituted. Original error state is retained in `status-before-recovery.json`; current `status.json` is `completed_benchmarks_recovered`, with `recovery.json` and recovery generation events.

Run artifacts include `payloads.json`, all raw SSE events, per-variant server logs, request timings, GPU inventory snapshots, and the original restore error. Later script revisions explicitly restart the eager baseline before measuring, so rerunning while an optimized server is resident cannot mislabel it as an eager baseline.

Inspect/stop using `python3 /root/pear-yt/scripts/control.py status|stop`. This now recognizes the scaling supervisor as well as the original deployment supervisor. No new benchmark or task run is scheduled automatically.
