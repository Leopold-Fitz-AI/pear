# jy: shared PEAR BF16 deployment

## Current state: four GPUs

At the user's request, the eight-GPU server has been replaced with **TP=4 on physical GPUs 0,2,3,4**. The service is ready on the same localhost port 8015, using the same BF16 weights and 260096-token context. Its memory target is 40% per selected GPU, with observed additional allocation around 32.9 GiB per selected card. Other GPU jobs were retained; this avoids GPU 1, which has an active external workload.

Current deployment: `/root/pear-jy/runs/deployment-20261116-032943/`.

Matched xhigh speed test: `/root/pear-jy/runs/tp4-xhigh-speed-20261116-033424/`. Request payloads are byte-identical to the previous eight-card xhigh test. Each prompt gets a warmup and three measured natural-EOS responses, with greedy sampling and an 8192-token safety cap; no response hit the cap.

| Input | Eight-GPU end-to-end tok/s | Four-GPU end-to-end tok/s | Observed difference |
|---|---:|---:|---:|
| Short, two images | 75.14 | 82.80 | +10.2% |
| Synthetic long, two images | 74.51 | 81.53 | +9.4% |

Four-card outputs contained 3619 total / 3024 reasoning tokens (short, 43.71 s), and 1213 total / 660 reasoning tokens (long, 14.88 s). The eight-card outputs were different lengths (1507 and 1737 total tokens), so the throughput improvement does **not** imply every complete answer finishes sooner. Memory targets and GPU selection also changed, and the host is shared: this is an operational comparison, not an isolated causal TP-scaling experiment. No robot evaluation ran as part of the speed test.

## Original eight-card deployment (historical)

The original server ran on **jy**, not yt, with eight tensor-parallel workers (TP0–TP7); it has now been replaced by the four-card configuration above.

- Model: `mit-han-lab/Qwen3.8-27B-PEAR`.
- Revision: `65176e435e5fa80a059ebfafaf8215e6ad00a83e`.
- BF16, no quantization, no draft, no FP8 KV cache.
- Tensor parallelism 8, physical GPUs 0–7, UUID masks recorded in the deployment status.
- Context: **260096** tokens, matching official script `4096 + 256000`.
- Compilation/CUDA Graph enabled, single concurrent sequence, 4096 batched prefill tokens.
- PEAR memory target: 25% per GPU. Observed additional allocation after smoke: about **20.7 GiB per GPU**, including overhead. This setting is not a hard bound on all future activation peaks.
- API: **localhost only**, `http://127.0.0.1:8015/v1`.
- xhigh vision smoke passed: 214 generated tokens, 144 reasoning tokens, 4.458 seconds, normal EOS. This is not a matched scaling benchmark or a robot success test.
- Successful run: `/root/pear-jy/runs/deployment-20261116-024730/`.
- yt's model and evaluation supervisors remain stopped.

Other jobs were retained. GPU 1 had active computation before and after deployment; other cards also had resident allocations. This is an explicitly authorized **shared deployment**, not exclusive hardware or a promise of no contention. Startup requires 16 GiB projected headroom beyond PEAR's memory target on every GPU.

## Paths and storage

- Runtime, logs and controls: `/root/pear-jy/`.
- Persistent shared weights and transfer bundles: `/vla-cd/pear-jy-deployment/`.
- Active model cache: `/dev/shm/pear-jy-deployment/models/Qwen3.8-27B-PEAR/`.

The local model cache occupies approximately **52 GiB RAM** in tmpfs. It is volatile across reboot; the shared copy is persistent. The loader uses tmpfs because the object-backed FSX-TOS shared mount was extremely slow under safetensors mmap loading, even with forced prefetching. The large `/nix` mount is read-only; no permissions, mounts or Nix store files were changed. Cache creation checks substantial physical RAM and tmpfs headroom, then streams and SHA-verifies every file against the official manifest.

Stopping the model frees its GPU allocations but does **not** delete the RAM cache or persistent weights. Cache cleanup should be a separate explicit operation after serving and staging have stopped.

## Environment and integrity

The independently installed yt environment's public packages and managed Python runtime were archived, SHA-256 checked after transfer, and installed into a new jy virtual environment. No credentials, global configuration, or global site-packages were copied or altered.

- Python 3.12.13, torch 2.13.0+cu130, vLLM 0.28.0, transformers 5.10.4.
- The host driver remains 550.127.05.
- CUDA 13 compatibility libraries are local to `/root/pear-jy/cuda-compat-13/`; only PEAR's processes prepend them to `LD_LIBRARY_PATH`.
- Actual BF16 CUDA matrix multiplication and Triton JIT/launch passed on jy before model serving.
- Weight hashes were verified on the shared copy and again while populating tmpfs.

An initial transfer status rename failed on the object-backed shared mount. Its error was preserved at `/root/pear-yt/jy-transfer-attempt-01.json`; verified weight files were reused, and status handling changed to an authoritative local atomic checkpoint plus a shared JSON snapshot whose readers tolerate incomplete writes. Earlier loading attempts and the rejected read-only-cache attempt are also retained in deployment run directories. No task result was discarded: no robot trial ran on jy during these attempts.

## Inspect / stop

```bash
ssh jy 'python3 /root/pear-jy/scripts/control_jy.py status'
ssh jy 'python3 /root/pear-jy/scripts/control_jy.py stop'
```

The control script checks `/proc/<pid>/cmdline` ownership before sending signals. It does not stop unrelated jobs or touch yt services. No boot-time autostart was configured.

A manual restart uses the existing verified runtime/cache and rechecks GPU memory and port ownership:

```bash
ssh jy 'nohup /root/miniconda3/bin/python /root/pear-jy/scripts/deploy_jy_tp8.py --gpus 0 2 3 4 --memory-fraction 0.40 >> /root/pear-jy/logs/deploy.log 2>&1 < /dev/null &'
```

Run that only after confirming this deployment is stopped; do not create multiple supervisors. The server remains running after a successful smoke test. A deployment error/interruption stops only processes started by that supervisor.

## Requests and evaluation scope

Send these request fields for the requested effort:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "preserve_thinking": true,
    "reasoning_effort": "xhigh"
  }
}
```

The smoke test's 8192-token response limit is only a finite deployment check, **not** the intended paper-style episode budget. A robot evaluator must use the episode-wide 256000-token continuation budget, including inserted observations and retained reasoning, with no extra 8192 per-turn ceiling. `/tokenize` and chat should both receive canonical `reasoning` fields; the two vLLM endpoints differ in their handling of legacy `reasoning_content` aliases.

A separate original-protocol, xhigh LIBERO four-suite smoke evaluation has now been launched; see `EXPERIMENTS.md` on the server or `PAPER_SIMULATION_PLAN_jy.md` in this repository. It is not the full paper benchmark. No success-rate claim follows from deployment or speed tests. The selected final PEAR checkpoint also differs from the paper's LIBERO-only checkpoint.
