# yt / GPU 0 / non-quantized PEAR deployment

## Scope

The user selected `mit-han-lab/Qwen3.8-27B-PEAR`, not NVFP4. Fixed revision:
`65176e435e5fa80a059ebfafaf8215e6ad00a83e`.

The two weight shards total 55,635,071,360 bytes. They include authored BF16/F32 tensors; inference is configured as BF16 without weight quantization, without a draft model, and without FP8 KV cache. This is the final PEAR checkpoint, not a claim to reproduce the paper's LIBERO-only checkpoint result.

Server workspace: `/root/pear-yt`. Nothing in existing inference environments or other services was replaced. Model API is **localhost-only**, `http://127.0.0.1:8015/v1`, served name `mit-han-lab/Qwen3.8-27B-PEAR`. GPU 0 is an A100-SXM4-80GB, identified by UUID `GPU-3585d677-1b15-e7af-13c6-51d68c676f1f`; the model server uses that UUID as its sole CUDA-visible device. No boot-time service was installed.

## Current rollout status

Weights are fully downloaded and verified, and full-model BF16 serving is working. The first five-task evaluation ended with five wall-time interruptions, not five completed environment failures. Single-GPU optimization and a TP=2 speed comparison have since completed; see [SCALING_RESULTS.md](SCALING_RESULTS.md). An explicitly requested TP=4 follow-up also completed on GPUs 0,1,4,6 with existing jobs retained and a reduced memory budget; its shared-environment result is about 78–79 end-to-end tok/s. Optimized GPU-0-only serving has been restored; check live status before assuming a service is running.

- Server environment installed: Python 3.12.13, vLLM 0.28.0, torch 2.13.0+cu130, transformers 5.10.4.
- GPU 0 BF16 matrix multiplication and a real Triton JIT/launch test passed under the isolated CUDA 13 compatibility libraries.
- Simulator environment installed separately: Python 3.10.12, PEAR `.[libero]` constrained by the repository's LIBERO lock, CPU-only torch.
- LIBERO's 1116 tracked files match SHA-256 hashes of the pinned local checkout at `8f1084e3132a39270c3a13ebe37270a43ece2a01`. Existing identical server files were copied and verified; the existing checkout was not modified.
- EGL rendering smoke test passed on physical graphics device 0. The simulator requires numeric `CUDA_VISIBLE_DEVICES=0` because robosuite 1.4.1 rejects UUID masks for its numeric EGL device check. Simulator torch is CPU-only; model CUDA still uses the explicit GPU UUID.
- Mac/server initial stove agentview images have the same shape/orientation and mean absolute RGB difference 0.683/255, not pixel identity; rendering backend is an additional comparison difference.

## CUDA compatibility and isolation

The host driver reports 535.129.03 and its original CUDA driver API is 12.4. We did **not** upgrade the host driver or change global loader settings. An official NVIDIA `cuda-compat-13-0` Debian package (580.196.07) was SHA-256 verified and extracted under `/root/pear-yt/cuda-compat-13/`. Only the PEAR model server prepends its compatibility-library directory to `LD_LIBRARY_PATH`. The simulator does not inherit that override.

The initial CUDA 12.9/GitHub-wheel route was abandoned because the release-asset download was extremely slow. Its partial file and logs were retained. The functioning installed environment uses the standard PyPI CUDA 13 vLLM package. A CUDA 12.9 compatibility bundle also remains as a tested diagnostic artifact, not the active server runtime.

## Download and integrity

The server could not connect directly to huggingface.co. The downloader uses hf-mirror.com with **no implicit authentication token**, pinned to the official revision. `official-model-manifest.json` was obtained from the official Hugging Face API, not the mirror. After downloading, all files must match its size and Git-blob SHA-1 or LFS SHA-256:

- `model-00001-of-00002.safetensors`: `d9a150d119cf966ef8b797fa96fd911c88128c2a5b735a6592a941adda1aa790`
- `model-00002-of-00002.safetensors`: `4f0a55d9f3069ad67cfa60bc3058cf917fddbdf42be5911547b4073b02ac28bb`

The supervisor refuses to serve unverified files. Interrupted downloads retain Hugging Face's resumable staging files. Download attempt 1's status is retained; a subsequent resumed process explicitly disables implicit tokens. No validation episode ran during that restart.

## Automatic validation plan

`run_bf16_validation.py` waits for weights and environments, rechecks GPU 0 free memory and port 8015, starts only its own server, waits for `/v1/models`, then performs a simulator-render and model-vision preflight. Startup errors are recorded and only this supervisor's processes are stopped.

It then runs the same five tasks as the earlier Mac assisted batch, each once, initial state 0, no retries:

1. Goal 1: bowl onto stove.
2. Goal 5: push plate to stove front.
3. Goal 7: turn on stove.
4. Object 0: alphabet soup into basket.
5. Spatial 2: center bowl onto plate.

Settings: low effort, temperature 1.0, top_p .95, top_k disabled, 8192 response tokens, no forced thinking cap, 30 model turns, original per-suite native-step limits, 900 seconds per case, 128K context. Controller guidance and measured robot-motion feedback are enabled, matching the prior five-task assisted batch. **No calibrated overlays, object ground truth or scripted solution is supplied.** This first comparison is not an xhigh/full-paper evaluation.

The adapter explicitly sends effort to the authored chat template and preserves reasoning history. It accepts vLLM 0.28's `reasoning` response field as well as Splash's `reasoning_content`; conflicting fields are rejected. vLLM mode refuses a finite Splash-only thinking budget instead of silently ignoring it.

After successful deployment and completion of validation, the **model service is left running** for the requested server deployment. A pipeline error/interruption stops the server it owns. There is no automatic restart or task retry.

## Inspect and stop

From the Mac:

```bash
ssh yt 'python3 /root/pear-yt/scripts/control.py status'

# Stop this evaluation and model server, but leave an ongoing download intact:
ssh yt 'python3 /root/pear-yt/scripts/control.py stop'

# Also stop this model's downloader:
ssh yt 'python3 /root/pear-yt/scripts/control.py stop-all'
```

PID ownership is checked against `/proc/<pid>/cmdline` before signals are sent. Other ports, GPU jobs, and services are untouched.

Artifacts:

- `/root/pear-yt/download-status.json`: downloading/verifying/verified/error.
- `/root/pear-yt/latest-validation.txt`: current run directory.
- Per-run `status.json`, `REPORT.md`, per-case logs, raw responses, frames and native actions.
- `/root/pear-yt/logs/server.log`, `gpu-preflight.log`, `egl-smoke-index0.log`.
- `/root/pear-yt/server-requirements.txt`, `sim-requirements.txt`: actual installed versions.

The initial supervisor was stopped while still waiting, to correct robosuite's UUID/EGL incompatibility. Its interrupted preparation record remains. The replacement run is `/root/pear-yt/runs/bf16-matched-20261115-153502/` (server timezone; no episode retry).

For local API access once ready:

```bash
ssh -N -L 8015:127.0.0.1:8015 yt
# In a separate terminal:
curl http://127.0.0.1:8015/v1/models
```

Precision, inference backend, speculation, rendering backend, and possibly checkpoint conversion details differ from the Mac. A better result would implicate the local configuration as a contributor, not establish quantization as the sole cause. A poor result would not by itself prove the checkpoint incapable under the paper's longer/xhigh settings.
