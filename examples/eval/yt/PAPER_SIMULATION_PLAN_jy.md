# jy: paper simulation experiment rollout

## Scope and reference recipes

The repository implements five robotic simulation benchmarks (`examples/eval/README.md`):

| Benchmark | Full recipe episodes | Current jy state |
|---|---:|---|
| LIBERO (Goal, Spatial, Object, Long) | 2,000 | Runtime SHA-verified; first four-suite subset launched and executing real native actions |
| LIBERO-Plus | 10,030 | Not run; separate pinned checkout, assets, environment and camera/noise handling required |
| LIBERO-Pro | 10,000 | Not run; separate pinned checkout, perturbation task assets and environment required |
| RoboDojo pickup + align | 328 | Not run; official checkout, Isaac container/runtime and Kit extensions are not present in this deployment |
| RoboLab-120 | 120 | Not run; pinned RoboLab, Isaac Sim/Lab 5.1 environment, LFS and remote scene assets required |

Total: **22,478 episodes**. A four-episode subset is not the full experiment. The official recipes target distributed Ray/veRL clusters and include separate simulator resources; they cannot simply be labeled complete after an API-server smoke test. GPU/driver/RTX compatibility for Isaac must be checked independently of the successful CUDA/model tests. No global driver upgrade or existing-container modification is authorized or performed here.

Docker is reachable on jy, but the available image inventory did not contain the required Isaac runtime. PEAR's installation guide explicitly says the RoboDojo/RoboLab setup guides are forthcoming. The RoboLab installer and asset checksums are available; the RoboDojo installer by itself prepares only its Ray environment/graphics libraries, not the official simulator checkout and image.

## First executable phase

Use the existing optimized **jy TP=4 service on GPUs 0,2,3,4**, not another model replica.

The first supervised test is task 0 / initial state 0 from each of:

1. `libero_goal`
2. `libero_spatial`
3. `libero_object`
4. `libero_10` (Long)

Each is attempted once. No automatic retry, privileged target coordinates, oracle motion, calibration overlays, extra controller guidance, or measured-motion feedback is supplied. Success must be returned by the original environment, not inferred from the model's words. Preserve raw images, reasoning/final responses, parsed actions, all native actions, per-turn robot states and terminal summaries.

- Fixed authored final PEAR checkpoint, BF16. **Not** the paper's LIBERO-only checkpoint.
- Original task/system prompts and original PEAR motion controller/success predicate.
- `xhigh`; temperature 1.0, top_p .95, top_k disabled, as `pear/configs/embodied.yaml`.
- Initial prompt budget 4096; subsequent episode token budget 256000, including injected observations and retained reasoning. Server context 260096.
- No 8192 per-turn response ceiling or forced thinking cap. Each request can use the remaining episode budget. Canonical `reasoning` fields are used for both token counting and generation, whose prompt counts must agree before an action is executed.
- Original native-step budgets: Goal 300, Spatial 220, Object 280, Long 520. A defensive 508-turn limit is consistent with the 1016-image server limit; actual context includes at least two image encodings and state/action text per turn.
- A **7200-second safety wall limit per case** is additional to paper budgets; an interrupted case is reported separately, not as a completed environment failure.
- Single-server API evaluator rather than the full Ray/veRL recipe. NVIDIA EGL rendering rather than paper CPU Mesa. These differences must remain visible in comparisons.

## Runtime isolation and preparation audit

The CPU-only Python 3.10 simulation environment, PEAR source and LIBERO checkout are copied from the already-tested yt setup in a SHA-256 checked archive. LIBERO's 1116 tracked files are also checked against the pinned revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`. The virtual environment and editable package mappings are relocated only inside `/root/pear-jy/`; no global package installation or default `~/.libero` configuration change is performed.

An initial preparation attempt rejected setuptools' editable mapping because it includes namespace resource paths as well as the main package path. This was corrected to relocate the validated package prefix consistently. That attempt stopped before any simulator or model episode ran; its error record is retained.

Before model evaluation, a separate render-only preflight must return `render_only`. Setup failures stop the batch; infrastructure failures during an episode stop remaining tasks rather than generating a chain of invalid results.

## Verified launch

Run `/root/pear-jy/runs/libero-paper-settings-smoke-20261116-045030/` passed render preflight and started Goal task 0 (middle drawer), followed by the other three suites. Initial audit: five completed action rounds / 56 native steps, sixth model request in flight. The first response generated 24,770 tokens (24,711 reasoning), naturally ending at EOS; subsequent saved responses also ended normally. Tokenizer and generation prompt counts matched on every audited request. This demonstrates that the earlier 8192-token per-turn ceiling is no longer applied; it does not yet establish task success.

## Status and lifecycle

- Workspace: `/root/pear-jy/`.
- Latest evaluation pointer: `/root/pear-jy/latest-libero-evaluation.txt`.
- Per-run `status.json`, `REPORT.md`, suite logs and case artifacts.
- Supervisor: `/root/pear-jy/scripts/run_jy_libero_smoke.py`.

```bash
ssh jy 'python3 /root/pear-jy/scripts/control_jy.py status'
# Stop only evaluation; retain the TP4 model service:
ssh jy 'python3 /root/pear-jy/scripts/control_jy.py stop-evaluation'
# Stop this evaluation and this model deployment; leave other jobs untouched:
ssh jy 'python3 /root/pear-jy/scripts/control_jy.py stop'
```

The model remains serving after evaluation. Completion of this phase does not automatically schedule 22,478 episodes or claim that the four other benchmark families were tested. Next phases need validated adapters/runtimes and explicit, separately reported benchmark coverage.
