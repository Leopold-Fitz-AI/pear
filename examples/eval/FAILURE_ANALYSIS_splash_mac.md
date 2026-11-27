# Drawer failure analysis — local NVFP4/Splash experiment

## What is established

The immediate failure was **failure to acquire the middle handle**, not insufficient pulling force or an impossible success predicate. Replaying recorded native actions reproduces the saved robot states. Simulator contacts and drawer joint travel expose the failure more clearly than the model's verbal interpretation.

| Trajectory | Replayed turns / steps | Closest tool-site to middle handle center* | Maximum middle opening | Thinking-cap hits |
|---|---:|---:|---:|---:|
| Fixed low, trial 0 | 27 / 300 | 15.12 cm | 0 cm | 0 |
| Fixed low, trial 1 | 22 / 300 | 11.26 cm | 0 cm | 1 |
| Resumed low, trial 2, including its original prefix | 27 / 300 | 12.28 cm | 0 cm | 0 |
| xhigh, completed trial 0 | 27 / 300 | 8.34 cm | 0 cm | 10 |
| xhigh, interrupted trial 1, saved prefix only | 13 / 119 | 12.41 cm | 0 cm | 7 |

*Approximate distance, using the exposed robot position rounded to 0.1 cm, sampled after native steps. This is center-to-center distance, not a surface-clearance or finger-tip distance. The resumed low trajectory is NOT an additional independent trial. The interrupted xhigh prefix is NOT a completed failure. No gripper contact with the middle handle's collision geometry was observed in these replays.

### 1. Collision and pose interpretation

In xhigh trial 0, action `003` requested `z=-14 cm`; actual descent was only about 3.4 cm, leaving a -10.6 cm position residual. The contact was with `wooden_cabinet_1_g9`, the cabinet's top panel. Subsequent actions continued to demand descent or close and pull despite not having reached the middle handle.

The initial middle handle center in this trial is approximately `(68.74, -13.74, 10.34) cm` in the robot base frame. These coordinates are **privileged diagnostic values**, not observations supplied to PEAR.

The model eventually tried a side-facing orientation at turn 15, reversed it at 16, and restored it at 17, but remained around `z=18.7 cm`, well above the middle handle. The problem is therefore not simply "it never rotates"; it fails to combine a collision-free approach, correct height, orientation, and grasp verification.

### 2. Gripper semantics misunderstanding

Response `004` explicitly reasons that width `-4` will reduce an approximately 7.5 cm gap to around 3 cm. The actual controller interprets only the sign: negative means close fully, positive means open fully, zero holds the previous actuator target. It does NOT implement relative gap positioning.

Response `005` observes an approximately 0.5 cm gap but speculates that a protrusion was probably pinched, then commands a 15 cm pull. The physical diagnostics show no middle-handle acquisition and no middle-drawer travel.

The adapter reused the upstream prompt; this is not evidence that the adapter reversed a gripper command. The prompt/action interface is insufficiently explicit for this model in these trajectories.

### 3. xhigh was often forcibly capped

The 6144-token thinking budget was reached in 10/27 responses of the completed xhigh trial, versus 0/27 in the first low trial. Splash documents this budget as forcibly closing thinking. Thus the xhigh result does not represent unconstrained xhigh deliberation.

This is a confound, not a demonstrated sole cause: the low runs also failed without regularly hitting the thinking cap. Simply raising effort cannot be assumed to solve the geometry problem.

### 4. Sampling and model-comparison limitations

The earlier local runs used greedy sampling (`temperature=0`); upstream evaluation uses `temperature=1.0`, `top_p=0.95`, and no top-k truncation. They also differ in checkpoint/quantization and token budgets. The paper reports BF16 evaluation and explicitly discusses NVFP4 accuracy degradation.

There is no matched BF16/NVFP4 comparison here. These observations cannot isolate quantization, Splash, or a general checkpoint weakness as the ultimate cause of the policy errors. No evidence currently establishes a coordinate-axis inversion, flipped-image bug, or impossible task.

## Positive control: same controller can solve the task

A **privileged scripted control**, separate from the model evaluator, uses the simulator handle center to execute:

1. Retreat to clear space in front of the cabinet.
2. Rotate to a side-facing grasp (`roll ≈ pi/2`).
3. Lower in free space to the middle handle height.
4. Approach the handle horizontally.
5. Close, then pull approximately 16 cm along +y.

On trial 0, this obtained an actual environment success and reward `1.0`, with middle-joint displacement about `-0.149 m`, in 102 native steps. A second check saving phase images succeeded in 101 steps; the extra observation/render updates alter exact step timing. Both are well within 300 steps. No joint was teleported and no success check was bypassed.

**This proves feasibility for that initial state through the existing controller. It is not PEAR autonomous success, not a learned policy, and not a benchmark success-rate improvement.** The oracle's coordinates and trajectory are not injected into model requests.

## Implemented mitigations

`eval_libero_splash.py` now supports independent, opt-in changes:

- `--control-guidance`: explicitly describes sign-only gripper control, orientation conventions, and the distinction between reaching a motion target and grasping an object.
- `--measured-feedback`: supplies requested versus achieved arm increments, residual error, native steps and execution outcome. Uses only already-observed robot states and controller reports, never object coordinates or contacts.
- `--temperature` / `--top-p`: makes sampling configurable and records it. Previous defaults remain for reproducibility; use `1.0` / `0.95` for the upstream temperature/nucleus settings. Splash's default top-k is disabled; this script does not override a separately configured server top-k.
- `thinking_cap_hits` in summaries and `--reject-capped-actions`: allows refusing a forced-thinking-cap response before executing its action. `--thinking-budget -1` disables that forced cap; the separate response-token cap still applies, and truncated responses remain non-executable.
- Resume checks reject changed guidance, feedback, sampling or rejection policy. Feedback is recomputed and checked during resume.
- Reports explicitly label added guidance/feedback as **ASSISTED**, so the modified protocol cannot be mistaken for the unmodified policy.

These mitigations are unit-tested. **No new model-driven episode has yet validated whether they improve autonomous task completion.** Original model success statistics remain unchanged.

## Artifacts on this machine

- Completed xhigh replay and pictured positive control:
  `~/.local/share/pear-sim/diagnostics/verified-20261115-182404/`
- First positive control without phase images:
  `~/.local/share/pear-sim/diagnostics/xhigh-20261115-181855/`
- Fixed low replay (third trajectory is only its original interrupted prefix):
  `~/.local/share/pear-sim/diagnostics/low-20261115-182626/`
- Completed third low trajectory replay, including its verified resumed suffix:
  `~/.local/share/pear-sim/diagnostics/low-resumed-20261115-182851/`
- Final successful privileged-control image:
  `~/.local/share/pear-sim/diagnostics/verified-20261115-182404/oracle-pull-agentview.png`

All original failed/interrupted records remain untouched. Splash and the model-driven evaluator were not restarted for this analysis.

## Reproduce offline diagnostics

No model server is needed. This command initializes the local simulator and briefly replays saved native actions; the optional oracle is explicitly privileged.

```bash
~/.local/share/pear-sim/venv/bin/python examples/eval/diagnose_libero_splash.py \
  --run ~/.local/share/pear-sim/runs/goal0-xhigh-20261115-171913 \
  --output ~/.local/share/pear-sim/diagnostics/new-diagnostic \
  --oracle-control
```

The output directory must not already exist. The script currently supports only `libero_goal` task 0; its oracle tests only initial state 0.
