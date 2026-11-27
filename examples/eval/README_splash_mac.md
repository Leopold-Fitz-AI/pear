# Local LIBERO diagnostic evaluation with Splash on macOS

This small sequential runner uses the existing PEAR `LiberoEnv` and
`MotionController`. Only Linux's Mesa/EGL binding is replaced with macOS GLFW.
Task initial states, seed, camera rotation, instruction mapping, native action
budgets and success/reward checks stay PEAR's. No Ray or CUDA is used.

## Installed on this machine

- Environment: `~/.local/share/pear-sim/venv` (Python 3.10.22).
- LIBERO: `~/.local/share/pear-sim/LIBERO`, commit
  `8f1084e3132a39270c3a13ebe37270a43ece2a01` (the repository's pin).
- Dependencies installed from `.[libero]` with
  `scripts/install/sim/libero/constraints.txt`.
- LIBERO paths: `~/.local/share/pear-sim/config/config.yaml`.
- Model server: configured for Splash at `http://127.0.0.1:8000/v1`; currently stopped.
- Model: `mit-han-lab/Qwen3.8-27B-PEAR-NVFP4`, with vision enabled.

Run from the PEAR repository:

```bash
~/.local/bin/pear-splash status

# Rendering-only check. Output directory must not already exist.
~/.local/share/pear-sim/venv/bin/python examples/eval/eval_libero_splash.py \
  --smoke --trials 1 --output "$HOME/.local/share/pear-sim/runs/smoke-$(date +%Y%m%d-%H%M%S)"

# Same task, three different initial states (trial IDs 0, 1, 2).
~/.local/share/pear-sim/venv/bin/python examples/eval/eval_libero_splash.py \
  --suite libero_goal --task-id 0 --trials 3 --max-turns 30 \
  --effort low --max-tokens 8192 --thinking-budget 6144 \
  --output "$HOME/.local/share/pear-sim/runs/goal0-$(date +%Y%m%d-%H%M%S)"
```

Task 0 of LIBERO-Goal: **open the middle drawer of the cabinet**.
Task IDs 0..9 and suites libero_goal/libero_spatial/libero_object/libero_10
can be selected. A normal macOS desktop session is needed for GLFW.

## Failure diagnosis and optional assisted protocol

See [the evidence-based failure analysis](FAILURE_ANALYSIS_splash_mac.md). Offline
replays found no middle-handle contact; a privileged scripted positive control
opened the drawer in 101–102 native steps. This is NOT a PEAR autonomous success.
No model service is needed to run `diagnose_libero_splash.py` (see that report).

New optional flags are `--control-guidance`, `--measured-feedback`,
`--temperature`, `--top-p`, and `--reject-capped-actions`. Original prompt/feedback
behavior and greedy defaults remain unchanged. Added guidance and measured
feedback are clearly labeled **ASSISTED**, not an unmodified-policy evaluation.
Feedback never exposes object coordinates or contact truth to the model.

If explicitly choosing a new assisted experiment later, after manually starting
the model service, a bounded single-initial-state configuration is:

```bash
~/.local/share/pear-sim/venv/bin/python examples/eval/eval_libero_splash.py \
  --suite libero_goal --task-id 0 --trials 1 --max-turns 12 \
  --effort low --max-tokens 8192 --thinking-budget -1 \
  --temperature 1.0 --top-p 0.95 --control-guidance --measured-feedback \
  --output "$HOME/.local/share/pear-sim/runs/assisted-$(date +%Y%m%d-%H%M%S)"
```

This configuration has not yet been model-evaluated. Twelve turns is an action
count limit, not a wall-time guarantee. The response cap still applies, and
truncated outputs remain non-executable. When using a finite thinking budget,
`--reject-capped-actions` stops with `thinking_budget_exhausted` before executing
an answer whose reported reasoning-token count reaches that budget. The count is
a usage-based indicator, not an independent server-side forced-closure event.

## Calibrated visual references (opt-in)

`--calibrated-overlays` adds a transparent base-XY grid on the **reference plane
z=0 cm**, a red measured TCP projection, and magenta projections of the same tool
XY at reference heights 0/5/10 cm. It also adds small-lift verification and
high-clearance placement/push-contact guidance. The plane is not a depth sensor,
object detector, support-surface estimate or collision map. Raised objects have
parallax. The feature reads only robot kinematics and camera calibration, never
task-object poses, masks, contacts, or goal predicates.

Camera calibration is captured at image-render time, including initial sensor
sampling, rather than after subsequent physics steps. Raw input images remain as
`NNN-<camera>.raw.png`; annotated model inputs retain `NNN-<camera>.png`.
`NNN-camera-references.json` records the calibration used. Native-action video
replay verifies the raw images. Resume requires the same overlay setting and
reconstructs the annotated images for pixel-exact checks. Original defaults and
upstream prompts/controllers are unchanged.

This is another **ASSISTED** protocol, not a paper replication. See
[calibration investigation and validation](CALIBRATION_FIX_splash_mac.md).

## Five-task, single-attempt batch

`eval_libero_splash_batch.py` runs five distinct tasks sequentially: goal 1 (bowl
on stove), goal 5 (push plate), goal 7 (turn on stove), object 0 (soup into basket),
and spatial 2 (center bowl onto plate). Each uses initial state 0 exactly once,
with no retries. The fixed profile is assisted feedback/guidance, low effort,
8192 response tokens, no forced thinking cap, temperature 1.0, top_p 0.95,
and at most 30 turns. The original per-task native-step limits remain unchanged.

Start the model service explicitly first, then invoke:

```bash
~/.local/share/pear-sim/venv/bin/python examples/eval/eval_libero_splash_batch.py \
  --output /path/to/new-batch --case-seconds 900
```

To select a smaller set without repeats, use, for example,
`--tasks libero_spatial:2 libero_goal:1`. Add `--calibrated-overlays` to enable the
calibrated references for every selected case. No omitted task is run.

Optionally pass `--stop-server-pid <launcher-pid>` when that server was started
specifically for this batch. Cleanup stops it only if the saved launcher PID
still matches. A SIGTERM to the batch process also terminates its active evaluator
and triggers that cleanup; do not use SIGKILL if you want cleanup to run.

`batch.json` and `REPORT.md` update after every attempted task. Per-task logs and
original evaluator artifacts remain below the batch directory. Wall-time
interruptions are recorded as `wall_limit`, with `STOPPED.md` and a last-progress
snapshot, never converted into a completed environment failure. This is a
five-task exploratory diagnostic, not a whole-suite estimate.

## Records and interpretation

`settings.json` records the settings. Each trial directory holds:

- `NNN-agentview.png`, `NNN-wrist.png`: actual model input frames.
- `NNN-response.json`: server response, reasoning, timings and usage.
- `NNN-transition.json`: requested action, actual native actions and rewards.
- `native-actions.json`: replay history.
- `summary.json`: outcome, wall time, model turns, native steps, thinking-cap hits and protocol mode.
- New transition files include measured motion feedback even when it is not sent to the model.

The run-level `summary.json` is updated after every completed trial. Its
success rate is successes / completed attempts, including failed/error attempts;
`status_counts` separates the outcomes. While running, check completed_trials against
requested_trials; `progress.json` tracks the active trial. Render-only checks have no
success rate. `server-models.json` captures server identity, vision and context limits.

Outcomes distinguish simulator success, simulator step budget, turn limit,
model output limit and errors (including context exhaustion and malformed actions).
The runner retains all conversation images, final answers **and reasoning**, never
silently truncates history, and never retries with changed settings. A model's claim
of success is not accepted: only PEAR's environment reward and done status count.

### Split-response compatibility (simulator only)

PEAR's original Qwen parser takes the last JSON block after the last `</think>`;
when the marker is absent it searches the entire generation. Splash splits the
same kind of generation into `reasoning_content` and `content`. The adapter:

1. Uses normal final content whenever present; it never repairs an invalid final
   answer from a speculative action mentioned in reasoning.
2. For empty final content, requires `finish_reason=stop`, all completion tokens
   reported as reasoning, no closing marker, and a terminal fenced JSON action.
   That action must pass exact schema, numeric finiteness and duplicate-key checks.
3. Saves `NNN-parsed.json` with action source and normalized history. The count
   `reasoning_only_actions` makes this compatibility path visible in each trial.
4. Rejects truncated responses even if they already contain a JSON block. Unknown
   finish reasons, HTTP errors, context exhaustion and parsing errors remain distinct.

This is a simulator adapter, not a change to real-robot action parsing. Normal
content follows PEAR's last-block rule; the reasoning-only fallback is more restrictive.
No new action is invented. Historical raw responses remain unchanged.

## Limits

This is **not** the paper's benchmark: NVFP4 rather than BF16, a selected task,
three initial states rather than 50/task, greedy sampling, low rather than xhigh
reasoning, 8192 output tokens/turn, a 6144 thinking-token cap, a 30-turn cap and
128K server context. The thinking cap is an explicit Splash intervention, not a
paper-matched setting; `--thinking-budget -1` disables it. It reserves room for a
final action, but does not guarantee a valid response. Startup checks require a
vision-capable model, at least 128K context, and enough image slots for both cameras
at every turn (Splash currently permits 64, so at most 32 turns with full history).
Actual tokenized history may still exceed context; this is recorded, never hidden.
The HTTP timeout defaults to 900 seconds to allow a cold long-history prefill.
Increasing budgets may require restarting Splash with more context and memory.
Do not infer a whole-suite success rate from this subset. No demonstration dataset
is required for these tasks; the pinned repository supplies scenes/initial states.

## Reports and observation videos

After a run, build a report and the dual-camera observation slideshows:

```bash
~/.local/share/pear-sim/venv/bin/python examples/eval/report_libero_splash.py \
  --run /path/to/run --video
```

For a background evaluation, add `--watch-pid <evaluation-pid>` to wait for all
requested trials (up to four hours, without stopping the evaluation). Add
`--compare /path/to/earlier/comparison.json` for the earlier low-effort aggregate.
The watcher writes `REPORT.md`, `comparison.json` and per-trial videos only after
completion; premature process exit or watcher timeout writes `report-error.txt`
instead of inventing a final result. Videos are observation slideshows, not motion
recordings. A fresh xhigh evaluation uses the same runner with `--effort xhigh`;
keep other budgets unchanged and record the explicit thinking cap.

## Audited recovery from context exhaustion

A context-limit interruption can be resumed into a **new** output directory after
increasing the server context. This is continuation of the same trial, not a new
independent trial or a silent retry. Completed prefix files are copied; the source
is never overwritten. The runner checks experiment settings, contiguous transitions,
actions against raw responses, native step count, replayed state, and pixel-exact
camera frames at the interrupted request. It does not execute an action twice or
trim any history. Only context-limit interruptions are currently supported.

```bash
~/.local/share/pear-sim/venv/bin/python examples/eval/eval_libero_splash.py \
  --resume-from /path/to/interrupted-run --trial-start 2 --trials 1 \
  --output "$HOME/.local/share/pear-sim/runs/resumed-$(date +%Y%m%d-%H%M%S)"
```

Specify the original model/suite/task/effort/budgets if not the defaults. Sampling,
guidance, feedback and capped-answer rejection settings must also match. The
continuation's `seconds` measures the resumed invocation, not the original prefix;
`resumed_from` and `resumed_turns` identify the provenance. Do not count both the
interrupted record and its continuation as separate trials in aggregate statistics.

For the first repaired background run on this machine:

```bash
tail -f "$HOME/Library/Logs/pear-sim/eval-fixed.log"
cat "$HOME/Library/Logs/pear-sim/latest-run.txt"
```

Unit checks:

```bash
~/.local/share/pear-sim/venv/bin/python -m pytest -q tests/eval/test_libero_splash_on_cpu.py
```
