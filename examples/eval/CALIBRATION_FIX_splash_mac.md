# Camera calibration investigation and bounded validation

## Evidence and what it does NOT prove

The previous five-task assisted batch obtained one environment success (stove switch) and four native-budget failures. Offline replay showed different failure stages:

- Goal 1 bowl-to-stove: a real rim grasp/lift eventually occurred, but the model descended when the bowl was still approximately 17.9 cm horizontally from the cooking-region center.
- Goal 5 plate push: the plate moved approximately 5.9 cm in the wrong direction, then stayed almost stationary while the hand repeatedly moved.
- Object 0 soup: repeated closing/lifting did not produce sustained object lift (maximum height gain 1.4 cm).
- Spatial 2 center bowl: the first closing pose was roughly 10.5 cm short of the bowl in x; no sustained lift occurred (maximum height gain 0.3 cm).

These are failures of metric visual grounding, contact selection, and verification of object motion. Gripper-gap readings alone cannot distinguish a thin-rim grasp from empty closure. The added measured-motion feedback reports robot motion, not object motion.

A geometric camera probe in spatial task 2 initial state 0 found:

- The wrist optical image center is approximately `(255.5, 255.5)` pixels.
- The measured tool point projects to approximately `(255.5, 427.5)` pixels.
- The same tool XY projected on reference z=0 is approximately `(255.7, 286.9)` pixels.
- At that reference point, +1 cm base x moves about 9 pixels UP in the wrist image; +1 cm base y moves about 9 pixels RIGHT.
- In agentview, +1 cm base x moves approximately 3.5 pixels DOWN and +1 cm base y approximately 5.5 pixels LEFT.

Thus the upstream axis-direction prompt is correct; there is no evidence of an x/y sign inversion. However, wrist image center is not the tool/grasp point, and pixel-to-centimeter scale depends on depth. The original prompt supplies directions but no metric projection reference. The robot XML independently places the wrist camera 5 cm off the hand-frame origin.

**The offset is normal camera geometry, not a demonstrated simulator bug.** Supplying calibration is an engineering mitigation for a policy that is failing to infer that geometry. It does not prove that calibration alone is the ultimate cause. NVFP4, low effort, checkpoint capability and image interpretation remain confounded without matched comparisons.

## Changes

`examples/eval/libero_camera_reference.py` implements:

1. Camera calibration capture at actual image-render time (not after another physics step).
2. Base-centimeter to PEAR-image projection respecting MuJoCo's -z viewing direction, bottom-up OpenGL output, and PEAR's 180-degree image rotation.
3. Transparent reference-plane metric grid plus robot-only tool/height markers.
4. Explicit warnings about parallax, depth uncertainty, camera/tool offset, and the need to verify object motion after short test lifts and test pushes.

`eval_libero_splash.py --calibrated-overlays` opts into these assisted inputs. Raw images and calibration metadata are saved, annotated images remain the exact model inputs, and resume disallows changing this protocol. Native-action replay verifies raw frames for these runs. No upstream model weights, controller, task, initial state or success predicate is changed.

The renderer reads **only** camera intrinsics/extrinsics and robot base/tool kinematics. It never reads task-object positions, instance masks, contact predicates or target regions. The grid's z=0 is a fixed reference plane, not measured object depth or a claimed table surface. A marker does not certify visibility, collision-free space or a grasp.

A separate termination bug was fixed: when the last allowed native step returns `reached`, the local evaluator now terminates at the known native budget instead of making another model request whose rollout executes zero steps. Actual environment success is preserved. This avoids wasted inference but does not rescue unsuccessful trajectories.

## Validation protocol

A rendering-only smoke check and CPU unit tests cover projection signs/scaling, camera offset, rigid-transform invariance, rejection of behind-camera points, deterministic overlays without raw-image mutation, robot/camera-only calibration access, protocol resume guards, exact native-budget termination, and selected-task batch behavior.

Two previously unsuccessful tasks were tested once each, same initial state 0:

- `libero_spatial:2` — center bowl onto plate (grasp/localization failure)
- `libero_goal:1` — bowl onto stove (destination/placement failure)

Same selected model, low reasoning effort, temperature 1.0, top_p 0.95, 8192-token response cap, no forced thinking cap, at most 30 turns, original native budgets, 900-second per-case wall limit, no retries. The only planned policy additions versus the previous assisted batch are the calibration references and their guidance; the zero-step terminal-budget fix may remove an otherwise redundant final request.

Run: `~/.local/share/pear-sim/runs/calibrated-validation-20261115-224857/`.

### Actual results

| Task | Previous assisted attempt | Calibrated assisted attempt |
|---|---|---|
| Bowl onto stove | Failure, 21 turns / 300 steps / 335.1 s | **Environment success**, 9 turns / 138 steps / 99.1 s |
| Center bowl onto plate | Failure, 19 turns / 220 steps / 668.1 s; maximum bowl lift 0.3 cm | **Still unsuccessful**, 18 turns / 220 steps / 204.1 s; real grasp and maximum bowl lift 13.77 cm |

The successful bowl-to-stove episode was independently replayed. All nine boundary robot states and both raw input camera frames at each turn match the saved originals, and the replay's environment success is true. Video and verification metadata are in `libero_goal-01/verified-replay/`. Success is LIBERO's existing On predicate; no new release-and-retreat requirement was added.

In the still-unsuccessful bowl-to-plate case, the model actually transported the bowl to within approximately 0.67 cm of the plate center in XY at the end of zero-indexed turn 14. It then incorrectly reasoned that more +x motion was needed, moved away, and descended off-center before the native budget ran out. This is important: calibration helped this sampled trajectory reach grasp/transport, but did not make object-relative placement verification reliable. Its final outcome remains a failure, not a partial success counted as success.

The same model, initialization, native budget and sampling parameters were used, but stochastic outputs and both visual/guidance changes prevent causal attribution to a single component. This is an exploratory assisted-policy comparison, not evidence of a statistically established success-rate improvement or paper replication. Do not overwrite or merge these results with the previous batch. Plate pushing and soup grasping have **not** been revalidated with this feature.

`comparison-with-previous.json` and `offline-failure-analysis/` preserve the measured comparison. The code passed **72 CPU tests** and Ruff checks. The evaluator and Splash server exited after the two attempts; neither remains running.

## What remains unresolved

The model still sometimes compares image positions across different depths as though they were on one plane, misreads grid positions, and substitutes a speculative explanation for a verified object-motion check. Overlays provide calibrated references, not reliable object tracking, depth, or a grasp detector. More prompts cannot be claimed to guarantee a fix. A stronger next engineering step would require an explicit perception/geometry and object-motion verification layer (using observable sensor data, not simulator object poses), or a matched higher-precision policy comparison. Neither is implemented or validated here; do not claim all four task failures have been repaired.
