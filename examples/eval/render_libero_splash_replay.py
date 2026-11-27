"""Render saved native actions continuously, without running the model again.

Unlike observation slideshows, this renders every 20-Hz simulator action. Check
saved boundary states and input-camera pixels, plus the final success predicate.
Model thinking time is omitted; the beginning/end have explicit display holds.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-pixels", action="store_true", help="Record pixel differences instead of requiring pixel-identical rendering; state and success checks remain mandatory")
    args = parser.parse_args()
    settings = json.loads((args.run / "settings.json").read_text())
    trial = args.run / f"trial-{args.trial_id:02d}"
    summary = json.loads((trial / "summary.json").read_text())
    paths = sorted(trial.glob("*-transition.json"))
    records = [json.loads(p.read_text()) for p in paths]
    if not records or any(p.name != f"{i:03d}-transition.json" for i, p in enumerate(paths)):
        raise ValueError("Missing or noncontiguous transitions")
    actions = [a for r in records for a in r["actions"]]
    if actions != json.loads((trial / "native-actions.json").read_text()) or len(actions) != summary["native_steps"]:
        raise ValueError("Saved native histories disagree")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"
    os.environ["LIBERO_CONFIG_PATH"] = settings["config_dir"]
    sys.path.insert(0, settings["libero_root"])
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    from pear.envs.libero.controller import MotionController
    from pear.envs.libero.env import LiberoEnv

    class LocalEnv(LiberoEnv):
        def _bind_rendering(self):
            from libero.libero.envs import OffScreenRenderEnv

            return OffScreenRenderEnv

    env = LocalEnv(settings["suite"], settings["task_id"], args.trial_id)
    checks = []
    pixel_checks = []
    step = 0
    fps = 20
    dual_path = args.output / "replay-dual-camera.mp4"
    main_path = args.output / "replay-agentview.mp4"
    try:
        observation = env._open_scene()
        sim = env.scene.sim
        if env.scene.env.control_freq != fps:
            raise ValueError("Expected 20-Hz native control")

        def cameras():
            # Render directly, without force-updating observation/controller caches.
            return [
                np.ascontiguousarray(sim.render(width=512, height=512, camera_name=name)[::-1, ::-1])
                for name in ("agentview", "robot0_eye_in_hand")
            ]

        def check_input_frame(turn, pixels):
            for name, array in zip(env.CAMERAS, pixels):
                suffix = ".raw.png" if settings.get("calibrated_overlays") else ".png"
                with Image.open(trial / f"{turn:03d}-{name}{suffix}") as saved:
                    expected = np.asarray(saved)
                    if expected.shape != array.shape:
                        raise ValueError(f"Input camera shape differs at turn {turn}, {name}")
                    delta = np.abs(expected.astype(np.int16) - array.astype(np.int16))
                    matches = bool(np.array_equal(expected, array))
                    fraction = float(np.any(delta != 0, axis=-1).mean())
                    pixel_checks.append({"turn": turn, "camera": name, "exact_match": matches,
                                         "mean_absolute_error": float(delta.mean()),
                                         "max_channel_error": int(delta.max()),
                                         "changed_pixel_fraction": fraction})
                    # Permit only sparse, tiny rasterization differences, never a different scene.
                    if args.audit_pixels and (fraction > 0.0001 or int(delta.max()) > 8):
                        raise ValueError(f"Pixel differences exceed replay tolerance at turn {turn}, {name}")
                    if not matches and not args.audit_pixels:
                        raise ValueError(f"Input camera pixels differ at turn {turn}, {name}")

        def labeled(pixels, turn, success):
            dual = Image.new("RGB", (1024, 560), "white")
            single = Image.new("RGB", (512, 560), "white")
            for i, pixel in enumerate(pixels):
                dual.paste(Image.fromarray(pixel), (512 * i, 48))
            single.paste(Image.fromarray(pixels[0]), (0, 48))
            text = f"{summary['instruction']} | native-action REPLAY | step {step}/{len(actions)}"
            detail = f"20 Hz | turn {turn}/{len(records)} | {'SUCCESS (environment)' if success else 'executing'} | thinking time omitted"
            for frame in (dual, single):
                draw = ImageDraw.Draw(frame)
                draw.text((8, 6), text, fill="black")
                draw.text((8, 25), detail, fill="green" if success else "black")
            return np.asarray(dual), np.asarray(single)

        with (
            imageio.get_writer(dual_path, fps=fps, codec="libx264", quality=8) as dual_writer,
            imageio.get_writer(main_path, fps=fps, codec="libx264", quality=8) as main_writer,
        ):
            # Initial model observation is the last stabilization-step sensor
            # sample, which can precede the end-of-step direct-render frame.
            pixels = [
                np.ascontiguousarray(observation[env.OBSERVATION_KEY_BY_CAMERA[name]][::-1, ::-1])
                for name in env.CAMERAS
            ]
            check_input_frame(0, pixels)
            dual, single = labeled(pixels, 0, False)
            for _ in range(fps):
                dual_writer.append_data(dual)
                main_writer.append_data(single)
            for turn, record in enumerate(records):
                if turn:
                    check_input_frame(turn, pixels)
                for action in record["actions"]:
                    observation = env._step(np.asarray(action))
                    step += 1
                    pixels = cameras()
                    success = bool(env.scene.check_success())
                    dual, single = labeled(pixels, turn + 1, success)
                    dual_writer.append_data(dual)
                    main_writer.append_data(single)
                actual = MotionController.state(env.scene, observation)
                if actual != record["state"]:
                    raise ValueError(f"Robot state mismatch at transition {turn}")
                checks.append({"turn": turn, "state_matches": True,
                               "input_pixels_match": all(c["exact_match"] for c in pixel_checks if c["turn"] == turn)})
            success = bool(env.scene.check_success())
            if success != summary["success"]:
                raise ValueError("Final environment success differs from saved outcome")
            Image.fromarray(dual).save(args.output / "final-frame.png")
            for _ in range(3 * fps):
                dual_writer.append_data(dual)
                main_writer.append_data(single)
        metadata = {
            "source": str(trial.resolve()),
            "native_steps": step,
            "fps": fps,
            "simulated_motion_seconds": step / fps,
            "display_hold_seconds": 4,
            "model_thinking_time_included": False,
            "model_called": False,
            "environment_success": success,
            "boundary_checks": checks,
            "pixel_checks": pixel_checks,
            "pixel_identical": all(c["exact_match"] for c in pixel_checks),
            "kind": "native_action_replay_not_original_recording",
            "videos": [str(dual_path), str(main_path)],
        }
        (args.output / "replay-verification.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2))
    except Exception as exc:
        # Do not leave a failed replay looking like a verified success video.
        for path in (dual_path, main_path):
            path.unlink(missing_ok=True)
        (args.output / "ERROR.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:
        env._close_scene()


if __name__ == "__main__":
    main()
