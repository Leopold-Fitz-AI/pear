"""Offline drawer-task diagnostics. Privileged simulator state NEVER goes to the model.

Replays saved native actions, measures drawer travel and physical contacts. Optional
oracle control is an infrastructure test, NOT a PEAR success-rate measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--oracle-control", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path.home() / ".local/share/pear-sim"
    os.environ["LIBERO_CONFIG_PATH"] = str(root / "config")
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"
    sys.path.insert(0, str(root / "LIBERO"))
    import numpy as np

    from pear.envs.libero.controller import MotionController
    from pear.envs.libero.env import LiberoEnv

    class LocalEnv(LiberoEnv):
        def _bind_rendering(self):
            from libero.libero.envs import OffScreenRenderEnv

            return OffScreenRenderEnv

    settings = json.loads((args.run / "settings.json").read_text())
    if (settings["suite"], settings["task_id"]) != ("libero_goal", 0):
        p.error("Drawer diagnostic currently supports libero_goal task 0 only")
    reports = []
    for trial in sorted(args.run.glob("trial-*")):
        transitions = sorted(trial.glob("*-transition.json"))
        if not transitions:
            continue
        env = LocalEnv("libero_goal", 0, int(trial.name.split("-")[1]))
        try:
            observation = env._open_scene()
            sim = env.scene.sim
            middle_body = sim.model.body_name2id("wooden_cabinet_1_cabinet_middle")
            handle_geoms = [
                i
                for i in range(sim.model.ngeom)
                if sim.model.geom_bodyid[i] == middle_body
                and np.allclose(sim.model.geom_size[i], [0.00770, 0.00817, 0.04445])
            ]
            if len(handle_geoms) != 1:
                raise ValueError("Middle handle collision geometry not uniquely identified")
            handle_id = handle_geoms[0]
            base_pos, base_rot = MotionController._base(env.scene)

            def handle_base(sim=sim, handle_id=handle_id, base_rot=base_rot, base_pos=base_pos):
                return (base_rot.T @ (sim.data.geom_xpos[handle_id] - base_pos) * 100).tolist()

            joints = [n for n in sim.model.joint_names if n and "cabinet" in n]

            def snapshot(observation, sim=sim, joints=joints, handle_base=handle_base, env=env):
                return {
                    "drawer_qpos_m": {j: float(sim.data.get_joint_qpos(j)) for j in joints},
                    "middle_handle_base_cm": handle_base(),
                    "state": MotionController.state(env.scene, observation),
                    "success": bool(env.scene.check_success()),
                }

            initial = snapshot(observation)
            rows = []
            total_contacts = Counter()
            any_success = False
            peak_opening = 0.0
            min_handle_distance = float("inf")
            previous = initial["state"]
            for file in transitions:
                record = json.loads(file.read_text())
                contacts = Counter()
                for action in record["actions"]:
                    observation = env._step(np.asarray(action))
                    q = float(sim.data.get_joint_qpos("wooden_cabinet_1_middle_level"))
                    peak_opening = max(peak_opening, -q)
                    any_success = any_success or bool(env.scene.check_success())
                    tool = np.array([MotionController.state(env.scene, observation)["arm"][k] for k in ("x", "y", "z")])
                    min_handle_distance = min(min_handle_distance, float(np.linalg.norm(tool - handle_base())))
                    pairs = set()
                    for c in sim.data.contact[: sim.data.ncon]:
                        a, b = [sim.model.geom_id2name(int(g)) or "" for g in (c.geom1, c.geom2)]
                        if ("gripper" in a and "cabinet" in b) or ("gripper" in b and "cabinet" in a):
                            pairs.add(" / ".join(sorted([a, b])))
                    contacts.update(pairs)
                now = snapshot(observation)
                if now["state"] != record["state"]:
                    raise ValueError(f"Replay mismatch at {file}")
                response = json.loads(file.with_name(file.name.replace("-transition", "-response")).read_text())
                reasoning_n = response["usage"]["completion_tokens_details"].get("reasoning_tokens", 0)
                requested = record["requested_action"]["arm"]
                residual = {
                    k: round(previous["arm"][k] + requested[k] - now["state"]["arm"][k], 3) for k in ("x", "y", "z")
                }
                row = {
                    "turn": int(file.name[:3]),
                    "requested_action": record["requested_action"],
                    "outcome": record["outcome"],
                    "steps": record["executed"],
                    "position_residual_cm": residual,
                    "contacts_native_steps": dict(contacts),
                    "reasoning_tokens": reasoning_n,
                    "thinking_cap_reached": settings.get("thinking_budget", -1) > 0
                    and reasoning_n >= settings["thinking_budget"],
                    **now,
                }
                rows.append(row)
                total_contacts.update(contacts)
                previous = now["state"]
            report = {
                "trial": trial.name,
                "source": str(trial),
                "initial": initial,
                "turns": rows,
                "max_middle_drawer_opening_m": peak_opening,
                "min_tool_to_middle_handle_center_cm": min_handle_distance,
                "any_success": any_success,
                "thinking_cap_hits": sum(r["thinking_cap_reached"] for r in rows),
                "contact_native_steps_by_pair": dict(total_contacts),
                "middle_handle_geom": sim.model.geom_id2name(handle_id),
                "contact_geometry": {
                    n: {
                        "body": sim.model.body_id2name(sim.model.geom_bodyid[i]),
                        "local_position_m": sim.model.geom_pos[i].tolist(),
                        "half_sizes_m": sim.model.geom_size[i].tolist(),
                    }
                    for i, n in enumerate(sim.model.geom_names)
                    if n and any(n in pair.split(" / ") for pair in total_contacts)
                },
                "privileged_diagnostics_only": True,
            }
            reports.append(report)
            print(
                json.dumps(
                    {k: v for k, v in report.items() if k not in ("turns", "initial", "contact_native_steps_by_pair")},
                    indent=2,
                ),
                flush=True,
            )
            (args.output / f"{trial.name}.json").write_text(json.dumps(report, indent=2))
        finally:
            env._close_scene()

    if args.oracle_control:
        # Explicit privileged-state control: tests whether the unmodified PEAR
        # controller and benchmark can open the drawer at all. Not a learned policy.
        env = LocalEnv("libero_goal", 0, 0)
        try:
            observation = env._open_scene()
            sim = env.scene.sim
            body = sim.model.body_name2id("wooden_cabinet_1_cabinet_middle")
            ids = [
                i
                for i in range(sim.model.ngeom)
                if sim.model.geom_bodyid[i] == body and np.allclose(sim.model.geom_size[i], [0.00770, 0.00817, 0.04445])
            ]
            if len(ids) != 1:
                raise ValueError("Ambiguous handle")
            bp, br = MotionController._base(env.scene)
            x, y, z = (br.T @ (sim.data.geom_xpos[ids[0]] - bp) * 100).tolist()
            goals = [
                ("retreat_above", x, y + 12, 30, 3.14159, 0, 0, 1),
                ("orient", x, y + 12, 30, 1.5708, 0, 0, 1),
                ("lower_in_front", x, y + 12, z, 1.5708, 0, 0, 1),
                ("approach", x, y, z, 1.5708, 0, 0, 1),
                ("close", x, y, z, 1.5708, 0, 0, -1),
                ("pull", x, y + 16, z, 1.5708, 0, 0, 0),
            ]
            rows = []
            history = []
            for name, *values, grip in goals:
                measured = MotionController._measured_quantities(env.scene, observation)
                action = {
                    "arm": {k: float(v - measured[k]) for k, v in zip(MotionController.SCHEMA["arm"], values)},
                    "gripper": {"width": grip},
                }
                with env._no_render():
                    observation, native, rewards, done, outcome = MotionController(env.scene, observation, action).move(
                        300 - len(history)
                    )
                history.extend(native)
                row = {
                    "phase": name,
                    "action": action,
                    "state": MotionController.state(env.scene, observation),
                    "steps": len(history),
                    "outcome": outcome,
                    "middle_qpos_m": float(sim.data.get_joint_qpos("wooden_cabinet_1_middle_level")),
                    "success": bool(env.scene.check_success()),
                    "rewards": rewards,
                }
                rows.append(row)
                print("ORACLE", json.dumps(row), flush=True)
                from PIL import Image

                rendered = env.scene.env._get_observations(force_update=True)
                for camera in env.CAMERAS:
                    pixels = rendered[env.OBSERVATION_KEY_BY_CAMERA[camera]][::-1, ::-1]
                    Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8)).save(
                        args.output / f"oracle-{name}-{camera}.png"
                    )
                if done or len(history) >= 300:
                    break
            (args.output / "oracle-control.json").write_text(
                json.dumps(
                    {"privileged_not_model_evaluation": True, "phases": rows, "native_actions": history}, indent=2
                )
            )
        finally:
            env._close_scene()


if __name__ == "__main__":
    main()
