"""Small sequential LIBERO evaluation through an OpenAI-compatible vision server.

No Ray/CUDA required. On macOS uses GLFW instead of PEAR's Linux Mesa EGL.
This is a diagnostic subset, not the paper's BF16/full-suite evaluation.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


def parse_action(text, schema):
    from pear.agent_loop.protocol import JsonBlockProtocol

    # PEAR's Qwen parser selects the last JSON block after the last </think>.
    # With no closing marker, rpartition returns the whole generated text.
    text = text.rpartition("</think>")[2]
    blocks = list(JsonBlockProtocol.JSON_BLOCK_PATTERN.finditer(text))
    if not blocks:
        raise ValueError("No JSON action after the last thinking boundary")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = value
        return result

    action = json.loads(blocks[-1].group(1), object_pairs_hook=unique)
    JsonBlockProtocol.check_action(action, schema)
    return action


class ResponseFailure(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def decode_response(answer, schema):
    """Adapt split Chat responses to PEAR's simulator protocol, never real hardware.

    Normal content always wins. A reasoning-only EOS is equivalent to PEAR's
    no-</think> case, but only accept a terminal, schema-valid action and log it.
    Never use reasoning to repair an invalid/nonempty final answer or truncation.
    """
    choice = answer["choices"][0]
    finish = choice.get("finish_reason")
    if finish != "stop":
        raise ResponseFailure("model_output_limit" if finish == "length" else "unexpected_finish", str(finish))
    message = choice["message"]
    final = message.get("content")
    if (
        message.get("reasoning_content")
        and message.get("reasoning")
        and message["reasoning_content"] != message["reasoning"]
    ):
        raise ResponseFailure("parse_error", "Conflicting reasoning fields")
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    source = "content"
    if final is None or final == "":
        from pear.agent_loop.protocol import JsonBlockProtocol

        source = "reasoning_only_eos"
        blocks = list(JsonBlockProtocol.JSON_BLOCK_PATTERN.finditer(reasoning))
        usage = answer.get("usage", {})
        reasoning_n = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        if (
            not blocks
            or reasoning[blocks[-1].end() :].strip()
            or "</think>" in reasoning
            or reasoning_n is None
            or reasoning_n != usage.get("completion_tokens")
        ):
            raise ResponseFailure("parse_error", "Empty final answer without a verified reasoning-only terminal action")
        final = blocks[-1].group(0)
        reasoning = reasoning[: blocks[-1].start()].rstrip()
    if not isinstance(final, str) or not isinstance(reasoning, str):
        raise ResponseFailure("parse_error", "Expected string content/reasoning")
    try:
        action = parse_action(final, schema)
    except ValueError as exc:
        raise ResponseFailure("parse_error", str(exc)) from exc
    # Keep reasoning in history, as the original target's template does.
    history_message = {"role": "assistant", "content": final, "reasoning_content": reasoning}
    return action, history_message, source


def summarize(reports, requested, smoke=False):
    counts = dict(Counter(r["status"] for r in reports))
    return {
        "scope": "diagnostic subset; not paper-comparable",
        "requested_trials": requested,
        "completed_trials": len(reports),
        "successes": sum(r["success"] for r in reports),
        "success_rate": None if smoke or not reports else sum(r["success"] for r in reports) / len(reports),
        "status_counts": counts,
        "trials": reports,
    }


CONTROL_GUIDANCE = """
Controller semantics (implementation facts, not object locations):
- Arm actions are increments from the measured pose, in centimeters and radians.
  Orientation uses fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll).
  Near pitch=0 and yaw=0, roll=pi points the tool's z axis down;
  roll=pi/2 points it along -y. A pose's z is not clearance above an obstacle.
- gripper.width is SIGN-ONLY: positive commands fully open, negative fully closed,
  zero holds the previous actuator target. Magnitude is NOT a gap increment or force.
  The observed width is the measured finger gap in centimeters.
- reached means the motion target was reached, not that the object was grasped.
  stopped means motion stopped short. Compare actual motion with requested motion.
  A nearly closed empty gap is not evidence of a grasp; obstruction is not proof of
  grasping the intended object. Do not infer success from issuing a pull command.
- If descent is blocked, do not repeatedly demand more descent at the same position.
  Reinspect both views, retreat to free space, and change approach or orientation.
""".strip()


def measured_feedback(before, result):
    """Use only already-observed robot states and execution outcome, no simulator truth."""
    command = result["requested_action"]["arm"]
    achieved, residual = {}, {}
    for key in command:
        delta = result["state"]["arm"][key] - before["arm"][key]
        error = before["arm"][key] + command[key] - result["state"]["arm"][key]
        if key in ("roll", "pitch", "yaw"):
            delta = math.atan2(math.sin(delta), math.cos(delta))
            error = math.atan2(math.sin(error), math.cos(error))
        achieved[key] = round(delta, 3)
        residual[key] = round(error, 3)
    return {
        "outcome": result["outcome"],
        "native_steps": result["executed"],
        "requested_arm_delta": command,
        "achieved_arm_delta": achieved,
        "remaining_arm_error": residual,
        "measured_gripper_width_cm": result["state"]["gripper"]["width"],
        "warning": "Motion stopped short; do not assume the requested pose or a successful grasp."
        if result["outcome"] == "stopped"
        else None,
    }


def finish_at_native_limit(result, native_steps, limit):
    # A controller can report 'reached' on its last allowed step. Do not query
    # the model again merely to discover that the next rollout has zero budget.
    if not result["done"] and native_steps >= limit:
        return dict(result, done=True, outcome="budget")
    return result


def sampling_parameters(temperature, top_p):
    if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("temperature must be finite and nonnegative; top-p must be in (0, 1]")
    return {"temperature": temperature, "top_p": top_p}


def backend_parameters(server_kind, effort, thinking_budget):
    if server_kind == "vllm":
        if thinking_budget != -1:
            raise ValueError("vLLM adapter requires --thinking-budget -1; no silent ignored budget")
        template = {"preserve_thinking": True, "enable_thinking": effort != "none"}
        if effort != "none":
            template["reasoning_effort"] = effort
        return {"chat_template_kwargs": template, "top_k": -1}
    return {
        "reasoning_effort": effort,
        "thinking_token_budget": thinking_budget,
        "chat_template_kwargs": {"preserve_thinking": True},
    }


def thinking_cap_hit(answer, budget):
    count = answer.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    return budget >= 0 and isinstance(count, int) and count > 0 and count >= budget


def observation_content(state, frames, feedback=None):
    from pear.agent_loop.protocol import JsonBlockProtocol

    text = JsonBlockProtocol.turn_text(
        state, "The current camera frames follow. End your response with one json block containing the next action."
    )
    if feedback is not None:
        text += "\n\nPrevious action measured feedback (not task-success evidence):\n" + json.dumps(feedback)
    content = [{"type": "text", "text": text}]
    for name, png in frames:
        content.extend(
            [
                {"type": "text", "text": f"[{name}]"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            ]
        )
    return content


def validate_resume(settings, current):
    for key in ("model", "suite", "task_id", "effort", "max_tokens", "thinking_budget", "max_turns"):
        if settings.get(key) != current.get(key):
            raise ValueError(f"Resume setting mismatch: {key}")
    for key, default in {
        "temperature": 0.0,
        "top_p": 1.0,
        "control_guidance": False,
        "measured_feedback": False,
        "reject_capped_actions": False,
        "calibrated_overlays": False,
        "server_kind": "splash",
        "episode_response_budget": 0,
        "required_context": 131072,
        "server_image_limit": 64,
    }.items():
        if settings.get(key, default) != current.get(key, default):
            raise ValueError(f"Resume setting mismatch: {key}")


def restore_episode(env, source, folder, messages, initial, include_feedback=False):
    """Replay only persisted executed transitions; never repeat an executed action.

    Only context-limit interruptions are supported, with raw responses and images
    retained. Refuse incomplete/noncontiguous records, altered actions, terminal
    episodes or a state that cannot be recreated from the saved native history.
    """
    previous = json.loads((source / "summary.json").read_text())
    if previous["status"] not in ("context_limit", "token_accounting_error"):
        raise ValueError("Resume supports only context_limit or token_accounting_error interruptions")
    transitions = sorted(source.glob("*-transition.json"))
    if not transitions:
        raise ValueError("No executed transitions to resume")
    history = []
    state = initial["state"]
    restored = 0
    feedback = None
    for turn, path in enumerate(transitions):
        if path.name != f"{turn:03d}-transition.json":
            raise ValueError("Noncontiguous transition history")
        record = json.loads(path.read_text())
        answer = json.loads((source / f"{turn:03d}-response.json").read_text())
        action, assistant, action_source = decode_response(answer, env.SCHEMA)
        if action != record["requested_action"] or record["done"]:
            raise ValueError("Altered action or already-terminal episode")
        frames = [(name, (source / f"{turn:03d}-{name}.png").read_bytes()) for name in env.CAMERAS]
        messages.extend([{"role": "user", "content": observation_content(state, frames, feedback)}, assistant])
        feedback = measured_feedback(state, record) if include_feedback else None
        if include_feedback and record.get("measured_feedback") != feedback:
            raise ValueError("Saved measured feedback differs from reconstructed feedback")
        history.extend(record["actions"])
        state = record["state"]
        restored += int(action_source == "reasoning_only_eos")
        for artifact in source.glob(f"{turn:03d}-*"):
            shutil.copy2(artifact, folder / artifact.name)
    if len(history) != previous["native_steps"]:
        raise ValueError("Native step count differs from interrupted report")
    result = env.rollout(history, None)
    if result["state"] != state:
        raise ValueError("Replayed state differs from saved state")
    # The failed request's observation must also be exactly reproducible.
    from PIL import Image

    for name, image in zip(env.CAMERAS, result["images"]):
        with Image.open(source / f"{len(transitions):03d}-{name}.png") as saved:
            if saved.tobytes() != image.tobytes():
                raise ValueError(f"Replayed {name} frame differs from interrupted request")
    return history, result, len(transitions), restored


def canonical_vllm_messages(messages):
    """vLLM chat normalizes the legacy reasoning alias, but /tokenize does not."""
    result = []
    for message in messages:
        item = dict(message)
        legacy = item.pop("reasoning_content", None)
        if legacy is not None:
            if item.get("reasoning") is not None and item["reasoning"] != legacy:
                raise ValueError("Conflicting historical reasoning fields")
            item["reasoning"] = legacy
        result.append(item)
    return result


def episode_remaining_tokens(initial_prompt, current_prompt, response_budget, context_limit):
    """Include injected observations and preserved reasoning in the episode budget."""
    return min(response_budget - (current_prompt - initial_prompt), context_limit - current_prompt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--libero-root", type=Path, default=Path.home() / ".local/share/pear-sim/LIBERO")
    p.add_argument("--config-dir", type=Path, default=Path.home() / ".local/share/pear-sim/config")
    p.add_argument("--server-kind", choices=["splash", "vllm"], default="splash")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="mit-han-lab/Qwen3.8-27B-PEAR-NVFP4")
    p.add_argument(
        "--suite", default="libero_goal", choices=["libero_goal", "libero_spatial", "libero_object", "libero_10"]
    )
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--trial-start", type=int, default=0)
    p.add_argument("--resume-from", type=Path, help="Previous run; only context-limit interruptions can resume")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--required-context", type=int, default=131072)
    p.add_argument("--server-image-limit", type=int, default=64)
    p.add_argument(
        "--episode-response-budget",
        type=int,
        default=0,
        help="vLLM only: budget for all tokens after the initial prompt, including observations; 0 disables",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=6144,
        help="Splash reasoning cap, reserving tokens for the final action; -1 disables",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 preserves the earlier diagnostic; official evaluation uses 1.0",
    )
    p.add_argument("--top-p", type=float, default=1.0, help="Official evaluation uses 0.95")
    p.add_argument(
        "--control-guidance", action="store_true", help="Add controller facts; an assisted-policy experiment"
    )
    p.add_argument(
        "--measured-feedback", action="store_true", help="Send measured motion/outcome; not the original protocol"
    )
    p.add_argument(
        "--reject-capped-actions", action="store_true", help="Stop instead of executing a forced thinking-cap answer"
    )
    p.add_argument(
        "--calibrated-overlays",
        action="store_true",
        help="Add camera/robot-only metric references and verification guidance; assisted protocol",
    )
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--effort", choices=["none", "low", "medium", "xhigh"], default="low")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true", help="Render only, do not call the model")
    args = p.parse_args()
    try:
        sampling_parameters(args.temperature, args.top_p)
        backend_parameters(args.server_kind, args.effort, args.thinking_budget)
    except ValueError as exc:
        p.error(str(exc))
    if (
        not (0 <= args.trial_start < 50 and 1 <= args.trials <= 50 - args.trial_start)
        or args.max_turns < 1
        or args.max_tokens < 1
    ):
        p.error("trials must be 1..50; budgets must be positive")
    if not 0 <= args.task_id < 10:
        p.error("task-id must be 0..9")
    if args.timeout <= 0 or (args.thinking_budget != -1 and not 0 <= args.thinking_budget <= args.max_tokens - 512):
        p.error("timeout must be positive; thinking budget must leave at least 512 output tokens")
    if args.episode_response_budget and args.server_kind != "vllm":
        p.error("Episode token-budget mode requires vLLM")
    if args.episode_response_budget < 0 or args.required_context < 1 or args.server_image_limit < 1:
        p.error("Invalid context/image/episode budget")
    if args.resume_from:
        validate_resume(json.loads((args.resume_from / "settings.json").read_text()), vars(args))
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.smoke:
        with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=30) as response:
            server_models = json.load(response)
        (args.output / "server-models.json").write_text(json.dumps(server_models, indent=2))
        model = next((m for m in server_models["data"] if m["id"] == args.model), None)
        if model is None or (args.server_kind == "splash" and not model.get("vision")):
            p.error("The selected model must be available with vision enabled")
        if args.max_turns * 2 > model.get("limits", {}).get("images", args.server_image_limit):
            p.error("Two cameras per turn exceed the server image limit; lower --max-turns")
        if model.get("context_length", model.get("max_model_len", 0)) < args.required_context:
            p.error(f"Server must advertise at least {args.required_context} context tokens; no silent truncation")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.config_dir.resolve())
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else os.environ.get("MUJOCO_GL", "egl")
    sys.path.insert(0, str(args.libero_root.resolve()))
    from pear.agent_loop.episode import Episode
    from pear.agent_loop.protocol import JsonBlockProtocol
    from pear.envs.libero.env import LiberoEnv

    class LocalEnv(LiberoEnv):
        def _bind_rendering(self):
            # All scene initialization, replay, cameras, controller and rewards
            # remain PEAR's; only its Linux-specific GL binding is bypassed.
            from libero.libero.envs import OffScreenRenderEnv

            if args.calibrated_overlays:
                from libero_camera_reference import calibrated_scene_class

                return calibrated_scene_class(OffScreenRenderEnv)
            return OffScreenRenderEnv

        def _close_scene(self):
            if self.scene is not None and args.calibrated_overlays:
                self.saved_camera_references = dict(self.scene.camera_references)
            super()._close_scene()

        def rollout(self, history, action):
            result = super().rollout(history, action)
            if args.calibrated_overlays and result["images"] is not None:
                from libero_camera_reference import overlay

                result["raw_images"] = result["images"]
                references = [self.saved_camera_references[name] for name in ("agentview", "robot0_eye_in_hand")]
                result["images"] = [
                    overlay(image, calibration) for image, calibration in zip(result["raw_images"], references)
                ]
                result["camera_references"] = dict(zip(self.CAMERAS, references))
            return result

    template = (Path(__file__).resolve().parents[2] / "pear/envs/system_prompt.md").read_text()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2))
    reports = []
    for trial in range(args.trial_start, args.trial_start + args.trials):
        folder = args.output / f"trial-{trial:02d}"
        folder.mkdir()
        report = {
            "trial_id": trial,
            "task_id": args.task_id,
            "suite": args.suite,
            "success": False,
            "turns": 0,
            "native_steps": 0,
            "status": "running",
            "reasoning_only_actions": 0,
            "thinking_cap_hits": 0,
            "protocol_mode": "assisted"
            if args.control_guidance or args.measured_feedback or args.calibrated_overlays
            else "original",
        }
        stage = "environment"
        (args.output / "progress.json").write_text(json.dumps(report, indent=2))
        history = []
        initial_prompt_tokens = None
        start = time.monotonic()
        env = None
        try:
            env = LocalEnv(args.suite, args.task_id, trial)
            result = env.rollout([], None)
            report["instruction"] = result["instruction"]
            environment_prompt = env.environment_prompt(result)
            if args.control_guidance:
                environment_prompt += "\n\n" + CONTROL_GUIDANCE
            if args.calibrated_overlays:
                from libero_camera_reference import GUIDANCE

                environment_prompt += "\n\n" + GUIDANCE
            feedback = None
            messages = [
                {
                    "role": "system",
                    "content": JsonBlockProtocol.system_prompt(template, environment_prompt, result["instruction"]),
                }
            ]
            first_turn = 0
            if args.resume_from:
                source_folder = args.resume_from / f"trial-{trial:02d}"
                if args.episode_response_budget:
                    original_meta = json.loads((source_folder / "000-request-meta.json").read_text())
                    original_usage = json.loads((source_folder / "000-response.json").read_text())["usage"]
                    initial_prompt_tokens = original_meta["token_budget"]["initial_prompt_tokens"]
                    if original_usage["prompt_tokens"] != initial_prompt_tokens or initial_prompt_tokens > 4096:
                        raise ValueError("Unverified original prompt token count")
                history, result, first_turn, restored = restore_episode(
                    env, source_folder, folder, messages, result, args.measured_feedback
                )
                if args.measured_feedback:
                    last = json.loads((source_folder / f"{first_turn - 1:03d}-transition.json").read_text())
                    feedback = last["measured_feedback"]
                report["thinking_cap_hits"] = sum(
                    thinking_cap_hit(
                        json.loads((source_folder / f"{i:03d}-response.json").read_text()), args.thinking_budget
                    )
                    for i in range(first_turn)
                )
                report.update(
                    resumed_from=str(source_folder.resolve()),
                    resumed_turns=first_turn,
                    native_steps=len(history),
                    reasoning_only_actions=restored,
                    restored_state_and_images_verified=True,
                )
                print(f"Resumed trial={trial} at turn={first_turn + 1} steps={len(history)}", flush=True)
            for turn in range(first_turn, args.max_turns):
                if args.calibrated_overlays:
                    for name, image in zip(env.CAMERAS, result["raw_images"]):
                        image.save(folder / f"{turn:03d}-{name}.raw.png")
                    (folder / f"{turn:03d}-camera-references.json").write_text(
                        json.dumps(result["camera_references"], indent=2)
                    )
                frames = []
                for name, image in zip(env.CAMERAS, result["images"]):
                    image.save(folder / f"{turn:03d}-{name}.png")
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    frames.append((name, buffer.getvalue()))
                content = observation_content(result["state"], frames, feedback if args.measured_feedback else None)
                if args.smoke:
                    report["status"] = "render_only"
                    break
                messages.append({"role": "user", "content": content})
                stage = "inference"
                report["turns"] = turn + 1
                (args.output / "progress.json").write_text(json.dumps(report, indent=2))
                payload = {
                    "model": args.model,
                    "messages": canonical_vllm_messages(messages) if args.server_kind == "vllm" else messages,
                    **sampling_parameters(args.temperature, args.top_p),
                    "max_tokens": args.max_tokens,
                    **backend_parameters(args.server_kind, args.effort, args.thinking_budget),
                }
                token_budget = None
                if args.episode_response_budget:
                    tokenize_payload = {
                        "model": args.model,
                        "messages": payload["messages"],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": payload["chat_template_kwargs"],
                    }
                    tokenize_request = urllib.request.Request(
                        args.base_url.rstrip("/").removesuffix("/v1") + "/tokenize",
                        data=json.dumps(tokenize_payload).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(tokenize_request, timeout=args.timeout) as response:
                        counted = json.load(response)
                    prompt_tokens = counted["count"]
                    if initial_prompt_tokens is None:
                        initial_prompt_tokens = prompt_tokens
                        if initial_prompt_tokens > 4096:
                            raise ResponseFailure("initial_prompt_limit", "Initial prompt exceeds official 4096 budget")
                    remaining = episode_remaining_tokens(
                        initial_prompt_tokens, prompt_tokens, args.episode_response_budget, args.required_context
                    )
                    token_budget = {
                        "initial_prompt_tokens": initial_prompt_tokens,
                        "prompt_tokens": prompt_tokens,
                        "remaining": remaining,
                        "episode_response_budget": args.episode_response_budget,
                    }
                    if remaining <= 0:
                        report.update(status="context_limit", token_budget=token_budget)
                        break
                    payload["max_tokens"] = min(args.max_tokens, remaining)
                (folder / f"{turn:03d}-request-meta.json").write_text(
                    json.dumps(
                        {**{k: v for k, v in payload.items() if k != "messages"}, "token_budget": token_budget},
                        indent=2,
                    )
                )
                request = urllib.request.Request(
                    args.base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    answer = json.load(response)
                (folder / f"{turn:03d}-response.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2))
                stage = "parse"
                if token_budget is not None:
                    if answer.get("usage", {}).get("prompt_tokens") != token_budget["prompt_tokens"]:
                        raise ResponseFailure("token_accounting_error", "Tokenizer and generation prompt counts differ")
                    if answer["choices"][0].get("finish_reason") == "length":
                        raise ResponseFailure("context_limit", "Episode token budget exhausted; action not executed")
                cap_hit = thinking_cap_hit(answer, args.thinking_budget)
                report["thinking_cap_hits"] += int(cap_hit)
                if cap_hit and args.reject_capped_actions:
                    raise ResponseFailure(
                        "thinking_budget_exhausted", "Thinking was forcibly capped; action not executed"
                    )
                action, history_message, source = decode_response(answer, env.SCHEMA)
                report["reasoning_only_actions"] += int(source == "reasoning_only_eos")
                (folder / f"{turn:03d}-parsed.json").write_text(
                    json.dumps({"action": action, "source": source, "history_message": history_message}, indent=2)
                )
                messages.append(history_message)
                stage = "environment"
                before = result["state"]
                result = env.rollout(history, action)
                history.extend(result["actions"])
                result = finish_at_native_limit(result, len(history), env.MAX_STEP_N_BY_SUITE[args.suite])
                record = {k: v for k, v in result.items() if k not in ("images", "raw_images")}
                record["requested_action"] = action
                feedback = measured_feedback(before, record)
                record["measured_feedback"] = feedback
                (folder / f"{turn:03d}-transition.json").write_text(json.dumps(record, indent=2))
                report["native_steps"] = len(history)
                (args.output / "progress.json").write_text(json.dumps(report, indent=2))
                print(f"trial={trial} turn={turn + 1} steps={len(history)} outcome={result['outcome']}", flush=True)
                if result["done"]:
                    report["success"] = Episode.is_success(result)
                    report["status"] = "success" if report["success"] else "environment_budget"
                    break
            else:
                report["status"] = "turn_limit"
        except Exception as exc:
            if isinstance(exc, ResponseFailure):
                report["status"] = exc.status
            elif isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode(errors="replace")
                report["http_error_body"] = detail
                report["status"] = "context_limit" if "context" in detail.lower() else "http_error"
            elif isinstance(exc, (TimeoutError, urllib.error.URLError)):
                report["status"] = "transport_error"
            else:
                report["status"] = stage + "_error"
            report["error"] = f"{type(exc).__name__}: {exc}"
            print(report["error"], flush=True)
        finally:
            if env is not None:
                env._close_scene()
            report["seconds"] = time.monotonic() - start
            (folder / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            (folder / "native-actions.json").write_text(json.dumps(history))
        reports.append(report)
        summary = summarize(reports, args.trials, args.smoke)
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        (args.output / "progress.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
