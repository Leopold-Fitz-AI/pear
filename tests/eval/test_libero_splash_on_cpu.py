import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "eval_libero_splash", Path(__file__).resolve().parents[2] / "examples/eval/eval_libero_splash.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SCHEMA = {"arm": ("x",), "gripper": ("width",)}
ACTION = {"arm": {"x": 1}, "gripper": {"width": 0}}
BLOCK = '```json\n{"arm":{"x":1},"gripper":{"width":0}}\n```'


def response(content, reasoning="", finish="stop", tokens=20, reasoning_tokens=20):
    return {
        "choices": [{"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": finish}],
        "usage": {"completion_tokens": tokens, "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
    }


def test_parse_valid_action():
    assert module.parse_action("Move.\n" + BLOCK, SCHEMA) == ACTION


def test_last_block_after_last_boundary_matches_pear():
    assert module.parse_action("<think>```json\n{}\n```</think>" + BLOCK + " suffix", SCHEMA) == ACTION
    assert module.parse_action("```json\n{}\n```" + BLOCK, SCHEMA) == ACTION
    with pytest.raises(ValueError):
        module.parse_action(BLOCK + "</think>No action", SCHEMA)


@pytest.mark.parametrize(
    "text",
    [
        "no action",
        '```json\n{"arm":{"x":true},"gripper":{"width":0}}\n```',
        '```json\n{"arm":{"x":NaN},"gripper":{"width":0}}\n```',
        '```json\n{"arm":{"x":Infinity},"gripper":{"width":0}}\n```',
        '```json\n{"arm":{"x":1,"x":2},"gripper":{"width":0}}\n```',
        '```json\n{"arm":{"x":1}}\n```',
        '```json\n{"arm":{"x":1},"gripper":{"width":0},"extra":{}}\n```',
        '```json\n{"arm":\n```',
    ],
)
def test_reject_invalid_action(text):
    with pytest.raises(ValueError):
        module.parse_action(text, SCHEMA)


def test_normal_answer_wins_and_history_retains_reasoning():
    action, history, source = module.decode_response(response(BLOCK, "Plan"), SCHEMA)
    assert action == ACTION
    assert source == "content"
    assert history == {"role": "assistant", "content": BLOCK, "reasoning_content": "Plan"}


@pytest.mark.parametrize("empty", [None, ""])
def test_reasoning_only_eos_recovered_and_audited(empty):
    action, history, source = module.decode_response(response(empty, "Plan\n" + BLOCK), SCHEMA)
    assert action == ACTION
    assert source == "reasoning_only_eos"
    assert history["reasoning_content"] == "Plan"
    assert history["content"] == BLOCK


@pytest.mark.parametrize(
    "answer",
    [
        response("No action", BLOCK),
        response(" ", BLOCK),
        response(None, BLOCK + " continue thinking"),
        response(None, BLOCK + "</think>"),
        response(None, "no action"),
        response(None, BLOCK, reasoning_tokens=19),
        response(None, BLOCK, reasoning_tokens=None),
        response(None, "```json\n{}\n```"),
    ],
)
def test_never_repair_invalid_final_or_unverified_reasoning(answer):
    with pytest.raises(module.ResponseFailure) as exc:
        module.decode_response(answer, SCHEMA)
    assert exc.value.status == "parse_error"


@pytest.mark.parametrize(
    "finish,status",
    [("length", "model_output_limit"), ("content_filter", "unexpected_finish"), ("tool_calls", "unexpected_finish")],
)
def test_never_execute_incomplete_or_unexpected_finish(finish, status):
    with pytest.raises(module.ResponseFailure) as exc:
        module.decode_response(response(BLOCK, BLOCK, finish=finish), SCHEMA)
    assert exc.value.status == status


def test_summary_keeps_failures_in_denominator_and_marks_partial_runs():
    reports = [
        {"status": "success", "success": True},
        {"status": "parse_error", "success": False},
        {"status": "environment_budget", "success": False},
    ]
    summary = module.summarize(reports, 5)
    assert summary["success_rate"] == 1 / 3
    assert summary["completed_trials"] == 3
    assert summary["requested_trials"] == 5
    assert summary["status_counts"] == {"success": 1, "parse_error": 1, "environment_budget": 1}
    assert module.summarize([], 3)["success_rate"] is None
    assert module.summarize(reports, 3, smoke=True)["success_rate"] is None


@pytest.mark.parametrize("key", ["model", "suite", "task_id", "effort", "max_tokens", "thinking_budget", "max_turns"])
def test_resume_rejects_changed_experiment(key):
    old = dict(
        model="pear", suite="libero_goal", task_id=0, effort="low", max_tokens=8192, thinking_budget=6144, max_turns=30
    )
    module.validate_resume(old, dict(old, timeout=900))
    with pytest.raises(ValueError, match=key):
        module.validate_resume(old, dict(old, **{key: "changed"}))


@pytest.fixture
def resume_rig(tmp_path):
    from PIL import Image

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    image = Image.new("RGB", (8, 8), "red")
    state = {"arm": {"x": 1}, "gripper": {"width": 0}}
    for turn in (0, 1):
        image.save(source / f"{turn:03d}-agentview.png")
    (source / "summary.json").write_text(json.dumps({"status": "context_limit", "native_steps": 1}))
    (source / "000-response.json").write_text(json.dumps(response(BLOCK, "Plan")))
    transition = {"requested_action": ACTION, "done": False, "state": state, "actions": [[1.0]]}
    (source / "000-transition.json").write_text(json.dumps(transition))
    calls = []

    def rollout(history, action):
        calls.append((history.copy(), action))
        return {"state": state, "images": [image]}

    env = SimpleNamespace(SCHEMA=SCHEMA, CAMERAS=("agentview",), rollout=rollout)
    return source, target, env, calls, transition


def test_resume_replays_native_history_not_model_actions(resume_rig):
    source, target, env, calls, _ = resume_rig
    messages = [{"role": "system", "content": "test"}]
    history, result, first_turn, restored = module.restore_episode(env, source, target, messages, {"state": {}})
    assert history == [[1.0]]
    assert calls == [([[1.0]], None)]
    assert first_turn == 1 and restored == 0
    assert messages[-1]["reasoning_content"] == "Plan"
    assert (target / "000-response.json").exists()


@pytest.mark.parametrize("change", ["terminal", "gap", "action", "count", "state", "image"])
def test_resume_refuses_invalid_records_or_replay(resume_rig, change):
    from PIL import Image

    source, target, env, _, record = resume_rig
    if change == "terminal":
        record["done"] = True
    elif change == "action":
        record["requested_action"] = {}
    elif change == "count":
        (source / "summary.json").write_text(json.dumps({"status": "context_limit", "native_steps": 2}))
    elif change == "state":
        env.rollout = lambda h, a: {"state": {}, "images": []}
    elif change == "image":
        Image.new("RGB", (8, 8), "blue").save(source / "001-agentview.png")
    (source / "000-transition.json").write_text(json.dumps(record))
    if change == "gap":
        (source / "000-transition.json").rename(source / "001-transition.json")
    with pytest.raises(ValueError):
        module.restore_episode(env, source, target, [], {"state": {}})


@pytest.mark.parametrize("tampered", [False, True])
def test_resume_checks_assisted_feedback(resume_rig, tampered):
    source, target, env, _, record = resume_rig
    before = {"arm": {"x": 0}, "gripper": {"width": 0}}
    record.update(executed=1, outcome="reached")
    record["measured_feedback"] = module.measured_feedback(before, record)
    if tampered:
        record["measured_feedback"]["achieved_arm_delta"]["x"] = 99
    (source / "000-transition.json").write_text(json.dumps(record))
    if tampered:
        with pytest.raises(ValueError, match="measured feedback"):
            module.restore_episode(env, source, target, [], {"state": before}, True)
    else:
        history, _, _, _ = module.restore_episode(env, source, target, [], {"state": before}, True)
        assert history == [[1.0]]


def test_feedback_exposes_blocked_descent_without_privileged_state():
    before = {"arm": {"x": 1, "y": 2, "z": 26, "roll": 3.14}, "gripper": {"width": 8}}
    result = {
        "state": {"arm": {"x": 1, "y": 2, "z": 22.6, "roll": -3.14}, "gripper": {"width": 0.5}},
        "requested_action": {"arm": {"x": 0, "y": 0, "z": -14, "roll": 0}},
        "outcome": "stopped",
        "executed": 7,
        "secret_object_position": [99, 99, 99],
    }
    feedback = module.measured_feedback(before, result)
    assert feedback["achieved_arm_delta"]["z"] == -3.4
    assert feedback["remaining_arm_error"]["z"] == -10.6
    assert abs(feedback["achieved_arm_delta"]["roll"]) < 0.01
    assert feedback["warning"]
    assert "secret_object_position" not in json.dumps(feedback)
    assert module.observation_content(before, [], feedback)[0]["text"].find("Previous action measured feedback") > 0
    assert "Previous action" not in module.observation_content(before, [])[0]["text"]


@pytest.mark.parametrize(
    "temperature,top_p", [(float("nan"), 0.95), (float("inf"), 0.95), (-1, 0.95), (1, 0), (1, 1.01), (1, float("nan"))]
)
def test_bad_sampling_parameters_rejected(temperature, top_p):
    with pytest.raises(ValueError):
        module.sampling_parameters(temperature, top_p)


def test_native_budget_stops_without_zero_step_model_turn_and_preserves_success():
    original = {"done": False, "outcome": "reached", "rewards": [0.0]}
    terminal = module.finish_at_native_limit(original, 300, 300)
    assert terminal["done"] and terminal["outcome"] == "budget"
    assert original["done"] is False
    assert module.finish_at_native_limit(original, 299, 300) is original
    success = {"done": True, "outcome": "done", "rewards": [1.0]}
    assert module.finish_at_native_limit(success, 300, 300) is success


def test_vllm_reasoning_alias_preserves_history_and_rejects_conflicts():
    answer = response(BLOCK, "plan")
    message = answer["choices"][0]["message"]
    message["reasoning"] = message.pop("reasoning_content")
    _, history, _ = module.decode_response(answer, SCHEMA)
    assert history["reasoning_content"] == "plan"
    message["reasoning_content"] = "different plan"
    with pytest.raises(module.ResponseFailure, match="Conflicting"):
        module.decode_response(answer, SCHEMA)


def test_tokenize_and_chat_use_same_canonical_reasoning_field():
    history = [{"role": "assistant", "content": "action", "reasoning_content": "retained thought"}]
    converted = module.canonical_vllm_messages(history)
    assert converted == [{"role": "assistant", "content": "action", "reasoning": "retained thought"}]
    assert history[0]["reasoning_content"] == "retained thought"
    with pytest.raises(ValueError, match="Conflicting"):
        module.canonical_vllm_messages([{"reasoning_content": "a", "reasoning": "b"}])


def test_episode_budget_includes_observations_and_history_without_per_turn_cap():
    assert module.episode_remaining_tokens(1000, 1000, 256000, 260096) == 256000
    assert module.episode_remaining_tokens(1000, 4000, 256000, 260096) == 253000
    assert module.episode_remaining_tokens(1000, 257000, 256000, 260096) == 0
    assert module.episode_remaining_tokens(1000, 260100, 256000, 260096) < 0
    assert module.episode_remaining_tokens(1000, 4000, 256000, 5000) == 1000


def test_backend_options_do_not_silently_ignore_effort_or_budget():
    options = module.backend_parameters("vllm", "low", -1)
    assert options["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert options["chat_template_kwargs"]["preserve_thinking"] is True
    assert options["top_k"] == -1
    assert "thinking_token_budget" not in options
    assert not module.backend_parameters("vllm", "none", -1)["chat_template_kwargs"]["enable_thinking"]
    with pytest.raises(ValueError, match="no silent ignored budget"):
        module.backend_parameters("vllm", "low", 6144)
    assert module.backend_parameters("splash", "low", 6144)["thinking_token_budget"] == 6144


def test_official_sampling_is_configurable_and_cap_detection_is_explicit():
    assert module.sampling_parameters(1.0, 0.95) == {"temperature": 1.0, "top_p": 0.95}
    assert module.thinking_cap_hit(response(BLOCK, reasoning_tokens=6145), 6144)
    assert not module.thinking_cap_hit(response(BLOCK, reasoning_tokens=6143), 6144)
    assert not module.thinking_cap_hit(response(BLOCK, reasoning_tokens=9999), -1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperature", 1),
        ("top_p", 0.95),
        ("control_guidance", True),
        ("measured_feedback", True),
        ("reject_capped_actions", True),
        ("calibrated_overlays", True),
        ("server_kind", "vllm"),
    ],
)
def test_resume_never_silently_changes_policy_protocol(key, value):
    settings = dict(
        model="pear", suite="libero_goal", task_id=0, effort="low", max_tokens=8192, thinking_budget=6144, max_turns=30
    )
    with pytest.raises(ValueError, match=key):
        module.validate_resume(settings, dict(settings, **{key: value}))
