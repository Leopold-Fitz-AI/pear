import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "report_libero_splash", Path(__file__).resolve().parents[2] / "examples/eval/report_libero_splash.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("completed,label", [(1, "INCOMPLETE"), (3, "completed")])
def test_report_does_not_hide_partial_runs(completed, label):
    summary = {"completed_trials": completed, "requested_trials": 3, "successes": 0, "trials": []}
    settings = {
        "model": "pear",
        "effort": "xhigh",
        "suite": "libero_goal",
        "task_id": 0,
        "max_tokens": 8192,
        "thinking_budget": 6144,
        "max_turns": 30,
    }
    baseline = {"aggregate": {"successes": 0, "unique_trials": 3}}
    text = module.report_text(summary, settings, 131072, baseline)
    assert f"Status: {label}" in text
    assert "not a perfectly controlled effort-only comparison" in text
    assert "0/3 unique trials" in text
    assisted = module.report_text(summary, dict(settings, measured_feedback=True, temperature=1, top_p=0.95), 131072)
    assert "ASSISTED protocol" in assisted
    assert "temperature=1, top_p=0.95" in assisted
    assert "Thinking-cap hits" in assisted


def test_completed_summary_does_not_need_live_process(tmp_path):
    summary = {"completed_trials": 3, "requested_trials": 3}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert module.wait_for_run(tmp_path, 123, 0) == summary


def test_watcher_timeout_does_not_fabricate_results(tmp_path):
    with pytest.raises(TimeoutError):
        module.wait_for_run(tmp_path, 123, 0)
    assert not (tmp_path / "summary.json").exists()


def test_watcher_detects_early_exit(tmp_path, monkeypatch):
    def missing(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(module.os, "kill", missing)
    with pytest.raises(RuntimeError, match="before all requested trials"):
        module.wait_for_run(tmp_path, 123, 10)
