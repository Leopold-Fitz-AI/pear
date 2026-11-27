import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "splash_batch", Path(__file__).resolve().parents[2] / "examples/eval/eval_libero_splash_batch.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_duplicate_tasks_rejected_before_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module.sys, "argv", ["batch", "--output", str(tmp_path / "run"), "--tasks", "libero_goal:1", "libero_goal:1"]
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("timeout_first", [False, True])
def test_distinct_single_attempts_and_timeout_is_not_failure(tmp_path, monkeypatch, timeout_first, selected):
    root = tmp_path / "batch"
    calls = []
    argv = ["batch", "--output", str(root)]
    if selected:
        argv += ["--tasks", "libero_spatial:2", "libero_goal:1", "--calibrated-overlays"]
    monkeypatch.setattr(module.sys, "argv", argv)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(module.os, "killpg", lambda *args: None)
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.StringIO(json.dumps({"data": [{"id": "mit-han-lab/Qwen3.8-27B-PEAR-NVFP4"}]})),
    )

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            calls.append(command)
            self.stall = timeout_first and len(calls) == 1
            self.code = None
            folder = Path(command[command.index("--output") + 1])
            (folder / "trial-00").mkdir(parents=True)
            if not self.stall:
                (folder / "trial-00/summary.json").write_text(
                    json.dumps({"success": True, "status": "success", "turns": 2, "native_steps": 20})
                )
            else:
                (folder / "progress.json").write_text(json.dumps({"status": "running", "turns": 2}))

        def wait(self, timeout=None):
            if self.stall:
                self.stall = False
                raise module.subprocess.TimeoutExpired("fake-evaluator", timeout)
            self.code = 0
            return self.code

        def poll(self):
            return self.code

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    module.main()
    report = json.loads((root / "batch.json").read_text())
    assert report["status"] == "finished"
    expected = 2 if selected else 5
    assert len(calls) == len(report["attempts"]) == expected
    assert len({(c[c.index("--suite") + 1], c[c.index("--task-id") + 1]) for c in calls}) == expected
    assert all(("--calibrated-overlays" in c) == selected for c in calls)
    assert all(c[c.index("--trials") + 1] == "1" for c in calls)
    assert all("--control-guidance" in c and "--measured-feedback" in c for c in calls)
    assert "No whole-suite success rate" in (root / "REPORT.md").read_text()
    if timeout_first:
        first = report["attempts"][0]
        assert first["status"] == "wall_limit"
        assert "result" not in first
        assert first["last_saved_progress"]["status"] == "running"
        assert (Path(first["directory"]) / "STOPPED.md").exists()
    else:
        assert all(r["result"]["success"] for r in report["attempts"])
