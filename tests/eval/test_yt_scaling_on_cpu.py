import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "yt_scaling", Path(__file__).parents[2] / "examples/eval/yt/benchmark_scaling.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.parametrize("tokens", [256, 200])
def test_stream_measurement_checks_fixed_length_and_ignores_role_chunk(monkeypatch, tmp_path, tokens):
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "a"}}]},
        {"choices": [{"delta": {"content": "b"}}]},
        {"choices": [], "usage": {"completion_tokens": tokens, "prompt_tokens": 100}},
    ]
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    body += b"data: [DONE]\n\n"
    captured = []

    def open_request(req, timeout):
        captured.append(json.loads(req.data))
        return io.BytesIO(body)

    ticks = iter(range(20))
    monkeypatch.setattr(module.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    payload = {"model": "test", "max_tokens": 256}
    if tokens != 256:
        with pytest.raises(RuntimeError, match="Invalid fixed-token"):
            module.request(payload, tmp_path / "events.json")
    else:
        result = module.request(payload, tmp_path / "events.json")
        assert result["ttft_seconds"] == 2
        assert result["seconds"] == 5
        assert result["approx_decode_tokens_per_second"] == 255
        assert result["end_to_end_tokens_per_second"] == 256 / 5
    assert captured[0]["stream_options"]["include_usage"]
    assert "stream" not in payload
    assert len(json.loads((tmp_path / "events.json").read_text())) == 4
