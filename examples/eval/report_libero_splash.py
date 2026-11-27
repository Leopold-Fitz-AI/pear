"""Build an honest local report (and optional observation videos) from saved trials."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def load_summary(run):
    try:
        return json.loads((run / "summary.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def wait_for_run(run, pid, timeout):
    deadline = time.monotonic() + timeout
    while True:
        summary = load_summary(run)
        if summary and summary["completed_trials"] == summary["requested_trials"]:
            return summary
        if time.monotonic() >= deadline:
            raise TimeoutError("Report watcher timed out; evaluation was not stopped")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # The process may have completed between the first read and this check.
            summary = load_summary(run)
            if summary and summary["completed_trials"] == summary["requested_trials"]:
                return summary
            raise RuntimeError("Evaluation exited before all requested trials were recorded") from None
        time.sleep(15)


def render_video(trial):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    frames = sorted(trial.glob("*-agentview.png"))
    if not frames:
        return
    with imageio.get_writer(trial / "observation-slideshow.mp4", fps=1, codec="libx264") as writer:
        for path in frames:
            with Image.open(path) as left, Image.open(path.with_name(path.name.replace("agentview", "wrist"))) as right:
                frame = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + 32), "white")
                frame.paste(left, (0, 32))
                frame.paste(right, (left.width, 32))
            ImageDraw.Draw(frame).text(
                (10, 8),
                f"{trial.name} observation {path.name[:3]} | agentview / wrist | NOT real-time video",
                fill="black",
            )
            writer.append_data(np.asarray(frame))


def report_text(summary, settings, context, baseline=None):
    complete = summary["completed_trials"] == summary["requested_trials"]
    lines = [
        "# LIBERO / Splash evaluation report",
        "",
        f"Status: {'completed' if complete else 'INCOMPLETE'}; "
        f"{summary['completed_trials']}/{summary['requested_trials']} recorded trials.",
        f"Model: `{settings['model']}`; effort: `{settings['effort']}`; context: {context} tokens.",
        f"Suite/task: {settings['suite']}/{settings['task_id']}; output cap {settings['max_tokens']}; "
        f"thinking cap {settings['thinking_budget']}; turn cap {settings['max_turns']}.",
        "",
        "This is a selected-task diagnostic, not the paper's full BF16 benchmark.",
        f"Sampling: temperature={settings.get('temperature', 0.0)}, top_p={settings.get('top_p', 1.0)}.",
        (
            "ASSISTED protocol: added controller guidance and/or measured action feedback; not an unmodified-policy result."
            if settings.get("control_guidance")
            or settings.get("measured_feedback")
            or settings.get("calibrated_overlays")
            else "Original observation protocol."
        ),
        "",
        "| Trial | Success | Status | Turns | Native steps | Seconds | Thinking-cap hits |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for trial in summary["trials"]:
        lines.append(
            f"| {trial['trial_id']} | {trial['success']} | {trial['status']} | "
            f"{trial['turns']} | {trial['native_steps']} | {trial['seconds']:.1f} | {trial.get('thinking_cap_hits', 'unknown')} |"
        )
    lines += ["", f"Successes: {summary['successes']}/{summary['completed_trials']} recorded attempts."]
    if baseline:
        aggregate = baseline["aggregate"]
        lines += [
            "",
            "## Earlier low-effort run (descriptive comparison only)",
            f"Low effort: {aggregate['successes']}/{aggregate['unique_trials']} unique trials succeeded.",
            "The low run began with 64K context; one trajectory was continued with 128K after a recorded context error.",
            "This run starts fresh with 128K throughout. It is not a perfectly controlled effort-only comparison.",
            "Three initial states of one task are not evidence of a whole-suite success rate.",
        ]
    lines += [
        "",
        "Raw outputs, parsed actions, transitions, settings and server-models.json remain beside this report.",
        "observation-slideshow.mp4 files, if requested, show sampled observations, not continuous real-time motion.",
        "Errors and failed trials are retained in the denominator; no success is inferred from the model's own claim.",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--compare", type=Path, help="Earlier comparison.json with a deduplicated aggregate")
    p.add_argument("--watch-pid", type=int, help="Wait until this running evaluation records all its trials")
    p.add_argument("--watch-timeout", type=float, default=14400)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    try:
        summary = (
            wait_for_run(args.run, args.watch_pid, args.watch_timeout) if args.watch_pid else load_summary(args.run)
        )
        if summary is None:
            raise ValueError("No readable summary yet")
        settings = json.loads((args.run / "settings.json").read_text())
        models = json.loads((args.run / "server-models.json").read_text())
        context = next(m["context_length"] for m in models["data"] if m["id"] == settings["model"])
        baseline = json.loads(args.compare.read_text()) if args.compare else None
        if args.video:
            for trial in summary["trials"]:
                render_video(args.run / f"trial-{trial['trial_id']:02d}")
        (args.run / "REPORT.md").write_text(report_text(summary, settings, context, baseline))
        (args.run / "comparison.json").write_text(
            json.dumps(
                {
                    "current": summary,
                    "settings": settings,
                    "context_length": context,
                    "baseline_path": str(args.compare) if args.compare else None,
                    "baseline": baseline,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"Report written: {args.run / 'REPORT.md'}", flush=True)
    except Exception as exc:
        (args.run / "report-error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
