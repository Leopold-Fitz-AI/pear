"""Run on yt: publish the already-tested CPU LIBERO environment and PEAR source."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("/root/pear-yt")
    shared = Path("/vla-cd/pear-jy-deployment")
    archive = shared / "jy-sim-bundle.tar"
    status = {"status": "packaging", "source": "yt pinned, tested CPU simulation environment"}
    path = shared / "jy-sim-transfer.json"
    path.write_text(json.dumps(status))
    try:
        partial = archive.with_suffix(".partial")
        subprocess.run(
            [
                "tar",
                "--exclude=__pycache__",
                "--exclude=.git",
                "-cf",
                str(partial),
                "-C",
                str(source),
                "envs/sim/lib/python3.10/site-packages",
                "pear",
                "LIBERO",
                "libero-file-manifest.json",
                "LIBERO_COMMIT.txt",
                "sim-requirements.txt",
            ],
            check=True,
        )
        h = hashlib.sha256()
        with partial.open("rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        partial.replace(archive)
        status.update(status="verified", sha256=h.hexdigest(), bytes=archive.stat().st_size)
    except Exception as exc:
        status.update(status="error", error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
