"""Run on yt: copy only PEAR's public weights/runtime into the shared jy workspace.

Does not start a model or copy home-directory credentials/configuration.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

SOURCE = Path("/root/pear-yt")
DEST = Path("/vla-cd/pear-jy-deployment")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    previous = {}
    if (DEST / "transfer-status.json").exists():
        previous = json.loads((DEST / "transfer-status.json").read_text())
    state = {
        "status": "copying",
        "pid": os.getpid(),
        "source_host": "yt",
        "verified_hashes": previous.get("verified_hashes", {}),
        "resumed_verified_files": list(previous.get("verified_hashes", {})),
    }

    def save():
        # This shared object-backed mount intermittently rejects rename-overwrite
        # while another host reads. Keep an atomic authoritative local checkpoint;
        # shared readers already tolerate incomplete JSON during direct writes.
        text = json.dumps(state, indent=2)
        p = SOURCE / "jy-transfer-status.tmp"
        p.write_text(text)
        p.replace(SOURCE / "jy-transfer-status.json")
        (DEST / "transfer-status.json").write_text(text)

    save()
    try:
        manifest = json.loads((SOURCE / "official-model-manifest.json").read_text())
        assert json.loads((SOURCE / "download-status.json").read_text())["status"] == "verified"
        shutil.copy2(SOURCE / "official-model-manifest.json", DEST / "official-model-manifest.json")
        folder = DEST / "models/Qwen3.8-27B-PEAR"
        folder.mkdir(exist_ok=True)
        for entry in manifest["siblings"]:
            name = entry["rfilename"]
            state["copying_file"] = name
            save()
            src = SOURCE / "models/Qwen3.8-27B-PEAR" / name
            target = folder / name
            expected = entry["lfs"]["sha256"] if entry.get("lfs") else entry["blobId"]
            if (
                state["verified_hashes"].get(name) == expected
                and target.exists()
                and target.stat().st_size == entry.get("size", target.stat().st_size)
            ):
                continue  # A previously copied and SHA-verified immutable artifact.
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            shutil.copyfile(src, partial)
            if entry.get("size") is not None:
                assert partial.stat().st_size == entry["size"], name
            if entry.get("lfs"):
                actual = digest(partial)
                assert actual == entry["lfs"]["sha256"], name
            else:
                data = partial.read_bytes()
                actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                assert actual == entry["blobId"], name
            partial.replace(target)
            state["verified_hashes"][name] = actual
            save()
        state.update(status="packaging_runtime", revision=manifest["sha"])
        save()
        runtime = Path("/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu")
        bundles = {
            "python-runtime.tar": (runtime.parent, [runtime.name]),
            "server-packages.tar": (SOURCE / "envs/server", ["lib/python3.12/site-packages"]),
            "cuda-compat.tar": (SOURCE, ["cuda-compat-13"]),
        }
        state["bundles"] = {}
        for name, (cwd, files) in bundles.items():
            partial = DEST / (name + ".partial")
            subprocess.run(["tar", "-cf", str(partial), "-C", str(cwd), *files], check=True)
            target = DEST / name
            partial.replace(target)
            state["bundles"][name] = {"sha256": digest(target), "bytes": target.stat().st_size}
            save()
        shutil.copy2(SOURCE / "server-requirements.txt", DEST / "server-requirements-source.txt")
        state["status"] = "verified"
    except Exception as exc:
        state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
