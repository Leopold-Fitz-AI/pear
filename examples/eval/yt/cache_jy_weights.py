"""Stream/hash verified PEAR weights into a local tmpfs cache.

Uses about 52 GiB of the host's large shared-memory filesystem. The apparent
large /nix disk is read-only; no mount permissions or Nix files are changed.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

SHARED = Path("/vla-cd/pear-jy-deployment")
LOCAL = Path("/dev/shm/pear-jy-deployment/models/Qwen3.8-27B-PEAR")
STATUS = Path("/root/pear-jy/local-weights-status.json")


def main():
    manifest = json.loads((SHARED / "official-model-manifest.json").read_text())
    previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    state = {
        "status": "copying",
        "revision": manifest["sha"],
        "directory": str(LOCAL),
        "verified_hashes": previous.get("verified_hashes", {}),
    }

    def save():
        p = STATUS.with_suffix(".tmp")
        p.write_text(json.dumps(state, indent=2))
        p.replace(STATUS)

    needed = sum(entry["size"] for entry in manifest["siblings"])
    if shutil.disk_usage("/dev/shm").free < needed + 64 * 1024**3:
        raise RuntimeError("Insufficient tmpfs headroom for the model cache")
    if os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") < needed + 128 * 1024**3:
        raise RuntimeError("Insufficient physical RAM headroom for the model cache")
    state.update(cache_kind="tmpfs", expected_bytes=needed)
    LOCAL.mkdir(parents=True, exist_ok=True)
    save()
    try:
        for entry in manifest["siblings"]:
            name = entry["rfilename"]
            target = LOCAL / name
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = entry["lfs"]["sha256"] if entry.get("lfs") else entry["blobId"]
            if (
                previous.get("revision") == manifest["sha"]
                and state["verified_hashes"].get(name) == expected
                and target.exists()
                and target.stat().st_size == entry["size"]
            ):
                continue
            state["file"] = name
            state["file_bytes"] = 0
            save()
            h = hashlib.sha256() if entry.get("lfs") else hashlib.sha1(f"blob {entry['size']}\0".encode())
            partial = target.with_name(target.name + ".partial")
            with (SHARED / "models/Qwen3.8-27B-PEAR" / name).open("rb") as src, partial.open("wb") as dst:
                for chunk in iter(lambda: src.read(16 * 1024 * 1024), b""):
                    dst.write(chunk)
                    h.update(chunk)
                    state["file_bytes"] += len(chunk)
                    if state["file_bytes"] % (256 * 1024 * 1024) == 0:
                        save()
            if partial.stat().st_size != entry["size"] or h.hexdigest() != expected:
                raise ValueError("Size/hash mismatch: " + name)
            partial.replace(target)
            state["verified_hashes"][name] = h.hexdigest()
            save()
        state["status"] = "verified"
    except Exception as exc:
        state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
