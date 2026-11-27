"""Download the user-selected BF16 PEAR snapshot; verify official LFS hashes."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    args = p.parse_args()
    manifest = json.loads((args.root / "official-model-manifest.json").read_text())
    assert manifest["id"] == "mit-han-lab/Qwen3.8-27B-PEAR"
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    from huggingface_hub import snapshot_download

    status = {
        "model": manifest["id"],
        "revision": manifest["sha"],
        "endpoint": args.endpoint,
        "status": "downloading",
        "pid": os.getpid(),
    }
    path = args.root / "download-status.json"
    path.write_text(json.dumps(status, indent=2))
    try:
        folder = Path(
            snapshot_download(
                repo_id=manifest["id"],
                revision=manifest["sha"],
                local_dir=str(args.root / "models/Qwen3.8-27B-PEAR"),
                max_workers=2,
                token=False,
            )
        )
        status["status"] = "verifying"
        path.write_text(json.dumps(status, indent=2))
        verified = {}
        for entry in manifest["siblings"]:
            file = folder / entry["rfilename"]
            if entry.get("size") is not None and file.stat().st_size != entry["size"]:
                raise ValueError(f"Size mismatch: {file.name}")
            if entry.get("lfs"):
                h = hashlib.sha256()
                with file.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                        h.update(chunk)
                if h.hexdigest() != entry["lfs"]["sha256"]:
                    raise ValueError(f"SHA-256 mismatch: {file.name}")
                verified[file.name] = h.hexdigest()
            else:
                # HF blobId is Git's SHA-1(blob <size> NUL <content>).
                data = file.read_bytes()
                digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if digest != entry["blobId"]:
                    raise ValueError(f"Git blob hash mismatch: {file.name}")
                verified[file.name] = digest
        status.update(status="verified", directory=str(folder), verified_hashes=verified)
    except Exception as exc:
        status.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        path.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
