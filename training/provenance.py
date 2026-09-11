"""Hash actual model and adapter files without relying on mutable model names."""

import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_manifest(root: str | Path) -> dict:
    root = Path(root)
    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix in {".json", ".safetensors", ".jinja"}
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"No weights under {root}")
    result = {"path": str(root), "files": {p.name: file_hash(p) for p in files}}
    metadata = root / ".cache/huggingface/download/model.safetensors.metadata"
    if metadata.exists():
        lines = metadata.read_text(encoding="utf-8").splitlines()
        result["download_revision"] = lines[0]
        result["download_etag"] = lines[1]
        result["weight_hash_matches_download_etag"] = (
            result["files"].get("model.safetensors") == lines[1]
        )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = model_manifest(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
