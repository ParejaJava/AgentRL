"""Inspect saved LoRA matrices on CPU and persist weight-integrity evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from training.provenance import file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.adapter / "adapter_model.safetensors"
    weights = load_file(path)
    b_weights = {k: v for k, v in weights.items() if ".lora_B." in k}
    report = {
        "adapter_sha256": file_hash(path),
        "tensor_count": len(weights),
        "lora_b_count": len(b_weights),
        "nonzero_lora_b_count": sum(bool(np.any(v != 0)) for v in b_weights.values()),
        "all_finite": all(bool(np.isfinite(v).all()) for v in weights.values()),
        "parameter_count": sum(v.size for v in weights.values()),
        "dtypes": sorted({str(v.dtype) for v in weights.values()}),
    }
    report["passed"] = (
        bool(b_weights)
        and report["all_finite"]
        and report["nonzero_lora_b_count"] == len(b_weights)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Adapter integrity check failed")


if __name__ == "__main__":
    main()
