"""Finish the local V2 comparison sequentially, with no competing GPU jobs.

Run in the application environment (psutil). Only waits for the explicitly
identified training process and stops server processes created by this runner.
"""

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

from training.data import digest, read_jsonl
from training.provenance import file_hash

ROOT = Path("eval/reports/executor-v2")
TRAIN = Path("training/outputs/executor-v2-1.7b-qlora")
PYTHON = str(Path("training/.venv/Scripts/python.exe").resolve())


def status(phase: str, **extra) -> None:
    value = {"phase": phase, "time": time.time(), **extra}
    (ROOT / "comparison_status.json").write_text(
        json.dumps(value, indent=2), encoding="utf-8"
    )
    print(json.dumps(value), flush=True)


def run(name: str, command: list[str]) -> None:
    status(name)
    with (ROOT / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed: see {ROOT / (name + '.log')}")


def stop_owned_server(server: subprocess.Popen) -> None:
    try:
        owned = psutil.Process(server.pid)
        processes = [*reversed(owned.children(recursive=True)), owned]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        process.kill()
    server.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-training-pid", type=int)
    args = parser.parse_args()
    if (ROOT / "comparison_complete.json").exists():
        raise ValueError("Comparison already completed; preserve evidence")
    if args.wait_training_pid:
        try:
            process = psutil.Process(args.wait_training_pid)
            command = process.cmdline()
            if (
                "training.train_sft" not in command
                or "training/outputs/executor-v2-1.7b-qlora" not in command
            ):
                raise ValueError("PID is not the authorized V2 training process")
            status("waiting_for_training", training_pid=process.pid)
            while process.is_running():
                try:
                    process.wait(timeout=30)
                except psutil.TimeoutExpired:
                    continue
        except psutil.NoSuchProcess:
            pass
    metrics = json.loads((TRAIN / "metrics.json").read_text(encoding="utf-8"))
    if [e["epoch"] for e in metrics["epochs"]] != [0, 1, 2]:
        raise ValueError("Training is not complete; cannot run held-out comparison")
    manifest = json.loads(
        Path("eval/executor_v2/manifest.json").read_text(encoding="utf-8")
    )
    for split in ("train", "dev", "test"):
        assert (
            digest(read_jsonl(f"eval/executor_v2/{split}.jsonl"))
            == manifest[f"{split}_sha256"]
        )
    assert (
        digest(read_jsonl("eval/executor_v2/complete_tasks.jsonl"))
        == manifest["complete_tasks_sha256"]
    )
    sources = {
        str(path): file_hash(path)
        for directory in ("app", "training")
        for path in Path(directory).rglob("*.py")
        if ".venv" not in path.parts and "outputs" not in path.parts
    }
    (ROOT / "comparison_source_hashes.json").write_text(
        json.dumps(sources, indent=2), encoding="utf-8"
    )
    adapter = str(TRAIN / "adapter")
    run(
        "adapter_integrity",
        [
            PYTHON,
            "-m",
            "training.check_adapter",
            adapter,
            "--output",
            str(ROOT / "adapter_integrity.json"),
        ],
    )
    for name, extra in (("base", []), ("adapter", ["--adapter", adapter])):
        run(
            f"decisions_{name}",
            [
                PYTHON,
                "-u",
                "-m",
                "training.evaluate_executor_v2",
                "--model",
                "data/models/Qwen3-1.7B",
                *extra,
                "--output",
                str(ROOT / f"decisions-{name}"),
            ],
        )
    for name, extra in (("base", []), ("adapter", ["--adapter", adapter])):
        # Never attach to or stop a pre-existing service on this port.
        with socket.socket() as check:
            check.bind(("127.0.0.1", 8010))
        model_name = f"globex-v2-1.7b-{name}"
        with (ROOT / f"server-{name}.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [
                    PYTHON,
                    "-u",
                    "-m",
                    "training.serve_executor",
                    "--model",
                    "data/models/Qwen3-1.7B",
                    *extra,
                    "--name",
                    model_name,
                    "--context-window",
                    "8192",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                status(f"starting_server_{name}", server_pid=server.pid)
                deadline = time.monotonic() + 180
                while True:
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Local server did not become ready")
                    try:
                        with urllib.request.urlopen(
                            "http://127.0.0.1:8010/health", timeout=2
                        ) as response:
                            health = json.load(response)
                        if health.get("ready") and health.get("model") == model_name:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(1)
                run(
                    f"agent_{name}",
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "training.evaluate_agent_v2",
                        "--model",
                        model_name,
                        "--output",
                        str(ROOT / f"agent-{name}"),
                    ],
                )
            finally:
                stop_owned_server(server)
    summary = {
        name: json.loads((ROOT / name / "summary.json").read_text(encoding="utf-8"))
        for name in (
            "decisions-base",
            "decisions-adapter",
            "agent-base",
            "agent-adapter",
        )
    }
    (ROOT / "comparison_complete.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    status("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
