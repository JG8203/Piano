#!/usr/bin/env python3
"""Run PianoFit and stream metrics to Weights & Biases.

Usage:
  tools/piano_fit/run_with_wandb.py --wandb-project piano-fit -- \
    ./build-fit/tools/piano_fit/PianoFit_artefacts/Release/PianoFit \
    --manifest build/vintage-upright/manifest.jsonl \
    --subset pilot \
    --max-evals 10000 \
    --output build/vintage-upright/pilot-fit.json \
    --export-dir build/vintage-upright/pilot-audio
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-project", default="piano-fit", help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")
    parser.add_argument("--wandb-mode", default=None, help="Optional W&B mode, e.g. offline")
    parser.add_argument("--metrics", type=Path, default=None, help="Metrics JSONL path")
    parser.add_argument("fit_command", nargs=argparse.REMAINDER, help="PianoFit command after --")
    args = parser.parse_args(argv)

    if args.fit_command and args.fit_command[0] == "--":
        args.fit_command = args.fit_command[1:]
    if not args.fit_command:
        parser.error("Provide the PianoFit command after --")
    return args


def fit_arg_value(command: list[str], name: str) -> str | None:
    for i, arg in enumerate(command):
        if arg == name and i + 1 < len(command):
            return command[i + 1]
    return None


def ensure_metrics_arg(command: list[str], metrics_path: Path) -> list[str]:
    if "--metrics" in command:
        return command
    return command + ["--metrics", str(metrics_path)]


def read_new_metrics(path: Path, offset: int) -> tuple[int, list[dict[str, object]]]:
    if not path.exists():
        return offset, []

    metrics: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return stream.tell(), metrics


def log_metric(wandb_run, metric: dict[str, object]) -> None:
    metric_type = metric.get("type")
    if metric_type == "config":
        wandb_run.config.update({k: v for k, v in metric.items() if k != "type"}, allow_val_change=True)
        return

    step = metric.get("evaluation")
    payload = {k: v for k, v in metric.items() if k != "type"}
    if metric_type:
        payload["event_type"] = metric_type
    wandb_run.log(payload, step=int(step) if isinstance(step, int) else None)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import wandb
    except ImportError:
        print(
            "W&B support is unavailable. Install package groups with: cd python && poetry install --with train,wandb",
            file=sys.stderr,
        )
        return 2

    metrics_path = args.metrics or Path(tempfile.mkdtemp(prefix="piano-fit-wandb-")) / "metrics.jsonl"
    command = ensure_metrics_arg(args.fit_command, metrics_path)
    output_path = fit_arg_value(command, "--output")
    export_dir = fit_arg_value(command, "--export-dir")

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "mode": args.wandb_mode,
        "config": {
            "fit_command": command,
            "metrics_path": str(metrics_path),
            "output_path": output_path,
            "export_dir": export_dir,
        },
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    with wandb.init(**init_kwargs) as run:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        offset = 0
        assert process.stdout is not None
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                print(line, end="")
            offset, metrics = read_new_metrics(metrics_path, offset)
            for metric in metrics:
                log_metric(run, metric)
            time.sleep(0.1)

        for line in process.stdout:
            print(line, end="")

        offset, metrics = read_new_metrics(metrics_path, offset)
        for metric in metrics:
            log_metric(run, metric)

        if output_path and Path(output_path).exists():
            run.save(output_path)
        if export_dir and Path(export_dir).is_dir():
            artifact = wandb.Artifact("piano-fit-audio", type="audio")
            for wav in Path(export_dir).glob("*.wav"):
                artifact.add_file(str(wav))
            run.log_artifact(artifact)

        return process.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
