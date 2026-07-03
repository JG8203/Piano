"""Shared configuration helpers for piano-fit commands."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetConfig:
    manifest: Path


@dataclass(frozen=True)
class BuildConfig:
    piano_fit: Path


@dataclass(frozen=True)
class TrainConfig:
    subset: str
    population: int
    sigma: float
    max_evals: int
    max_seconds: float
    seed: int
    device: str
    work_dir: Path | None


@dataclass(frozen=True)
class OutputConfig:
    metrics: Path | None
    result: Path
    export_dir: Path | None


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    run_name: str | None
    mode: str | None


@dataclass(frozen=True)
class FitConfig:
    dataset: DatasetConfig
    build: BuildConfig
    train: TrainConfig
    outputs: OutputConfig
    wandb: WandbConfig
    source_path: Path


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required section: {name}")
    return value


def _required(section: dict[str, Any], section_name: str, field: str) -> Any:
    if field not in section:
        raise ValueError(f"Missing required field: {section_name}.{field}")
    return section[field]


def _path(value: Any, base_dir: Path, field: str) -> Path:
    if value is None:
        raise ValueError(f"Missing required field: {field}")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _optional_path(value: Any, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _string_or_none(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def load_config(path: str | Path) -> FitConfig:
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    source_path = source_path.resolve()
    base_dir = source_path.parent

    loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping")

    dataset = _section(loaded, "dataset")
    build = _section(loaded, "build")
    train = _section(loaded, "train")
    outputs = _section(loaded, "outputs")
    wandb = _section(loaded, "wandb")

    return FitConfig(
        dataset=DatasetConfig(
            manifest=_path(_required(dataset, "dataset", "manifest"), base_dir, "dataset.manifest")
        ),
        build=BuildConfig(
            piano_fit=_path(_required(build, "build", "piano_fit"), base_dir, "build.piano_fit")
        ),
        train=TrainConfig(
            subset=str(_required(train, "train", "subset")),
            population=int(_required(train, "train", "population")),
            sigma=float(_required(train, "train", "sigma")),
            max_evals=int(_required(train, "train", "max_evals")),
            max_seconds=float(_required(train, "train", "max_seconds")),
            seed=int(_required(train, "train", "seed")),
            device=str(_required(train, "train", "device")),
            work_dir=_optional_path(train.get("work_dir"), base_dir),
        ),
        outputs=OutputConfig(
            metrics=_optional_path(outputs.get("metrics"), base_dir),
            result=_path(_required(outputs, "outputs", "result"), base_dir, "outputs.result"),
            export_dir=_optional_path(outputs.get("export_dir"), base_dir),
        ),
        wandb=WandbConfig(
            enabled=bool(_required(wandb, "wandb", "enabled")),
            project=str(_required(wandb, "wandb", "project")),
            entity=_string_or_none(wandb.get("entity"), "wandb.entity"),
            run_name=_string_or_none(wandb.get("run_name"), "wandb.run_name"),
            mode=_string_or_none(wandb.get("mode"), "wandb.mode"),
        ),
        source_path=source_path,
    )


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load and print a piano-fit YAML profile.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(config), default=_json_default, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
