"""Config-driven training command for Piano fitting."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from piano_fit.config import (
    FitConfig,
    load_config,
)
from piano_fit import train_evotorch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML training profile")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without training")
    parser.add_argument("--piano-fit", type=Path, default=None, help="Override PianoFit executable path")
    parser.add_argument("--manifest", type=Path, default=None, help="Override prepared manifest path")
    parser.add_argument("--subset", default=None, choices=("pilot", "all"), help="Override dataset subset")
    parser.add_argument("--population", type=int, default=None, help="Override EvoTorch population size")
    parser.add_argument("--sigma", type=float, default=None, help="Override CMA-ES sigma")
    parser.add_argument("--max-evals", type=int, default=None, help="Override evaluation budget")
    parser.add_argument("--max-seconds", type=float, default=None, help="Override target/render crop duration")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", default=None, help="Override EvoTorch torch device")
    parser.add_argument("--evaluator-mode", default=None, choices=("stdio", "files"), help="Override evaluator mode")
    parser.add_argument("--work-dir", type=Path, default=None, help="Override temporary genome batch directory")
    parser.add_argument("--metrics", type=Path, default=None, help="Override metrics JSONL path")
    parser.add_argument("--output", type=Path, default=None, help="Override final result JSON path")
    parser.add_argument("--export-dir", type=Path, default=None, help="Override final WAV export directory")
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true", default=None, help="Enable W&B logging")
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false", help="Disable W&B logging")
    parser.add_argument("--wandb-project", default=None, help="Override W&B project")
    parser.add_argument("--wandb-entity", default=None, help="Override W&B entity")
    parser.add_argument("--wandb-run-name", default=None, help="Override W&B run name")
    parser.add_argument("--wandb-mode", default=None, help="Override W&B mode")
    return parser.parse_args(argv)


def _override_path(path: Path | None) -> Path | None:
    return path.resolve() if path is not None else None


def apply_overrides(config: FitConfig, args: argparse.Namespace) -> FitConfig:
    dataset = replace(
        config.dataset,
        manifest=_override_path(args.manifest) or config.dataset.manifest,
    )
    build = replace(
        config.build,
        piano_fit=_override_path(args.piano_fit) or config.build.piano_fit,
    )
    train = replace(
        config.train,
        subset=args.subset if args.subset is not None else config.train.subset,
        population=args.population if args.population is not None else config.train.population,
        sigma=args.sigma if args.sigma is not None else config.train.sigma,
        max_evals=args.max_evals if args.max_evals is not None else config.train.max_evals,
        max_seconds=args.max_seconds if args.max_seconds is not None else config.train.max_seconds,
        seed=args.seed if args.seed is not None else config.train.seed,
        device=args.device if args.device is not None else config.train.device,
        evaluator_mode=args.evaluator_mode if args.evaluator_mode is not None else config.train.evaluator_mode,
        work_dir=_override_path(args.work_dir) if args.work_dir is not None else config.train.work_dir,
    )
    outputs = replace(
        config.outputs,
        metrics=_override_path(args.metrics) if args.metrics is not None else config.outputs.metrics,
        result=_override_path(args.output) or config.outputs.result,
        export_dir=_override_path(args.export_dir) if args.export_dir is not None else config.outputs.export_dir,
    )
    wandb = replace(
        config.wandb,
        enabled=args.wandb_enabled if args.wandb_enabled is not None else config.wandb.enabled,
        project=args.wandb_project if args.wandb_project is not None else config.wandb.project,
        entity=args.wandb_entity if args.wandb_entity is not None else config.wandb.entity,
        run_name=args.wandb_run_name if args.wandb_run_name is not None else config.wandb.run_name,
        mode=args.wandb_mode if args.wandb_mode is not None else config.wandb.mode,
    )
    return FitConfig(
        dataset=dataset,
        build=build,
        train=train,
        outputs=outputs,
        wandb=wandb,
        source_path=config.source_path,
    )


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def config_summary(config: FitConfig) -> dict[str, Any]:
    return asdict(config)


def serializable_config_summary(config: FitConfig) -> dict[str, Any]:
    return json.loads(json.dumps(config_summary(config), default=_json_default))


def training_args(config: FitConfig) -> argparse.Namespace:
    return argparse.Namespace(
        piano_fit=config.build.piano_fit,
        manifest=config.dataset.manifest,
        subset=config.train.subset,
        output=config.outputs.result,
        export_dir=config.outputs.export_dir,
        metrics=config.outputs.metrics,
        population=config.train.population,
        sigma=config.train.sigma,
        max_evals=config.train.max_evals,
        max_seconds=config.train.max_seconds,
        seed=config.train.seed,
        device=config.train.device,
        evaluator_mode=config.train.evaluator_mode,
        work_dir=config.train.work_dir,
    )


def make_metrics(config: FitConfig) -> Any:
    jsonl = train_evotorch.JsonlMetrics(config.outputs.metrics)
    if not config.wandb.enabled:
        return jsonl
    from piano_fit import wandb_logging

    try:
        import wandb
    except ImportError:
        print(
            "W&B support is unavailable. Install package groups with: cd python && poetry install --with train,wandb",
            file=sys.stderr,
        )
        raise SystemExit(2)

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.run_name,
        mode=config.wandb.mode,
        config=serializable_config_summary(config),
    )
    return train_evotorch.TeeMetrics(jsonl, wandb_logging.WandbMetrics(run))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    if args.dry_run:
        print(json.dumps(config_summary(config), default=_json_default, indent=2, sort_keys=True))
        return 0
    return train_evotorch.run_training(training_args(config), metrics=make_metrics(config))


if __name__ == "__main__":
    raise SystemExit(main())
