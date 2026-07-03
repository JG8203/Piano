"""Top-level command dispatcher for Piano fitting tools."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piano-fit", description=__doc__)
    parser.add_argument("command", nargs="?", choices=("prepare", "train", "train-evotorch", "wandb"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args:
        parser.print_help()
        return 0
    if args[0] in ("-h", "--help"):
        parser.print_help()
        return 0

    command = args[0]
    rest = args[1:]
    parser.parse_args([command])

    if command == "prepare":
        from piano_fit import prepare

        return prepare.main(rest)
    if command == "train":
        from piano_fit import train

        return train.main(rest)
    if command == "train-evotorch":
        from piano_fit import train_evotorch

        return train_evotorch.main(rest)
    if command == "wandb":
        from piano_fit import wandb_logging

        return wandb_logging.main(rest)

    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
