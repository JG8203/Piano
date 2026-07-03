#!/usr/bin/env python3
"""Compatibility wrapper for the packaged EvoTorch training command."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from piano_fit.train_evotorch import main


if __name__ == "__main__":
    raise SystemExit(main())
