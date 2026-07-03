#!/usr/bin/env python3
"""Verify Vintage Upright dataset checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checksums", type=Path, help="Checksum JSON file")
    args = parser.parse_args(argv)

    data = json.loads(args.checksums.read_text(encoding="utf-8"))
    if data.get("algorithm") != "sha256":
        print(f"Unsupported checksum algorithm: {data.get('algorithm')}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for relative, expected in sorted(data.get("files", {}).items()):
        path = REPO_ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"mismatch: {relative}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"verified {len(data.get('files', {}))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
