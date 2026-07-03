#!/usr/bin/env python3
"""Prepare the Vintage Upright NCW sample set for Piano fitting.

The output manifest is newline-delimited JSON so the C++ fitter can parse it
without pulling in a JSON dependency.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import warnings
import wave
from pathlib import Path
from typing import Iterable

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop
except ImportError:
    audioop = None


NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

LAYER_TO_VELOCITY = {
    "P": 0.35,
    "M": 0.65,
    "F": 1.0,
}

SAMPLE_RE = re.compile(r"^([A-G](?:#)?)(-?\d)_([FMP])\.ncw$")
REPO_ROOT = Path(__file__).resolve().parents[2]


def note_name_to_midi(note: str, octave: int) -> int:
    return (octave + 1) * 12 + NOTE_TO_SEMITONE[note]


def parse_sample_name(path: Path) -> tuple[int, str, float]:
    match = SAMPLE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected Vintage Upright sample name: {path.name}")

    note_name, octave_text, layer = match.groups()
    midi_note = note_name_to_midi(note_name, int(octave_text))
    return midi_note, layer, LAYER_TO_VELOCITY[layer]


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def tqdm_or_raise():
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required for progress display. Install it with: python3 -m pip install tqdm") from exc
    return tqdm


def iter_with_progress(paths: list[Path], enabled: bool) -> Iterable[Path]:
    if not enabled:
        yield from paths
        return

    tqdm = tqdm_or_raise()

    yield from tqdm(paths, desc="Converting Vintage Upright", unit="sample")


def default_jobs() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_ncw_converter() -> Path:
    """Build the pinned ncw-convert from third_party/ncw."""
    source = REPO_ROOT / "third_party" / "ncw"
    manifest = source / "examples" / "ncw-convert" / "Cargo.toml"
    converter = source / "target" / "release" / ("ncw-convert.exe" if os.name == "nt" else "ncw-convert")
    if converter.exists():
        return converter

    if not manifest.exists():
        raise RuntimeError(
            "Pinned NCW converter source is missing. Run: "
            "git submodule update --init --recursive third_party/ncw"
        )

    build_env = os.environ.copy()
    existing_rustflags = build_env.get("RUSTFLAGS", "")
    native_cpu = "-C target-cpu=native"
    build_env["RUSTFLAGS"] = f"{existing_rustflags} {native_cpu}".strip() if native_cpu not in existing_rustflags else existing_rustflags

    run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(manifest),
        ],
        env=build_env,
    )
    return converter


def convert_sample(converter: Path, ncw_path: Path, wav_path: Path, overwrite: bool) -> None:
    if wav_path.exists() and not overwrite:
        return

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run([str(converter), str(ncw_path), str(wav_path)])


def manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def convert_and_describe_sample(converter: Path, wav_root: Path, overwrite: bool, ncw_path: Path) -> dict[str, object]:
    midi_note, layer, velocity = parse_sample_name(ncw_path)
    wav_path = wav_root / f"{ncw_path.stem}.wav"
    convert_sample(converter, ncw_path, wav_path, overwrite)
    stats = wav_stats(wav_path)
    return {
        "source_path": manifest_path(ncw_path),
        "path": manifest_path(wav_path),
        "midi_note": midi_note,
        "layer": layer,
        "target_velocity": velocity,
        **stats,
    }


def pcm_bytes_to_float(raw: bytes, sample_width: int) -> Iterable[float]:
    if sample_width == 1:
        for value in raw:
            yield (value - 128) / 128.0
    elif sample_width == 2:
        for (value,) in struct.iter_unpack("<h", raw):
            yield value / 32768.0
    elif sample_width == 3:
        for i in range(0, len(raw), 3):
            b0, b1, b2 = raw[i : i + 3]
            value = b0 | (b1 << 8) | (b2 << 16)
            if value & 0x800000:
                value -= 0x1000000
            yield value / 8388608.0
    elif sample_width == 4:
        for (value,) in struct.iter_unpack("<i", raw):
            yield value / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def pcm_full_scale(sample_width: int) -> float:
    if sample_width == 1:
        return 128.0
    if sample_width == 2:
        return 32768.0
    if sample_width == 3:
        return 8388608.0
    if sample_width == 4:
        return 2147483648.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def wav_stats(wav_path: Path) -> dict[str, float | int]:
    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(frames)

    if audioop is not None and sample_width != 1:
        scale = pcm_full_scale(sample_width)
        peak = audioop.max(raw, sample_width) / scale
        rms = audioop.rms(raw, sample_width) / scale
    else:
        peak = 0.0
        sum_sq = 0.0
        count = 0
        for sample in pcm_bytes_to_float(raw, sample_width):
            peak = max(peak, abs(sample))
            sum_sq += sample * sample
            count += 1
        rms = math.sqrt(sum_sq / max(1, count))

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": frames / float(sample_rate),
        "peak": peak,
        "rms": rms,
    }


def write_manifest(records: list[dict[str, object]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as out:
        for record in sorted(records, key=lambda r: (int(r["midi_note"]), str(r["layer"]))):
            out.write(json.dumps(record, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(paths: Iterable[Path], checksum_path: Path) -> None:
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    files = {manifest_path(path): sha256_file(path) for path in sorted(paths, key=manifest_path)}
    checksum_path.write_text(
        json.dumps({"algorithm": "sha256", "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_coverage(records: list[dict[str, object]]) -> None:
    by_note: dict[int, set[str]] = {}
    for record in records:
        by_note.setdefault(int(record["midi_note"]), set()).add(str(record["layer"]))

    missing = {
        midi: sorted(set(LAYER_TO_VELOCITY) - layers)
        for midi, layers in by_note.items()
        if layers != set(LAYER_TO_VELOCITY)
    }

    print(f"Prepared {len(records)} samples across {len(by_note)} notes")
    if missing:
        print("Missing layers:")
        for midi, layers in sorted(missing.items()):
            print(f"  midi {midi}: {','.join(layers)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library_root", type=Path, help="Path to the Vintage Upright library root")
    parser.add_argument("--output", type=Path, default=Path("build/vintage-upright"), help="Prepared dataset output directory")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory containing Vintage Upright .ncw files")
    parser.add_argument("--wav-dir", type=Path, default=None, help="Directory for converted WAV files")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest JSONL path")
    parser.add_argument("--checksums", type=Path, default=None, help="Checksum JSON path")
    parser.add_argument("--ncw-convert", type=Path, default=None, help="Optional prebuilt ncw-convert path")
    parser.add_argument("--overwrite", action="store_true", help="Reconvert existing WAV files")
    parser.add_argument("--only", nargs="*", default=None, help="Optional sample stems to convert, e.g. A3_F C4_F")
    parser.add_argument("--jobs", type=positive_int, default=default_jobs(), help="Concurrent conversion jobs")
    parser.add_argument("--no-progress", action="store_true", help="Disable conversion progress display")
    args = parser.parse_args(argv)

    sample_dir = args.raw_dir if args.raw_dir else args.library_root / "Vintage Upright Samples"
    if not sample_dir.is_dir():
        print(f"Sample directory not found: {sample_dir}", file=sys.stderr)
        return 2

    output = args.output.resolve()
    wav_root = args.wav_dir.resolve() if args.wav_dir else output / "wav"
    manifest_path = args.manifest or (output / "manifest.jsonl")
    checksum_path = args.checksums or (output / "checksums.json")
    converter = args.ncw_convert.resolve() if args.ncw_convert else build_ncw_converter()

    records: list[dict[str, object]] = []
    only = set(args.only) if args.only else None
    sample_paths = [
        ncw_path
        for ncw_path in sorted(sample_dir.glob("*.ncw"))
        if only is None or ncw_path.stem in only
    ]

    jobs = min(args.jobs, max(1, len(sample_paths)))
    if jobs == 1:
        for ncw_path in iter_with_progress(sample_paths, enabled=not args.no_progress):
            records.append(convert_and_describe_sample(converter, wav_root, args.overwrite, ncw_path))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(convert_and_describe_sample, converter, wav_root, args.overwrite, ncw_path)
                for ncw_path in sample_paths
            ]
            completed = as_completed(futures)
            if not args.no_progress:
                tqdm = tqdm_or_raise()
                completed = tqdm(completed, total=len(futures), desc=f"Converting Vintage Upright ({jobs} jobs)", unit="sample")
            for future in completed:
                records.append(future.result())

    write_manifest(records, manifest_path)
    wav_paths = [wav_root / f"{ncw_path.stem}.wav" for ncw_path in sample_paths]
    write_checksums([*sample_paths, *wav_paths, manifest_path], checksum_path)
    validate_coverage(records)
    print(f"Manifest: {manifest_path}")
    print(f"Checksums: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
