#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock
import json
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from piano_fit import prepare as prep


class VintageUprightPrepTests(unittest.TestCase):
    def test_note_name_to_midi_supports_negative_octaves(self):
        self.assertEqual(prep.note_name_to_midi("G#", -1), 8)
        self.assertEqual(prep.note_name_to_midi("A", 0), 21)
        self.assertEqual(prep.note_name_to_midi("C", 4), 60)

    def test_parse_sample_name(self):
        midi_note, layer, velocity = prep.parse_sample_name(Path("A3_F.ncw"))
        self.assertEqual(midi_note, 57)
        self.assertEqual(layer, "F")
        self.assertEqual(velocity, 101 / 127)

    def test_manifest_is_jsonl_sorted_by_note_then_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            records = [
                {"path": "b", "midi_note": 61, "layer": "P"},
                {"path": "a", "midi_note": 60, "layer": "F"},
            ]
            prep.write_manifest(records, manifest)
            lines = manifest.read_text().splitlines()
            self.assertIn('"midi_note": 60', lines[0])
            self.assertIn('"layer": "P"', lines[1])

    def test_build_ncw_converter_uses_pinned_submodule(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            expected_manifest = repo_root / "third_party" / "ncw" / "examples" / "ncw-convert" / "Cargo.toml"
            expected_converter = repo_root / "third_party" / "ncw" / "target" / "release" / "ncw-convert"
            expected_manifest.parent.mkdir(parents=True)
            expected_manifest.write_text("[package]\n", encoding="utf-8")

            with mock.patch.object(prep, "REPO_ROOT", repo_root), mock.patch.object(prep, "run") as run:
                converter = prep.build_ncw_converter()

        self.assertEqual(expected_converter, converter)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual("cargo", command[0])
        self.assertIn(str(expected_manifest), command)

    def test_manifest_paths_are_repo_relative_for_repo_files(self):
        with tempfile.TemporaryDirectory(dir=prep.REPO_ROOT) as tmp:
            wav_path = Path(tmp) / "sample.wav"
            wav_path.write_bytes(b"wav")

            self.assertFalse(Path(prep.manifest_path(wav_path)).is_absolute())

    def test_write_checksums_uses_sha256_for_relative_paths(self):
        with tempfile.TemporaryDirectory(dir=prep.REPO_ROOT) as tmp:
            sample = Path(tmp) / "sample.ncw"
            checksum = Path(tmp) / "checksums.json"
            sample.write_text("tiny sample", encoding="utf-8")

            prep.write_checksums([sample], checksum)

            data = json.loads(checksum.read_text(encoding="utf-8"))
            self.assertEqual("sha256", data["algorithm"])
            key = prep.manifest_path(sample)
            self.assertEqual({"algorithm", "files"}, set(data))
            self.assertEqual(64, len(data["files"][key]))

    def test_raw_dir_argument_selects_sample_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            ncw = raw_dir / "A3_F.ncw"
            ncw.write_bytes(b"ncw")
            converter = Path(tmp) / "ncw-convert"

            def fake_convert(_converter, _ncw_path, wav_path, _overwrite):
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(wav_path), "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(44100)
                    out.writeframes(b"\x00\x00")

            with mock.patch.object(prep, "convert_sample", side_effect=fake_convert), mock.patch.object(prep, "validate_coverage"):
                result = prep.main(
                    [
                        str(Path(tmp) / "legacy-library"),
                        "--raw-dir",
                        str(raw_dir),
                        "--output",
                        str(Path(tmp) / "out"),
                        "--ncw-convert",
                        str(converter),
                        "--jobs",
                        "1",
                        "--no-progress",
                    ]
                )

        self.assertEqual(0, result)

    def test_ncw_convert_argument_overrides_default_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "Vintage Upright"
            (library / "Vintage Upright Samples").mkdir(parents=True)
            override = Path(tmp) / "custom-ncw-convert"

            with mock.patch.object(prep, "build_ncw_converter") as build, mock.patch.object(prep, "write_manifest"), mock.patch.object(prep, "write_checksums"), mock.patch.object(prep, "validate_coverage"):
                result = prep.main([
                    str(library),
                    "--output",
                    str(Path(tmp) / "out"),
                    "--ncw-convert",
                    str(override),
                    "--jobs",
                    "1",
                    "--no-progress",
                ])

        self.assertEqual(0, result)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
