#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_vintage_upright as prep


class VintageUprightPrepTests(unittest.TestCase):
    def test_note_name_to_midi_supports_negative_octaves(self):
        self.assertEqual(prep.note_name_to_midi("G#", -1), 8)
        self.assertEqual(prep.note_name_to_midi("A", 0), 21)
        self.assertEqual(prep.note_name_to_midi("C", 4), 60)

    def test_parse_sample_name(self):
        midi_note, layer, velocity = prep.parse_sample_name(Path("A3_F.ncw"))
        self.assertEqual(midi_note, 57)
        self.assertEqual(layer, "F")
        self.assertEqual(velocity, 1.0)

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

    def test_ncw_convert_argument_overrides_default_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "Vintage Upright"
            (library / "Vintage Upright Samples").mkdir(parents=True)
            override = Path(tmp) / "custom-ncw-convert"

            with mock.patch.object(prep, "build_ncw_converter") as build, mock.patch.object(prep, "write_manifest"), mock.patch.object(prep, "validate_coverage"):
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
