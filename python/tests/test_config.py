import tempfile
import unittest
from pathlib import Path

import yaml

from piano_fit.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_checked_in_configs(self):
        for path in sorted(Path("configs").glob("*.yaml")):
            with self.subTest(path=path):
                cfg = load_config(path)
                self.assertTrue(cfg.dataset.manifest.is_absolute())
                self.assertTrue(cfg.build.piano_fit.is_absolute())
                self.assertTrue(cfg.outputs.result.is_absolute())

    def test_relative_paths_resolve_from_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "nested"
            config_dir.mkdir()
            config_path = config_dir / "fit.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"manifest": "../data/manifest.jsonl"},
                        "build": {"piano_fit": "../bin/PianoFit"},
                        "train": {
                            "subset": "pilot",
                            "population": 8,
                            "sigma": 0.15,
                            "max_evals": 32,
                            "max_seconds": 1,
                            "seed": 12345,
                            "device": "cpu",
                            "work_dir": "../work",
                        },
                        "outputs": {
                            "metrics": "../out/metrics.jsonl",
                            "result": "../out/result.json",
                            "export_dir": "../out/audio",
                        },
                        "wandb": {
                            "enabled": False,
                            "project": "piano-fit",
                            "entity": None,
                            "run_name": "test",
                            "mode": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg = load_config(config_path)

        self.assertEqual((root / "data" / "manifest.jsonl").resolve(), cfg.dataset.manifest)
        self.assertEqual((root / "work").resolve(), cfg.train.work_dir)

    def test_missing_required_field_names_the_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "fit.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {},
                        "build": {"piano_fit": "PianoFit"},
                        "train": {
                            "subset": "pilot",
                            "population": 8,
                            "sigma": 0.15,
                            "max_evals": 32,
                            "max_seconds": 1,
                            "seed": 12345,
                            "device": "cpu",
                        },
                        "outputs": {"result": "result.json"},
                        "wandb": {"enabled": False, "project": "piano-fit"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as raised:
                load_config(config_path)

        self.assertIn("dataset.manifest", str(raised.exception))

    def test_gcp_gpu_uses_cuda(self):
        cfg = load_config("configs/gcp-gpu.yaml")

        self.assertEqual("cuda", cfg.train.device)


if __name__ == "__main__":
    unittest.main()
