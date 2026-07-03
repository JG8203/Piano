import contextlib
import io
import json
import sys
import unittest
from unittest import mock

from piano_fit import train


class TrainCommandTests(unittest.TestCase):
    def dry_run(self, *args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = train.main(["--config", "configs/local-smoke.yaml", "--dry-run", *args])
        self.assertEqual(0, result)
        return json.loads(stdout.getvalue())

    def test_cli_overrides_win_over_config(self):
        data = self.dry_run(
            "--max-evals",
            "64",
            "--device",
            "cpu",
            "--output",
            "override-result.json",
            "--metrics",
            "override-metrics.jsonl",
            "--export-dir",
            "override-audio",
        )

        self.assertEqual(64, data["train"]["max_evals"])
        self.assertEqual("cpu", data["train"]["device"])
        self.assertTrue(data["outputs"]["result"].endswith("/override-result.json"))
        self.assertTrue(data["outputs"]["metrics"].endswith("/override-metrics.jsonl"))
        self.assertTrue(data["outputs"]["export_dir"].endswith("/override-audio"))

    def test_wandb_flags_override_config(self):
        enabled = self.dry_run("--wandb", "--wandb-mode", "offline")
        disabled = self.dry_run("--wandb", "--no-wandb")

        self.assertTrue(enabled["wandb"]["enabled"])
        self.assertEqual("offline", enabled["wandb"]["mode"])
        self.assertFalse(disabled["wandb"]["enabled"])

    def test_dry_run_does_not_launch_training_or_import_wandb(self):
        with mock.patch("piano_fit.train_evotorch.run_checked") as run_checked, mock.patch(
            "piano_fit.train_evotorch.get_genome_size"
        ) as get_genome_size, mock.patch.dict(sys.modules, {"wandb": None}):
            data = self.dry_run("--wandb", "--wandb-mode", "offline")

        self.assertTrue(data["wandb"]["enabled"])
        run_checked.assert_not_called()
        get_genome_size.assert_not_called()


if __name__ == "__main__":
    unittest.main()
