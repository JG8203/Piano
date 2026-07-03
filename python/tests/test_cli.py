import contextlib
import io
import json
import unittest

from piano_fit import cli


class PianoFitCliTests(unittest.TestCase):
    def test_top_level_help_exits_zero(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = cli.main(["--help"])

        self.assertEqual(0, result)
        self.assertIn("train-evotorch", stdout.getvalue())
        self.assertIn("train", stdout.getvalue())

    def test_prepare_subcommand_forwards_to_prepare_parser(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["prepare", "--help"])

        self.assertEqual(0, raised.exception.code)
        self.assertIn("Vintage Upright", stdout.getvalue())

    def test_train_evotorch_subcommand_forwards_to_training_parser(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["train-evotorch", "--help"])

        self.assertEqual(0, raised.exception.code)
        self.assertIn("--piano-fit", stdout.getvalue())

    def test_train_subcommand_forwards_to_training_parser(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["train", "--help"])

        self.assertEqual(0, raised.exception.code)
        self.assertIn("--config", stdout.getvalue())

    def test_train_dry_run_prints_json(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = cli.main(["train", "--config", "configs/local-smoke.yaml", "--dry-run"])

        self.assertEqual(0, result)
        data = json.loads(stdout.getvalue())
        self.assertEqual(32, data["train"]["max_evals"])
        self.assertEqual("cpu", data["train"]["device"])


if __name__ == "__main__":
    unittest.main()
