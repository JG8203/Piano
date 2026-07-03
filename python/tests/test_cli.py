import contextlib
import io
import unittest

from piano_fit import cli


class PianoFitCliTests(unittest.TestCase):
    def test_top_level_help_exits_zero(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = cli.main(["--help"])

        self.assertEqual(0, result)
        self.assertIn("train-evotorch", stdout.getvalue())

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


if __name__ == "__main__":
    unittest.main()
