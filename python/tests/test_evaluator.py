import argparse
import io
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from piano_fit.train_evotorch import StdioEvaluator, get_genome_size, parse_evaluator_response


class EvaluatorProtocolTests(unittest.TestCase):
    def test_parse_response_returns_losses_in_candidate_order(self):
        line = json.dumps(
            {
                "type": "result",
                "batch_id": "000001",
                "losses": [{"id": "1", "loss": 2.5}, {"id": "0", "loss": 1.25}],
            }
        )

        self.assertEqual([1.25, 2.5], parse_evaluator_response(line, "000001", ["0", "1"]))

    def test_parse_response_rejects_batch_id_mismatch(self):
        line = json.dumps({"type": "result", "batch_id": "other", "losses": []})

        with self.assertRaisesRegex(RuntimeError, "batch_id mismatch"):
            parse_evaluator_response(line, "000001", [])

    def test_close_sends_shutdown_and_terminates_when_process_lingers(self):
        class FakeStdin(io.StringIO):
            def close(self):
                self.closed_called = True

        class FakeProcess:
            def __init__(self):
                self.stdin = FakeStdin()
                self.stdout = None
                self.stderr = None
                self.returncode = None
                self.terminated = False
                self.killed = False

            def poll(self):
                return None

            def wait(self, timeout=None):
                if not self.terminated:
                    raise subprocess.TimeoutExpired("PianoFit", timeout)
                self.returncode = 0
                return 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

        evaluator = StdioEvaluator.__new__(StdioEvaluator)
        evaluator.process = FakeProcess()

        evaluator.close()

        self.assertIn('"type":"shutdown"', evaluator.process.stdin.getvalue())
        self.assertTrue(evaluator.process.terminated)
        self.assertFalse(evaluator.process.killed)


class EvaluatorIntegrationTests(unittest.TestCase):
    def binary_and_manifest(self):
        repo = Path(__file__).resolve().parents[2]
        binary = repo / "build-fit/tools/piano_fit/PianoFit_artefacts/Release/PianoFit"
        manifest = repo / "data/vintage-upright/manifest.jsonl"
        if not binary.exists():
            self.skipTest(f"PianoFit binary not found: {binary}")
        if not manifest.exists():
            self.skipTest(f"Vintage Upright manifest not found: {manifest}")
        return repo, binary, manifest

    def test_stdio_evaluator_returns_finite_positive_loss(self):
        _, binary, manifest = self.binary_and_manifest()
        genome_size = get_genome_size(binary)
        args = argparse.Namespace(
            piano_fit=binary,
            manifest=manifest,
            subset="pilot",
            max_seconds=0.25,
        )
        with StdioEvaluator(args) as evaluator:
            losses = evaluator.evaluate([[0.5] * genome_size])

        self.assertEqual(1, len(losses))
        self.assertTrue(math.isfinite(losses[0]))
        self.assertGreater(losses[0], 0.0)

    def test_stdio_and_file_mode_match_for_same_genome(self):
        repo, binary, manifest = self.binary_and_manifest()
        genome_size = get_genome_size(binary)
        genome = [0.5] * genome_size

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "genomes.jsonl"
            output_path = Path(temp_dir) / "losses.jsonl"
            input_path.write_text(json.dumps({"id": "0", "genome": genome}) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    str(binary),
                    "--manifest",
                    str(manifest),
                    "--subset",
                    "pilot",
                    "--max-seconds",
                    "0.25",
                    "--eval-genomes",
                    str(input_path),
                    "--eval-output",
                    str(output_path),
                ],
                cwd=repo,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            file_loss = float(json.loads(output_path.read_text(encoding="utf-8"))["loss"])

        args = argparse.Namespace(
            piano_fit=binary,
            manifest=manifest,
            subset="pilot",
            max_seconds=0.25,
        )
        with StdioEvaluator(args) as evaluator:
            stdio_loss = evaluator.evaluate([genome])[0]

        self.assertTrue(math.isfinite(file_loss))
        self.assertTrue(math.isfinite(stdio_loss))
        self.assertEqual(file_loss, stdio_loss)


if __name__ == "__main__":
    unittest.main()
