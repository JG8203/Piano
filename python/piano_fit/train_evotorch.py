#!/usr/bin/env python3
"""Run Piano fitting with EvoTorch driving the evolutionary search.

The C++ PianoFit binary remains the renderer/loss evaluator. This wrapper lets
EvoTorch own the search distribution and sends each generated population to
PianoFit in one JSONL batch.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch
from evotorch import Problem
from tqdm import tqdm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piano-fit", type=Path, required=True, help="Path to the PianoFit executable")
    parser.add_argument("--manifest", type=Path, required=True, help="Prepared Vintage Upright manifest JSONL")
    parser.add_argument("--subset", default="pilot", choices=("pilot", "all"), help="Dataset subset")
    parser.add_argument("--output", type=Path, default=Path("piano_fit_evotorch_result.json"), help="Final result JSON path")
    parser.add_argument("--export-dir", type=Path, default=None, help="Optional final target/render WAV export directory")
    parser.add_argument("--metrics", type=Path, default=None, help="Optional wrapper metrics JSONL path")
    parser.add_argument("--population", type=int, default=40, help="EvoTorch population size")
    parser.add_argument("--sigma", type=float, default=0.15, help="Initial CMA-ES standard deviation")
    parser.add_argument("--max-evals", type=int, default=10000, help="Approximate evaluation budget, rounded up to a full population")
    parser.add_argument("--max-seconds", type=float, default=6.0, help="Target/render crop duration passed to PianoFit")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--device", default="cpu", help="EvoTorch torch device")
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory for temporary genome batches")
    parser.add_argument(
        "--evaluator-mode",
        default="stdio",
        choices=("stdio", "files"),
        help="Use one long-lived stdio evaluator or per-batch JSONL files",
    )
    return parser.parse_args(argv)


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
    return json.loads(completed.stdout)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def get_genome_size(piano_fit: Path) -> int:
    info = run_json([str(piano_fit), "--print-model-info"])
    return int(info["genome_size"])


def tensor_to_rows(values: Any) -> list[list[float]]:
    if not isinstance(values, torch.Tensor) and hasattr(values, "values"):
        batch_values = values.values
        values = batch_values() if callable(batch_values) else batch_values
    tensor = torch.as_tensor(values).detach().to("cpu", dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return [[float(x) for x in row] for row in tensor.tolist()]


def write_genomes(path: Path, genomes: list[list[float]]) -> list[str]:
    ids: list[str] = []
    with path.open("w", encoding="utf-8") as out:
        for index, genome in enumerate(genomes):
            candidate_id = str(index)
            ids.append(candidate_id)
            out.write(json.dumps({"id": candidate_id, "genome": genome}, separators=(",", ":")) + "\n")
    return ids


def read_losses(path: Path) -> list[float]:
    losses_by_id: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        losses_by_id[str(record["id"])] = float(record["loss"])
    return [losses_by_id[str(index)] for index in range(len(losses_by_id))]


def parse_evaluator_response(line: str, batch_id: str, ids: list[str]) -> list[float]:
    try:
        response = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid evaluator JSON response: {exc}") from exc

    if response.get("type") == "error":
        raise RuntimeError(f"PianoFit evaluator error: {response.get('message', 'unknown error')}")
    if response.get("type") != "result":
        raise RuntimeError(f"Unexpected evaluator response type: {response.get('type')!r}")
    if response.get("batch_id") != batch_id:
        raise RuntimeError(f"Evaluator batch_id mismatch: expected {batch_id}, got {response.get('batch_id')}")

    losses = response.get("losses")
    if not isinstance(losses, list):
        raise RuntimeError("Evaluator response is missing losses")
    losses_by_id: dict[str, float] = {}
    for item in losses:
        if not isinstance(item, dict):
            raise RuntimeError("Evaluator loss item must be an object")
        losses_by_id[str(item["id"])] = float(item["loss"])
    try:
        return [losses_by_id[candidate_id] for candidate_id in ids]
    except KeyError as exc:
        raise RuntimeError(f"Evaluator response missing loss for id {exc.args[0]}") from exc


class StdioEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.batch_index = 0
        self.stderr_lines: list[str] = []
        self.process = subprocess.Popen(
            [
                str(args.piano_fit),
                "--manifest",
                str(args.manifest),
                "--subset",
                args.subset,
                "--max-seconds",
                str(args.max_seconds),
                "--serve-jsonl",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=repo_root(),
        )
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_lines.append(line)

    def _stderr_text(self) -> str:
        return "".join(self.stderr_lines).strip()

    def _protocol_error(self, message: str) -> RuntimeError:
        stderr = self._stderr_text()
        if stderr:
            return RuntimeError(f"{message}\nPianoFit stderr:\n{stderr}")
        return RuntimeError(message)

    def evaluate(self, genomes: list[list[float]]) -> list[float]:
        if self.process.poll() is not None:
            raise self._protocol_error(f"PianoFit evaluator exited with code {self.process.returncode}")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("PianoFit evaluator pipes are unavailable")

        batch_id = f"{self.batch_index:06d}"
        self.batch_index += 1
        ids = [str(index) for index in range(len(genomes))]
        request = {
            "type": "evaluate",
            "batch_id": batch_id,
            "genomes": [{"id": candidate_id, "genome": genome} for candidate_id, genome in zip(ids, genomes)],
        }
        try:
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except BrokenPipeError as exc:
            raise self._protocol_error("PianoFit evaluator pipe closed") from exc

        if not line:
            raise self._protocol_error(f"PianoFit evaluator exited before responding to batch {batch_id}")
        try:
            return parse_evaluator_response(line, batch_id, ids)
        except RuntimeError as exc:
            raise self._protocol_error(str(exc)) from exc

    def close(self) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write(json.dumps({"type": "shutdown"}, separators=(",", ":")) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()

    def __enter__(self) -> "StdioEvaluator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class JsonlMetrics:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, separators=(",", ":")) + "\n")


class TeeMetrics:
    def __init__(self, *metrics: Any) -> None:
        self.metrics = metrics

    def log(self, record: dict[str, Any]) -> None:
        for metrics in self.metrics:
            metrics.log(record)


def make_cmaes(problem: Any, population: int, sigma: float, genome_size: int) -> Any:
    from evotorch.algorithms import CMAES

    center = torch.full((genome_size,), 0.5, dtype=torch.float32)
    attempts = (
        {"stdev_init": sigma, "popsize": population, "center_init": center},
        {"stdev_init": sigma, "pop_size": population, "center_init": center},
        {"stdev_init": sigma, "popsize": population},
        {"stdev_init": sigma, "pop_size": population},
    )
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return CMAES(problem, **kwargs)
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not construct EvoTorch CMAES: {last_error}")


def run_training(args: argparse.Namespace, metrics: Any | None = None) -> int:
    if args.population < 4:
        raise SystemExit("--population must be at least 4")
    if args.max_evals < 1:
        raise SystemExit("--max-evals must be at least 1")

    evaluator_mode = getattr(args, "evaluator_mode", "stdio")
    if evaluator_mode not in ("stdio", "files"):
        raise SystemExit("--evaluator-mode must be one of: stdio, files")

    genome_size = get_genome_size(args.piano_fit)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="piano-fit-evotorch-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics or JsonlMetrics(args.metrics)

    best_loss = math.inf
    best_genome = [0.5] * genome_size
    evaluations = 0
    batch_index = 0

    metrics.log(
        {
            "type": "config",
            "optimizer": "evotorch.cmaes",
            "genome_size": genome_size,
            "population": args.population,
            "sigma": args.sigma,
            "max_evaluations": args.max_evals,
            "subset": args.subset,
            "max_seconds": args.max_seconds,
            "seed": args.seed,
            "evaluator_mode": evaluator_mode,
        }
    )

    stdio_evaluator: StdioEvaluator | None = None

    def objective(values: Any) -> Any:
        nonlocal best_loss, best_genome, evaluations, batch_index
        genomes = tensor_to_rows(values)

        if evaluator_mode == "stdio":
            if stdio_evaluator is None:
                raise RuntimeError("Stdio evaluator was not initialized")
            losses = stdio_evaluator.evaluate(genomes)
        else:
            input_path = work_dir / f"population-{batch_index:06d}.jsonl"
            output_path = work_dir / f"losses-{batch_index:06d}.jsonl"
            batch_index += 1
            write_genomes(input_path, genomes)

            command = [
                str(args.piano_fit),
                "--manifest",
                str(args.manifest),
                "--subset",
                args.subset,
                "--max-seconds",
                str(args.max_seconds),
                "--eval-genomes",
                str(input_path),
                "--eval-output",
                str(output_path),
            ]
            run_checked(command, cwd=repo_root())
            losses = read_losses(output_path)

        for genome, loss in zip(genomes, losses):
            evaluations += 1
            if loss < best_loss:
                best_loss = loss
                best_genome = genome
            metrics.log(
                {
                    "type": "evaluation",
                    "evaluation": evaluations,
                    "loss": loss,
                    "best_loss": best_loss,
                    "phase": "evotorch",
                }
            )

        return torch.tensor(losses, dtype=torch.float32, device=args.device)

    torch.manual_seed(args.seed)
    problem = Problem(
        "min",
        objective,
        solution_length=genome_size,
        initial_bounds=(0.0, 1.0),
        vectorized=True,
        dtype=torch.float32,
        device=args.device,
        seed=args.seed,
    )
    searcher = make_cmaes(problem, args.population, args.sigma, genome_size)

    try:
        if evaluator_mode == "stdio":
            stdio_evaluator = StdioEvaluator(args)

        generations = math.ceil(args.max_evals / args.population)
        for generation in tqdm(range(generations), desc="EvoTorch CMA-ES", unit="gen"):
            searcher.step()
            metrics.log(
                {
                    "type": "generation",
                    "generation": generation + 1,
                    "evaluation": evaluations,
                    "best_loss": best_loss,
                }
            )
            if evaluations >= args.max_evals:
                break
    finally:
        if stdio_evaluator is not None:
            stdio_evaluator.close()

    best_path = work_dir / "best-genome.jsonl"
    write_genomes(best_path, [best_genome])
    final_command = [
        str(args.piano_fit),
        "--manifest",
        str(args.manifest),
        "--subset",
        args.subset,
        "--max-seconds",
        str(args.max_seconds),
        "--fixed-genome",
        str(best_path),
        "--output",
        str(args.output),
    ]
    if args.export_dir:
        final_command.extend(["--export-dir", str(args.export_dir)])
    run_checked(final_command, cwd=repo_root())
    metrics.log({"type": "final", "best_loss": best_loss, "evaluations": evaluations})
    print(f"Best loss: {best_loss}")
    print(f"Result: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_training(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
