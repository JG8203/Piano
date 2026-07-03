# Vintage Upright Fitting Pipeline

This folder contains the dataset preparation side of the offline fitting flow.
The C++ optimizer target lives in `tools/piano_fit`.

## Prepare The Dataset

Install the Python package and training integrations:

```bash
cd python
poetry install --with train,wandb
poetry run piano-fit prepare --help
poetry run piano-fit train-evotorch --help
```

Initialize the pinned NCW decoder fork before converting samples:

```bash
git submodule update --init --recursive third_party/ncw
```

Normal training should use the committed prepared dataset:

```text
data/vintage-upright/manifest.jsonl
```

To rebuild the committed WAVs and manifest from the imported raw samples:

```bash
cd python
poetry run piano-fit prepare \
  data/vintage-upright \
  --raw-dir data/vintage-upright/raw \
  --output data/vintage-upright \
  --wav-dir data/vintage-upright/wav \
  --manifest data/vintage-upright/manifest.jsonl \
  --jobs 8 \
  --overwrite
```

The script converts Kontakt `.ncw` files to WAV, writes a manifest, and records note, velocity layer, audio
metadata, peak, and RMS statistics. It builds `ncw-convert` from the pinned
`third_party/ncw` submodule, sourced from the `JG8203/ncw` fork with the Vintage
Upright decoder fixes committed there. Conversion progress is shown with `tqdm`;
pass `--no-progress` for quiet logs. Use `--jobs 1` for serial conversion if you
are debugging decoder behavior, or raise/lower `--jobs` to match your machine
and disk speed.

Once Plan 002 lands, normal training runs should use the committed/prepared WAV
files instead of converting from `.ncw` during each run.

For a quick decoder check:

```bash
cd python
poetry run piano-fit prepare \
  "/Users/armaine/Downloads/Vintage Upright" \
  --output /tmp/vintage-upright-check \
  --only A3_F C4_F C4_M C4_P
```

## Build The Fitter

```bash
cmake -B build-fit -DCMAKE_BUILD_TYPE=Release -DBUILD_FIT_TOOLS=ON
cmake --build build-fit --target PianoFit --config Release -j 2
```

## Run A Smoke Pass

```bash
./build-fit/tools/piano_fit/PianoFit_artefacts/Release/PianoFit \
  --manifest data/vintage-upright/manifest.jsonl \
  --subset pilot \
  --max-evals 0 \
  --max-seconds 1 \
  --metrics build/vintage-upright/smoke-metrics.jsonl \
  --output build/vintage-upright/smoke.json \
  --export-dir build/vintage-upright/smoke-audio
```

## Run Pilot Optimization With EvoTorch

Let EvoTorch drive CMA-ES while `PianoFit` evaluates each population:

```bash
cd python
poetry run piano-fit train-evotorch \
  --piano-fit ./build-fit/tools/piano_fit/PianoFit_artefacts/Release/PianoFit \
  --manifest data/vintage-upright/manifest.jsonl \
  --subset pilot \
  --population 40 \
  --sigma 0.15 \
  --max-evals 10000 \
  --max-seconds 6 \
  --metrics build/vintage-upright/pilot-metrics.jsonl \
  --output build/vintage-upright/pilot-fit.json \
  --export-dir build/vintage-upright/pilot-audio
```

Use `--subset all` after the pilot loss is moving in the right direction.
The original built-in C++ CMA-ES path is still available by running `PianoFit`
directly with `--max-evals`, which is useful as a smaller dependency fallback.

## Run With Weights & Biases

Log in to W&B, then wrap the normal fitter command:

```bash
wandb login
```

```bash
cd python
poetry run piano-fit wandb \
  --wandb-project piano-fit \
  --wandb-run-name vintage-upright-pilot \
  -- \
  ./build-fit/tools/piano_fit/PianoFit_artefacts/Release/PianoFit \
  --manifest data/vintage-upright/manifest.jsonl \
  --subset pilot \
  --population 40 \
  --sigma 0.15 \
  --max-evals 10000 \
  --max-seconds 6 \
  --output build/vintage-upright/pilot-fit.json \
  --export-dir build/vintage-upright/pilot-audio
```

The wrapper automatically adds `--metrics` if it is missing, logs evaluation
loss, best loss, generation sigma, final improvement, and saves the result JSON
plus exported WAVs as run artifacts. Use `--wandb-mode offline` for offline
logging.

## Compatibility Wrappers

The legacy entry points remain as thin wrappers for saved commands:

```bash
tools/vintage_upright/prepare_vintage_upright.py --help
tools/piano_fit/run_with_evotorch.py --help
tools/piano_fit/run_with_wandb.py --help
```
