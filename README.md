# Piano

[![Tests](https://github.com/FigBug/Piano/actions/workflows/test.yaml/badge.svg)](https://github.com/FigBug/Piano/actions/workflows/test.yaml)
[![Performance](https://github.com/FigBug/Piano/actions/workflows/performance.yaml/badge.svg)](https://figbug.github.io/Piano/dev/bench/)

A physically modeled piano synthesizer plugin using digital waveguide synthesis.

This repository also includes an offline fitting pipeline for tuning the model
against prepared Vintage Upright samples. The realtime plugin remains C++/JUCE;
the fitting workflow uses a C++ `PianoFit` evaluator plus Python orchestration.

## Features

- Physical modeling synthesis based on digital waveguide string models
- Hammer-string interaction simulation
- Longitudinal string modes
- Soundboard resonance
- Available as VST, VST3, AU (macOS), and LV2 (Linux) plugins
- Offline fitting tools with Poetry, EvoTorch, W&B support, Docker, and GCP-ready
  CPU/GPU profiles

## Repository Layout

```text
Source/                  JUCE plugin and synthesis implementation
tests/                   C++ unit and audio integration tests
python/                  piano-fit Python package and config profiles
tools/piano_fit/         C++ PianoFit fitting evaluator
tools/vintage_upright/   Vintage Upright dataset/fitting workflow docs
data/vintage-upright/    Prepared dataset metadata and sample placeholders
docker/                  GCP-ready fitting container docs and Dockerfile
modules/                 JUCE and other submodules
third_party/ncw/         Pinned NCW decoder fork used by dataset preparation
```

## Clone The Repository

Clone with submodules so JUCE, gin, and the pinned NCW decoder are available:

```bash
git clone --recurse-submodules https://github.com/JG8203/Piano.git
cd Piano
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

The prepared Vintage Upright metadata is included in the repo. Audio payloads
are represented by small placeholder files so cloning the fork does not require
private sample storage access. Full fitting runs need hydrated sample files from an
approved dataset source or an externally mounted dataset path.

## Requirements

### Plugin Build

- CMake 3.16 or higher
- C++17 compatible compiler
- Platform dependencies:
  - **macOS**: Xcode and command line tools
  - **Windows**: Visual Studio 2019 or higher
  - **Linux**: GCC or Clang, ALSA, JACK, X11, Mesa, freetype, curl, WebKitGTK,
    and related development packages

On Ubuntu, the CI dependency set is:

```bash
sudo apt-get update
sudo apt-get install -y \
  clang ninja-build ladspa-sdk freeglut3-dev g++ libasound2-dev \
  libcurl4-openssl-dev libfreetype6-dev libjack-jackd2-dev libx11-dev \
  libxcomposite-dev libxcursor-dev libxinerama-dev libxrandr-dev \
  mesa-common-dev libwebkit2gtk-4.1-dev xvfb
```

### Fitting Pipeline

- Python 3.11 or 3.12
- Poetry
- Rust/Cargo for rebuilding the NCW conversion helper
- Hydrated Vintage Upright samples for full fitting runs
- Docker if you want to use the containerized GCP workflow

## Build The Plugin

```bash
cmake -B build
cmake --build build --config Release
```

The built plugins will be in `build/Piano_artefacts/Release/`.

For Ninja or explicit release builds:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Run Tests

The project includes C++ unit tests for synthesis internals and integration
tests for audio output.

```bash
./test.sh
./test.sh Release ON
./test.sh Debug
```

Or run the CMake test target directly:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build --target PianoTests --config Release
cd build
ctest --output-on-failure -C Release
```

On Linux CI, tests run under `xvfb` because plugin initialization touches GUI
facilities.

## Use The Fitting Pipeline

Install the Python package and training dependencies:

```bash
cd python
poetry install --with train,wandb
poetry run piano-fit --help
cd ..
```

Build the C++ fitting evaluator:

```bash
cmake -B build-fit -DCMAKE_BUILD_TYPE=Release -DBUILD_FIT_TOOLS=ON
cmake --build build-fit --target PianoFit --config Release -j 2
./build-fit/tools/piano_fit/PianoFit_artefacts/Release/PianoFit --print-model-info
```

Run a config dry-run, then a small local smoke profile:

```bash
cd python
poetry run piano-fit train --config configs/local-smoke.yaml --dry-run
poetry run piano-fit train --config configs/local-smoke.yaml
cd ..
```

The fitting command defaults to `evaluator_mode: stdio`, keeping the C++
`PianoFit --serve-jsonl` process alive across EvoTorch batches. Use
`--evaluator-mode files` when debugging the legacy per-batch JSONL path.

More detail lives in [tools/vintage_upright/README.md](tools/vintage_upright/README.md).

## Track Runs With W&B

The Python package includes optional Weights & Biases logging for fitting runs.
Install the W&B dependency group with the training tools:

```bash
cd python
poetry install --with train,wandb
```

For online logging, authenticate once and enable W&B on a run:

```bash
poetry run wandb login
poetry run piano-fit train --config configs/local-pilot.yaml --wandb
```

You can also pass the API key through the environment, which is the usual path
for containers and GCP VMs:

```bash
export WANDB_API_KEY="..."
poetry run piano-fit train --config configs/gcp-cpu.yaml --wandb
```

For offline logging, keep the run local and sync later:

```bash
poetry run piano-fit train \
  --config configs/local-pilot.yaml \
  --wandb \
  --wandb-mode offline
```

Useful W&B-related environment variables:

- `WANDB_API_KEY`: API token for online logging.
- `WANDB_MODE=offline`: force local/offline logging.
- `WANDB_PROJECT`: override the project name if you do not want the config's
  default `piano-fit`.
- `WANDB_ENTITY`: set an account or team entity.

The config-driven command always writes a local JSONL metrics file when the
profile specifies one, so you still have local run telemetry if W&B is disabled
or offline.

## Docker And GCP

Build the fitting image from the repo root:

```bash
docker build -f docker/piano-fit.Dockerfile -t piano-fit:local .
docker run --rm piano-fit:local piano-fit --help
docker run --rm piano-fit:local piano-fit train --config python/configs/local-smoke.yaml --dry-run
```

GCP-oriented profiles are available at:

- `python/configs/gcp-cpu.yaml`
- `python/configs/gcp-gpu.yaml`

The GPU profile uses CUDA for EvoTorch tensor/search work only. C++ rendering and
loss evaluation remain CPU-bound in the current evaluator. See
[docker/README.md](docker/README.md) for container, W&B, dataset, and helper
script details.

## Continuous Integration

Current workflows cover:

- plugin and C++ test builds across supported platforms;
- performance checks;
- fitting pipeline smoke checks for the Python package, `PianoFit` build, model
  metadata, and config dry-runs.

The fitting smoke workflow intentionally avoids expensive optimization runs and
does not require private dataset bytes.

## Troubleshooting

- **Submodule headers or JUCE files are missing**: run
  `git submodule update --init --recursive`.
- **Fitting fails to read audio files**: hydrate `data/vintage-upright` from an
  approved dataset source, or mount a hydrated dataset directory and override the
  manifest path.
- **Docker build cannot connect to daemon**: start Docker Desktop or your Docker
  service, then retry the `docker build` command.
- **Poetry selects a different Python**: the package supports Python
  `>=3.11,<3.13`; install Python 3.11/3.12 and let Poetry use it.

## Contributing

Keep generated build outputs in `build*` directories and out of commits. Avoid
committing cloud credentials, local W&B runs, hydrated private dataset files, or
machine-specific paths. Prefer adding verification commands to the relevant
README when introducing new build or fitting workflows.

## License

This project is licensed under the GPL-3.0 License.

## Credits

Based on physical modeling research and digital waveguide synthesis techniques.
