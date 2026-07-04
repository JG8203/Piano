# Piano Fit Container

This container is a clean Linux surface for building the `PianoFit` evaluator,
installing the Python `piano-fit` package, and running the config-driven fitting
pipeline on local or GCP machines.

## Build Locally

```bash
docker build -f docker/piano-fit.Dockerfile -t piano-fit:local .
docker run --rm piano-fit:local piano-fit --help
docker run --rm piano-fit:local piano-fit train --config python/configs/local-smoke.yaml --dry-run
```

To run a CPU smoke profile with outputs outside the container:

```bash
docker run --rm piano-fit:local \
  piano-fit train --config python/configs/local-smoke.yaml
```

## GCP CPU VM

Build or pull the image on the VM, then run the CPU profile:

```bash
docker run --rm \
  -e WANDB_MODE=offline \
  piano-fit:latest \
  piano-fit train --config python/configs/gcp-cpu.yaml
```

Use `WANDB_API_KEY` when online W&B logging is desired:

```bash
docker run --rm \
  -e WANDB_API_KEY \
  piano-fit:latest \
  piano-fit train --config python/configs/gcp-cpu.yaml --wandb
```

## GCP GPU VM

Install the NVIDIA container runtime on the VM, use a CUDA-capable host image,
and pass GPU devices through Docker:

```bash
docker run --rm --gpus all \
  -e WANDB_API_KEY \
  piano-fit:latest \
  piano-fit train --config python/configs/gcp-gpu.yaml --wandb
```

The GPU profile sets EvoTorch's tensor/search device to CUDA. The C++ rendering
and C++ loss evaluator remain CPU-bound unless a future evaluator implementation
changes that.

## Dataset Behavior

The Docker build preserves the repo-visible `data/vintage-upright` layout,
including Git LFS pointer files. It does not run `git lfs pull` during image
build because local checkouts and CI jobs may not have remote LFS credentials.

Before a full training run, either:

- run `git lfs pull` in a checkout that has access to the dataset and build the
  image from that checkout;
- run `git lfs pull` on the VM before starting the container and mount the
  hydrated `data/vintage-upright` tree into the same relative path used by the
  configs;
- mount an operator-managed dataset path, such as a GCS bucket mounted with
  Cloud Storage FUSE, and override `--manifest` to point at that mounted path.

No cloud credentials, project IDs, or bucket names are baked into the image.

## GCP Helper

The optional helper prints a `gcloud compute ssh` command by default:

```bash
scripts/piano-fit-gcp.sh \
  --project "$PROJECT" \
  --zone "$ZONE" \
  --machine-type c3-standard-16 \
  --image piano-fit:latest \
  --config python/configs/gcp-cpu.yaml
```
