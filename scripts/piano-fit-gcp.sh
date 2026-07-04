#!/usr/bin/env bash
set -euo pipefail

project=""
zone=""
machine_type="c3-standard-16"
image="piano-fit:latest"
config="python/configs/gcp-cpu.yaml"
execute=0

usage() {
  cat <<'USAGE'
Usage: scripts/piano-fit-gcp.sh --project PROJECT --zone ZONE [options]

Print a gcloud command for running the Piano Fit container on a GCP VM. The
default mode is --dry-run; pass --execute to run the printed command.

Options:
  --project PROJECT          GCP project ID.
  --zone ZONE                GCP zone, for example us-central1-a.
  --machine-type TYPE        VM machine type. Default: c3-standard-16.
  --image IMAGE              Container image. Default: piano-fit:latest.
  --config CONFIG            Training config. Default: python/configs/gcp-cpu.yaml.
  --dry-run                  Print the command without running it. Default.
  --execute                  Run the printed command.
  -h, --help                 Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      project="${2:-}"
      shift 2
      ;;
    --zone)
      zone="${2:-}"
      shift 2
      ;;
    --machine-type)
      machine_type="${2:-}"
      shift 2
      ;;
    --image)
      image="${2:-}"
      shift 2
      ;;
    --config)
      config="${2:-}"
      shift 2
      ;;
    --dry-run)
      execute=0
      shift
      ;;
    --execute)
      execute=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$project" || -z "$zone" ]]; then
  echo "--project and --zone are required" >&2
  usage >&2
  exit 2
fi

vm_name="piano-fit-${machine_type//[^a-zA-Z0-9-]/-}"
remote_command="docker run --rm -e WANDB_API_KEY -e WANDB_MODE -v \$PWD/build:/workspace/build ${image} piano-fit train --config ${config}"
cmd=(
  gcloud compute ssh "$vm_name"
  --project "$project"
  --zone "$zone"
  --command "$remote_command"
)

printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$execute" -eq 1 ]]; then
  exec "${cmd[@]}"
fi
