# Vintage Upright Dataset

This directory stores the Vintage Upright metadata and sample placeholders used
by the Piano fitting pipeline.

## Source

Raw `.ncw` files come from an approved Vintage Upright sample source. Converted
`.wav` files are generated from those raw samples with the pinned
`third_party/ncw` converter. Keep the raw and converted files at the same
relative paths shown in `manifest.jsonl`.

## Permission And Redistribution

The operator confirmed they received verbal permission to use the Vintage
Upright audio files for this project. Do not publish or upload hydrated audio
payloads unless repository visibility and distribution remain consistent with
that permission.

## Audio Payloads

The public fork keeps clone-safe placeholder files at:

```text
data/vintage-upright/raw/**/*.ncw
data/vintage-upright/wav/**/*.wav
```

Hydrate those paths from an approved dataset source before running full fitting
jobs. Plugin builds, Python tests, and fitting dry-runs do not require hydrated
audio payloads.

## Verify Checksums

```bash
python3 tools/vintage_upright/verify_vintage_upright.py data/vintage-upright/checksums.json
```

## Regenerate WAVs

Initialize the pinned converter source, then rebuild the prepared WAVs and
manifest from the raw NCWs:

```bash
git submodule update --init --recursive third_party/ncw

tools/vintage_upright/prepare_vintage_upright.py \
  data/vintage-upright \
  --raw-dir data/vintage-upright/raw \
  --output data/vintage-upright \
  --wav-dir data/vintage-upright/wav \
  --manifest data/vintage-upright/manifest.jsonl \
  --jobs 8 \
  --overwrite
```
