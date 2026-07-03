# Vintage Upright Dataset

This directory stores the Vintage Upright sample data used by the Piano fitting
pipeline.

## Source

The raw `.ncw` files were imported from the local Vintage Upright library at:

```text
/Users/armaine/Downloads/Vintage Upright/Vintage Upright Samples
```

Converted `.wav` files are generated from those raw samples with the pinned
`third_party/ncw` converter.

## Permission And Redistribution

The operator confirmed they received verbal permission to store the Vintage
Upright audio files in this repository through Git LFS. The raw `.ncw` files
and converted `.wav` files are intended for this project's Git LFS storage.
Repository visibility and LFS remote privacy should remain consistent with that
permission before pushing or sharing this data.

## Git LFS

The committed audio payloads are stored through Git LFS:

```text
data/vintage-upright/raw/**/*.ncw
data/vintage-upright/wav/**/*.wav
```

Run `git lfs pull` after checkout if the audio files are still pointers.

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
