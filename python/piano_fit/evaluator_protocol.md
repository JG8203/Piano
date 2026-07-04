# PianoFit JSONL Stdio Evaluator Protocol

`PianoFit --serve-jsonl` keeps one evaluator process alive and exchanges one
JSON object per line over stdio. Python sends requests on stdin. PianoFit writes
responses on stdout.

Stdout in `--serve-jsonl` mode must contain JSONL protocol responses only.
Diagnostics, progress, target loading messages, and model information must go
to stderr.

## Evaluate

```json
{"type":"evaluate","batch_id":"000001","genomes":[{"id":"0","genome":[0.5,0.5]}]}
```

`batch_id` is an opaque string echoed in the response. `genomes` is an array of
candidate objects with a stable string `id` and a numeric `genome` array.

```json
{"type":"result","batch_id":"000001","losses":[{"id":"0","loss":14.6597}]}
```

Losses are returned in the same candidate id space as the request.

## Shutdown

```json
{"type":"shutdown"}
```

The evaluator exits cleanly after a shutdown request.

## Error

```json
{"type":"error","batch_id":"000001","message":"..."}
```

Errors include `batch_id` when the evaluator can parse one from the request.
