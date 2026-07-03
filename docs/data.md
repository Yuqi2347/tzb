# Data Preparation

This project expects external datasets and images to live outside Git. Use `datasets/` locally or pass absolute paths through environment variables.

## Training Schema

Training data follows a LLaVA-style conversation schema:

```json
{
  "id": "sample-0001",
  "image": "examples/images/sample.png",
  "conversations": [
    {"from": "human", "value": "<image>\n[caption] Describe the remote sensing image."},
    {"from": "gpt", "value": "A satellite image containing roads and buildings."}
  ]
}
```

See `examples/data/sample_train.json`.

## Evaluation Schema

Closed-set evaluation files use fields such as `Image`, `Text`, `Answer choices`, `Ground truth`, `Task`, `Subtask`, and `Question id`. See `examples/data/sample_eval.json`.

## External Assets

Recommended local layout:

```text
datasets/
  vrs/
    Images_train/
    Images_val/
  mme/
    images/
  valid/
    images/
models/
  FM9G4B-V/
outputs/
```

Do not commit full datasets, generated predictions, model checkpoints, or tokenizer exports.

## Benchmark Sources

- VRSBench: use the official dataset card and license terms.
- MME-RealWorld: use the official project or Hugging Face dataset release.
- Closed validation data: keep private unless you have explicit redistribution permission.
