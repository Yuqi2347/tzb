# Benchmarks

This directory keeps evaluation adapters and dataset-card references. Full benchmark annotations, images, generated answers, and result files are intentionally not tracked.

Expected local assets:

```text
datasets/
  vrs/
  mme/
  valid/
outputs/eval/
```

Use `scripts/eval/eval.sh` with explicit `QUESTION_FILE`, `IMAGE_FOLDER`, and `ANSWERS_FILE` paths.
