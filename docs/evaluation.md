# Evaluation

The default evaluation wrapper is configurable through environment variables:

```bash
export MODEL_PATH=/path/to/model
export DATA_NAME=valid
export EVAL_TYPE=caption
export QUESTION_FILE=/path/to/eval.json
export IMAGE_FOLDER=/path/to/images
export ANSWERS_FILE=outputs/eval/answers.json
bash scripts/eval/eval.sh
```

For VRSBench, MME-RealWorld, and closed-set validation, keep the full JSON files and image folders outside Git and pass their paths explicitly.

Efficiency evaluation:

```bash
MODEL_PATH=/path/to/model \
QUESTION_FILE=/path/to/eval.json \
IMAGE_FOLDER=/path/to/images \
bash scripts/eval/run_efficiency_test.sh
```

Some caption metrics require optional COCO caption dependencies and Java for SPICE. Keep those third-party repositories outside this repo.
