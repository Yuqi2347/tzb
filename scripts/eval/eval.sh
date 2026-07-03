#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_NAME="${DATA_NAME:-valid}"
EVAL_TYPE="${EVAL_TYPE:-caption}"
QUESTION_FILE="${QUESTION_FILE:-examples/data/sample_eval.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-examples/images}"
ANSWERS_FILE="${ANSWERS_FILE:-outputs/eval/answers.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29503}"

mkdir -p "$(dirname "$ANSWERS_FILE")"

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" scripts/eval/eval_ddp.py \
  --question-file="$QUESTION_FILE" \
  --answers-file="$ANSWERS_FILE" \
  --eval-type="$EVAL_TYPE" \
  --image-folder="$IMAGE_FOLDER" \
  --data-name="$DATA_NAME"
