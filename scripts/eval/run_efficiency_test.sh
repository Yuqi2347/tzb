#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/FM9G4B-V}"
IMAGE_FOLDER="${IMAGE_FOLDER:-examples/images}"
QUESTION_FILE="${QUESTION_FILE:-examples/data/sample_eval.json}"
ANSWERS_FILE="${ANSWERS_FILE:-outputs/eval/efficiency_results.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model path does not exist: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/eval/eval_efficiency.py \
    --model-path "$MODEL_PATH" \
    --conv-mode "vicuna_v1" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --data-name "valid" \
    --answers-file "$ANSWERS_FILE" \
    --temperature 0.2 \
    --max_new_tokens 64 \
    --eval-type "caption" \
    --no-merge \
    --output-dir "$OUTPUT_DIR"
