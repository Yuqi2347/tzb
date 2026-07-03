#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "Set DASHSCOPE_API_KEY before running annotation."
    exit 1
fi

DATASET_ROOT="${DATASET_ROOT:-datasets/hrscd}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/annotations/hrscd.json}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

python scripts/tools/annotate_hrscd.py \
  --dataset-root "$DATASET_ROOT" \
  --output-file "$OUTPUT_FILE"
