# Remote Sensing Optimization Based on the Jiuge MLLM

[中文](README.md)

This repository is the open-source showcase version of a 19th Challenge Cup "AI+" project for remote-sensing image analysis and semantic understanding. It organizes the project around domain adaptation, reasoning enhancement, and engineering cleanup for the Jiuge multimodal large language model.

The goal of this repository is not to provide a full one-click reproduction with private data and checkpoints. Instead, it presents a practical technical pipeline for adapting a general multimodal model to remote sensing: high-quality data filtering, LoRA-based parameter-efficient tuning, Chain-of-Thought reasoning, GRPO/RGRPO optimization, and hierarchical visual focusing for large-scale remote-sensing images.

## Results

| Task / Metric | Baseline | Ours | Improvement |
| --- | ---: | ---: | ---: |
| VRSBench VQA Overall | 49.22% | 66.11% | +16.89 pp |
| VRSBench Caption BLEU-1 | 0.24 | 0.44 | +0.20 |
| VRSBench Caption BLEU-4 | 0.03 | 0.11 | +0.08 |
| VRSBench Refer Acc@0.5 | 5.48% | 13.31% | +7.83 pp |
| VRSBench Refer Acc@0.7 | 0.74% | 3.73% | +2.99 pp |
| VRSBench Refer mean IoU | 12.14% | 18.66% | +6.52 pp |
| MME-RealWorld-RS EN | 14.93% | 38.90% | about +160% |
| MME-RealWorld-RS CN | 13.67% | 40.00% | about +192% |

The ablation study shows that LoRA and CoT/RL are complementary. VRSBench VQA Overall improves from 49.22 for the baseline to 58.65 with LoRA, 57.77 with CoT/RL, and 65.41 with the combined training strategy.

## Method

### 1. Remote-Sensing Data Construction

Remote-sensing image-text data often contains noisy annotations, inconsistent image-text alignment, insufficient description granularity, and unstable spatial relations. The project uses RCRS, a Rule-Guided Consistency Rejection Sampling mechanism:

- Rule-based filtering removes empty text, broken images, invalid formats, and low-quality labels.
- Teacher-model evaluation uses Qwen2.5-VL to judge multimodal consistency.
- Rejection sampling keeps samples with strong semantic consistency or sufficient similarity, improving the signal-to-noise ratio of the training set.

The training data covers VQA, Caption, and Refer tasks, with sources including EarthVQA, RSVQA-HR, NWPU-Captions, and RefDrone.

### 2. Three-Stage Model Optimization

- LoRA parameter-efficient tuning injects remote-sensing domain knowledge while keeping the backbone model mostly frozen.
- Chain-of-Thought reasoning guides the model to solve complex visual QA, spatial reasoning, and scene understanding tasks through explicit intermediate reasoning.
- GRPO/RGRPO optimization designs task-specific reward signals for VQA, Caption, and Refer, jointly considering answer correctness, semantic consistency, and reasoning-chain completeness.

### 3. Large-Scale Remote-Sensing Inference

Remote-sensing images combine wide-area context with small fine-grained targets. The project introduces hierarchical visual focusing: a multi-resolution image pyramid first builds global context, then selected regions are refined at higher resolution. The nine-grid slicing and local re-reasoning strategy helps balance global layout, local objects, and spatial relations.

## Evaluation

The project evaluates three core capabilities:

- VQA: remote-sensing visual question answering, measured by Accuracy across category, quantity, color, position, direction, scene, and reasoning subtasks.
- Caption: remote-sensing image captioning, measured by BLEU, METEOR, ROUGE-L, CIDEr, and related generation metrics.
- Refer: referring-expression grounding, measured by Acc@0.5, Acc@0.7, mean IoU, and cumulative IoU.

Experiments were conducted on 8 x NVIDIA V100 32GB GPUs with torchrun, DeepSpeed ZeRO-2, BF16 mixed precision, gradient checkpointing, and TensorBoard monitoring. The technical report records approximate training costs of 120K images / 4 hours for LoRA, 90K samples / 4 hours for CoT training, and 20K images / about 10 hours for reinforcement learning.

## Repository Contents

The repository has been reorganized for open-source presentation. It keeps source code, script entrypoints, configuration templates, documentation, and minimal examples. Model weights, full datasets, generated outputs, private paths, and API keys are excluded.

```text
src/remote_sensing_mllm/      Core dataset, trainer, reward, conversation, and model code
scripts/train/                SFT, LoRA, GRPO, and PPO training entrypoints
scripts/eval/                 VQA, Caption, Refer, and efficiency evaluation entrypoints
scripts/tools/                Data processing, LoRA merging, annotation, and maintenance tools
benchmarks/                   VRSBench, MME-RealWorld, and benchmark adapters
configs/                      DeepSpeed and environment templates
demo/                         FastAPI backend and browser frontend demo
docs/                         Data, training, evaluation, and demo guides
examples/data/                Minimal JSON examples for training and evaluation
```

## Quick Look

Install dependencies:

```bash
conda env create -f configs/environment.yml
conda activate tzb-remote-sensing-mllm
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set local model, data, and image paths:

```bash
export MODEL_PATH=/path/to/FM9G4B-V
export TRAIN_DATA=/path/to/train.json
export IMAGE_ROOT=/path/to/images
```

Training, evaluation, and demo entrypoints:

```bash
bash scripts/train/finetune_ds.sh
QUESTION_FILE=/path/to/eval.json IMAGE_FOLDER=/path/to/images bash scripts/eval/eval.sh
MODEL_PATH=/path/to/FM9G4B-V python -m uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000
```

## Data and Weights

Full datasets, generated answers, model checkpoints, tokenizer exports, and benchmark outputs are not tracked in Git. See [docs/data.md](docs/data.md) for expected layouts and dataset notes.

The JSON files in [examples/data](examples/data) only document the expected schemas; they are not sufficient for real training.

## Documentation

- [Data preparation](docs/data.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Demo](docs/demo.md)
- [Security](SECURITY.md)

## Open-Source Notes

This repository is a showcase and engineering review version. It keeps only public source code, configuration templates, documentation, and minimal examples. Plaintext secrets, local absolute paths, large datasets, checkpoints, and full benchmark outputs are not included.

Code is released under the license in [LICENSE](LICENSE). Third-party datasets and pretrained models keep their original licenses.
