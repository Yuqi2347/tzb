# Training

Set paths before launching training:

```bash
export MODEL_PATH=/path/to/FM9G4B-V
export TRAIN_DATA=/path/to/train.json
export OUTPUT_DIR=outputs/sft
```

Run supervised fine-tuning:

```bash
bash scripts/train/finetune_ds.sh
```

Run GRPO/RGRPO:

```bash
export PROMPTS_DATA=/path/to/prompts.json
export IMAGE_ROOT=/path/to/images
export OUTPUT_DIR=outputs/grpo
bash scripts/train/finetune_grpo.sh
```

Run PPO:

```bash
export PROMPTS_DATA=/path/to/prompts.json
export IMAGE_ROOT=/path/to/images
export OUTPUT_DIR=outputs/ppo
bash scripts/train/finetune_ppo.sh
```

RGRPO feedback uses an OpenAI-compatible API. Set `RGRPO_API_KEY`; override `RGRPO_API_BASE` in the script if needed.

LoRA merging:

```bash
python scripts/tools/merge_lora.py \
  --base-model /path/to/base-model \
  --checkpoint outputs/grpo/final \
  --output-dir outputs/merged_model
```
