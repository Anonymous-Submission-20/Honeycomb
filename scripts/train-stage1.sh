#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_PATH="${DATA_PATH:-/PATH-TO-HEXMEMORY-LMDB}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/stage1}"
MODEL_ROOT="${MODEL_ROOT:-data/Wan-AI/Wan2.2-TI2V-5B}"

mkdir -p "$OUTPUT_DIR"
python -c 'import json, sys; from pathlib import Path; (Path(sys.argv[1]) / "corpus_provenance.json").write_text(json.dumps({"data_path": sys.argv[2]}, indent=2) + "\n")' "$OUTPUT_DIR" "$DATA_PATH"

# keep batch size at 1 in both stages; larger batches can discard valid conditioning
# use gradient accumulation to increase the effective batch size
accelerate launch \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  scripts/train.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-config "$MODEL_ROOT/diffusion_pytorch_model*.safetensors" \
  --model-config "$MODEL_ROOT/Wan2.2_VAE.pth" \
  --model-config "$MODEL_ROOT/models_t5_umt5-xxl-enc-bf16.pth" \
  --tokenizer-path "$MODEL_ROOT/google/umt5-xxl" \
  --stage vace \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-steps 10000 \
  --lr-vace 1e-5 \
  --lr-schedule cosine \
  --lr-min-ratio 0 \
  --mixed-precision bf16 \
  --torch-dtype bf16 \
  --max-reference-frames 8 \
  --max-preceding-frames 2 \
  --preceding-cond-noise-max-timestep 0 \
  --drop-text-prompt 0.2 \
  --save-steps 1000 \
  --log-steps 10 \
  --seed 3407 \
  --num-workers 8 \
  --preceding-first
