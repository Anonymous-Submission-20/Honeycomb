#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_PATH="${DATA_PATH:-/PATH-TO-HEXMEMORY-LMDB}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/stage2}"
MODEL_ROOT="${MODEL_ROOT:-data/Wan-AI/Wan2.2-TI2V-5B}"
VACE_CKPT="${VACE_CKPT:-runs/stage1/vace/step_0010000_vace.safetensors}"

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
  --init-vace-checkpoint "$VACE_CKPT" \
  --stage lora \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-steps 5000 \
  --lr-lora 1e-4 \
  --lr-schedule cosine \
  --lr-min-ratio 0 \
  --lora-rank 64 \
  --lora-alpha 64 \
  --lora-target-modules "q,k,v,o,ffn.0,ffn.2" \
  --mixed-precision bf16 \
  --torch-dtype bf16 \
  --max-reference-frames 8 \
  --max-preceding-frames 2 \
  --preceding-cond-noise-max-timestep 50 \
  --drop-text-prompt 0.2 \
  --save-steps 1000 \
  --log-steps 10 \
  --seed 3407 \
  --num-workers 8 \
  --preceding-first
