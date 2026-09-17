# Data preparation

Run from the submission root with the `honeycomb` environment activated and `PYTHONPATH` set as described in the [main README](../README.md). Raw-video preparation also needs ViPE, FFmpeg, OpenEXR, LiteLLM, and a Qwen3-VL captioning service.

Replace `/PATH-TO-RAW-VIDEOS` with your video directory and `/PATH-TO-VIPE/bin/vipe` with your ViPE executable.

## 1. Configure the dataset

```bash
export HONEYCOMB_VIDEO_DIRS="/PATH-TO-RAW-VIDEOS"
export HONEYCOMB_OUTPUT_ROOT="data/re10k/train"
export HONEYCOMB_MAX_VIDEOS=none
export HONEYCOMB_REF_CANDIDATE_SCOPE=past_only
export HONEYCOMB_APPLY_DYNAMIC_MASK=0
export HONEYCOMB_EPS_IOU=0.04
```

Repeat for SpatialVID with its video path, output `data/spatialvid/train`, and `HONEYCOMB_EPS_IOU=0.01`.

## 2. Collect clips and prepare geometry

```bash
python -m data_process.run_video_collect
python -m data_process.run_vipe_geometry \
  --input-root "$HONEYCOMB_OUTPUT_ROOT" \
  --vipe-executable /PATH-TO-VIPE/bin/vipe
python -m data_process.run_pipeline
```

## 3. Generate captions and encode latents

Start your Qwen3-VL captioning service first. Replace `http://localhost:8000/v1` and `openai/qwen3vl` with its address and served model name. Set `DASHSCOPE_API_KEY` if the service requires authentication.

```bash
python -m data_process.run_video_captioning \
  --input-root "$HONEYCOMB_OUTPUT_ROOT" \
  --video-keys clip,train_target_rgb \
  --qwen-model-path openai/qwen3vl \
  --api-base http://localhost:8000/v1 \
  --skip-existing

python -m data_process.run_video_vae_encode \
  --input-root "$HONEYCOMB_OUTPUT_ROOT" \
  --video-keys clip,train_preceding_rgb,train_target_rgb,train_reference_rgb \
  --skip-existing
```

The encoder expects Wan2.2 weights under `data/Wan-AI/Wan2.2-TI2V-5B/`.

## 4. Build the training corpus

Continue with rollout-pack preparation, writer training, and HexMemory corpus building in the [main README](../README.md#1-prepare-clips-and-writer-packs).
