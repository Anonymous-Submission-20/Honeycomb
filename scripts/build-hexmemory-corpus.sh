#!/usr/bin/env bash
set -euo pipefail

HEX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HEX_ROOT"

SRC="${SRC:?set SRC to the prepped train/ dir}"
OUT="${OUT:?set OUT to the destination train/ dir}"
WRITER="${WRITER:?set WRITER to a writer checkpoint}"
DEVICE="${DEVICE:-cuda}"
RUN_PACK="${RUN_PACK:-1}"
FORCE="${FORCE:-0}"
EXCLUDE="${EXCLUDE:-}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
N=${#GPUS[@]}
LOGD="${LOGD:-$HEX_ROOT/runs/logs}"
mkdir -p "$LOGD" "$OUT"

EXTRA=(--writer-ckpt "$WRITER")
if [[ "$FORCE" == "1" ]]; then
  EXTRA+=(--force)
fi
if [[ -n "$EXCLUDE" ]]; then
  EXTRA+=(--exclude "$EXCLUDE")
fi

pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" PYTHONNOUSERSITE=1 \
    python -u adapter/build_corpus.py \
      --src "$SRC" --out "$OUT" --repo-root "$HEX_ROOT" \
      --device "$DEVICE" --shard "$i" --n-shards "$N" "${EXTRA[@]}" \
      > "$LOGD/buildcorpus_shard${i}of${N}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status != 0 )); then
  exit "$status"
fi

if [[ "$RUN_PACK" == "1" ]]; then
  python -m data_process.pack_to_lmdb --data-root "$OUT" \
    > "$LOGD/buildcorpus_pack.log" 2>&1
fi
