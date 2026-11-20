#!/usr/bin/env bash
# S3R-VLA training launcher. All settings are in the config; the version is selected by framework.name.
#
# Usage (from any directory):
#   bash scripts/train.sh v3                     # scripts/configs/s3r_vla_v3_libero_ft.yaml   (also v1, v2)
#   bash scripts/train.sh path/to/config.yaml    # any config (may inherit another one with `_base_:`)
#   SMOKE=1 bash scripts/train.sh v3             # 30-step smoke test, output in checkpoints/<run_id>_smoke
# Further arguments override config entries, e.g.
#   bash scripts/train.sh v3 --trainer.save_interval 5000 --framework.skill_model.wm_weight 0
# Environment variables:
#   NPROC                 number of processes (= GPUs), default: num_gpus of the config
#   CUDA_VISIBLE_DEVICES  default 0..NPROC-1
#   BS                    per-GPU batch size, default: the config; the global batch is NPROC x BS
#   ACCELERATE            accelerate executable, default: accelerate on PATH
#   PYTHON                python of the same environment, default: the python next to ACCELERATE
#   ACCEL_CONFIG          accelerate / DeepSpeed config, default ./starVLA/config/deepseeds/deepspeed_zero2.yaml
#   NCCL_SOCKET_IFNAME    network interface used by NCCL (multi-node); by default NCCL picks one
set -o pipefail
case "${1:-}" in
  v1|v2|v3) CONFIG="$(dirname "${BASH_SOURCE[0]}")/configs/s3r_vla_$1_libero_ft.yaml" ;;
  *.yaml|*.yml) CONFIG="$1" ;;
  *) echo "usage: bash scripts/train.sh <v1|v2|v3|config.yaml> [--key value ...]" >&2; exit 1 ;;
esac
shift
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd)"

ACCELERATE="${ACCELERATE:-accelerate}"
if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$(dirname "$(command -v "$ACCELERATE" || echo .)")/python"
  [[ -x "$PYTHON" ]] || PYTHON=python
fi

# make this repository take precedence over other editable installs of starVLA
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export NCCL_IB_DISABLE=1
[[ -n "${NCCL_SOCKET_IFNAME:-}" ]] && export NCCL_SOCKET_IFNAME
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export TMPDIR="${TMPDIR:-/tmp}"
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

read -r RUN_ID CFG_GPUS CKPT < <("$PYTHON" -m starVLA.training.config_loader "$CONFIG" \
                                  run_id num_gpus trainer.pretrained_checkpoint) \
  || { echo "ERROR: cannot read $CONFIG with $PYTHON" >&2; exit 1; }
NPROC="${NPROC:-${CFG_GPUS:-1}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC - 1)))}"
[[ -z "$CKPT" || -s "$CKPT" ]] || { echo "ERROR: pretrained checkpoint not found: $CKPT (see README, Pretrained Models)" >&2; exit 1; }

EXTRA_ARGS=()
[[ -n "${BS:-}" ]] && EXTRA_ARGS+=(--datasets.vla_data.per_device_batch_size "$BS")
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_ID="${RUN_ID}_smoke"
  EXTRA_ARGS+=(--trainer.max_train_steps 30 --trainer.num_warmup_steps 5 --trainer.save_interval 1000000
               --run_id "$RUN_ID" --output_dir "checkpoints/$RUN_ID"
               --framework.skill_model.ce_warmup_forwards 10 --framework.skill_model.revive_every 5)
fi

mkdir -p logs
LOG_FILE="logs/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%F %T')] config=$CONFIG nproc=$NPROC gpus=$CUDA_VISIBLE_DEVICES log=$LOG_FILE"

"$ACCELERATE" launch \
  --config_file "${ACCEL_CONFIG:-./starVLA/config/deepseeds/deepspeed_zero2.yaml}" \
  --num_processes "$NPROC" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  2>&1 | tee "$LOG_FILE"
