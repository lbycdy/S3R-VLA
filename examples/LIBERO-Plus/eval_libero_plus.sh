#!/bin/bash
# S3R-VLA - LIBERO-Plus evaluation: the seven perturbation dimensions (10,030 tasks, one episode each).
# One policy server per GPU; the dimensions are assigned to the servers round-robin, one simulator client per dimension.
#
# Before running, replace LIBERO-Plus/libero/libero/benchmark/__init__.py with
# examples/LIBERO-Plus/libero_plus_init.py (it selects the tasks of a perturbation dimension).
#
# Usage (from the repository root, with the S3R-VLA environment active):
#   CKPT=/path/to/<run_dir>/checkpoints/steps_120000_pytorch_model.pt \
#   LIBERO_HOME=/path/to/LIBERO-plus SIM_PYTHON=/path/to/libero_plus_env/bin/python \
#   bash examples/LIBERO-Plus/eval_libero_plus.sh
# Optional environment variables:
#   GPUS="0 1 2 3 4 5 6" GPUs of the policy servers (one server per GPU, about 6 GB each; a single GPU also works)
#   DIMENSIONS="Background_Textures Sensor_Noise"   dimensions to evaluate (default: all seven)
#   TASK_IDS="0,100"     only these tasks of each dimension (default: all)
#   SEED=7               environment seed; the paper reports the mean of SEED=3, 5, and 7
#   BASE_PORT=14082      SERVER_PYTHON=python (python of the S3R-VLA environment)
#   LIBERO_CONFIG_PATH   default ${LIBERO_HOME}/libero
#   MUJOCO_GL            default egl, rendered on the server's GPU
: "${CKPT:?set CKPT to the S3R-VLA checkpoint (.pt)}"
: "${LIBERO_HOME:?set LIBERO_HOME to your LIBERO-Plus code path}"
: "${SIM_PYTHON:?set SIM_PYTHON to the python executable of the LIBERO-Plus environment}"
GPUS=(${GPUS:-0 1 2 3 4 5 6})
DIMENSIONS=(${DIMENSIONS:-Background_Textures Camera_Viewpoints Language_Instructions Light_Conditions Objects_Layout Robot_Initial_States Sensor_Noise})
seed=${SEED:-7}
base_port=${BASE_PORT:-14082}
SERVER_PYTHON=${SERVER_PYTHON:-python}

export PYTHONDONTWRITEBYTECODE=1
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=$(pwd):${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}   # this repository (websocket tools) + LIBERO-Plus

folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
server_pids=(); client_pids=(); logs=()
task_args=(); [[ -n "${TASK_IDS:-}" ]] && task_args=(--args.task-ids "$TASK_IDS")
trap 'kill "${client_pids[@]}" "${server_pids[@]}" 2>/dev/null' EXIT

for j in "${!GPUS[@]}"; do
    # CUDA_VISIBLE_DEVICES isolates each server on its GPU (the model is first materialized on the current device)
    CUDA_VISIBLE_DEVICES=${GPUS[$j]} ${SERVER_PYTHON} ./deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT}" --port $((base_port + j)) --use_bf16 --cuda 0 &
    server_pids+=($!)
done

for i in "${!DIMENSIONS[@]}"; do
    dimension=${DIMENSIONS[$i]}
    j=$((i % ${#GPUS[@]}))
    out="results/plus_libero_mix/${dimension}/${folder_name}/seed${seed}"
    mkdir -p "$out"
    MUJOCO_EGL_DEVICE_ID=${GPUS[$j]} ${SIM_PYTHON} ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path "${CKPT}" \
        --args.host 127.0.0.1 \
        --args.port $((base_port + j)) \
        --args.task-suite-name libero_mix \
        --args.category_value "${dimension//_/ }" \
        --args.num-trials-per-task 1 \
        "${task_args[@]}" \
        --args.video-out-path "$out" \
        --args.seed "$seed" \
        --args.with_state true > "$out/eval.log" 2>&1 &
    client_pids+=($!); logs+=("$out/eval.log")
done

wait "${client_pids[@]}"
for log in "${logs[@]}"; do
    echo "$log: $(grep -ac 'Success: True' "$log") / $(grep -acE 'Success: (True|False)' "$log") episodes successful"
done
