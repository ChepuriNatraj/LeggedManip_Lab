#!/usr/bin/env bash
# Quick launcher to run pre-trained RL policies in NVIDIA Isaac Sim

TASK="${1:-GO2-PIPER-Flat}"
NUM_ENVS="${2:-4}"

CONDA_ENV_PYTHON="/home/amit/miniconda3/envs/robotis_lab_env/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1

echo "======================================================================"
echo " Launching Policy in NVIDIA Isaac Sim"
echo "======================================================================"
echo " Task:      $TASK"
echo " Envs:      $NUM_ENVS"
echo " Python:    $CONDA_ENV_PYTHON"
echo " Script:    $SCRIPT_DIR/play_pretrained_isaaclab.py"
echo "======================================================================"

"$CONDA_ENV_PYTHON" "$SCRIPT_DIR/play_pretrained_isaaclab.py" --task "$TASK" --num_envs "$NUM_ENVS" "${@:3}"
