#!/usr/bin/env bash
# Quick launcher for MuJoCo deployment with keyboard teleoperation

ROBOT="${1:-go2_piper}"
MODE="${2:-flat}"

CONDA_ENV_PYTHON="/home/amit/miniconda3/envs/robotis_lab_env/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1

CONFIG_FILE="config.yaml"
if [ "$MODE" == "wbc" ]; then
    CONFIG_FILE="config_wbc.yaml"
fi

DEPLOY_SCRIPT="mujoco/deploy/deploy_mujoco/${ROBOT}/${ROBOT}.py"

if [ ! -f "$DEPLOY_SCRIPT" ]; then
    echo "[ERROR]: Deployment script for robot '${ROBOT}' not found at ${DEPLOY_SCRIPT}"
    echo "Available robots: go2_piper, go2_arx5, b2_z1, go1_arx5, go1_wx250s, ago_z1"
    echo "Usage: ./scripts/run_mujoco_deploy.sh [robot] [flat|wbc]"
    exit 1
fi

echo "======================================================================"
echo "  Deploying Robot: ${ROBOT} (${MODE^^} Mode) in MuJoCo"
echo "======================================================================"
echo "  Python:  $CONDA_ENV_PYTHON"
echo "  Script:  $DEPLOY_SCRIPT"
echo "  Config:  $CONFIG_FILE"
echo "======================================================================"
echo "  Keyboard Controls:"
echo "    Locomotion:        W/S (Fwd/Back), A/D (Left/Right), Q/E (Turn)"
echo "    EE Translation:    I/K (Fwd/Back), J/L (Left/Right), U/O (Up/Down)"
echo "    EE Orientation:    1/2 (Roll), 3/4 (Pitch), 5/6 (Yaw)"
echo "    Reset Commands:    R"
echo "======================================================================"

cd "$(dirname "$DEPLOY_SCRIPT")" || exit 1
"$CONDA_ENV_PYTHON" "$(basename "$DEPLOY_SCRIPT")" "$CONFIG_FILE"
