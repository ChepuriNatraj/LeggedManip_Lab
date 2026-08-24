#!/usr/bin/env bash
# Script to launch Isaac Sim and view all 7 Legged Manipulator robots side-by-side

CONDA_ENV_PYTHON="/home/amit/miniconda3/envs/robotis_lab_env/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1

echo "======================================================================"
echo " Launching Isaac Sim: Viewing all 7 Legged Manipulator Robots"
echo "======================================================================"
echo "Python: $CONDA_ENV_PYTHON"
echo "Script: $SCRIPT_DIR/view_all_robots.py"
echo "======================================================================"

"$CONDA_ENV_PYTHON" "$SCRIPT_DIR/view_all_robots.py" "$@"
