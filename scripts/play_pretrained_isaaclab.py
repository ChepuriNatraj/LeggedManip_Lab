# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to deploy and play pre-trained policies directly inside NVIDIA Isaac Sim."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play pre-trained policy in Isaac Sim.")
parser.add_argument("--task", type=str, default="GO2-PIPER-Flat", help="Task name (e.g. GO2-PIPER-Flat, GO2-PIPER-WBC, B2-Z1-Flat)")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments in Isaac Sim")
parser.add_argument("--policy", type=str, default=None, help="Path to policy.pt (auto-resolved if None)")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import os
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import LeggedManip_Lab.tasks  # noqa: F401

TASK_TO_POLICY = {
    "GO2-PIPER-Flat": "mujoco/deploy/policy/go2_piper/policy.pt",
    "GO2-PIPER-WBC": "mujoco/deploy/policy/go2_piper/wbc/policy.pt",
    "GO2-ARX5-Flat": "mujoco/deploy/policy/go2_arx5/policy.pt",
    "B2-Z1-Flat": "mujoco/deploy/policy/b2_z1/policy.pt",
    "B2-Z1-WBC": "mujoco/deploy/policy/b2_z1/wbc/policy.pt",
    "GO1-ARX5-Flat": "mujoco/deploy/policy/go1_arx5/policy.pt",
    "GO1-ARX5-WBC": "mujoco/deploy/policy/go1_arx5/wbc/policy.pt",
    "AGO-Z1-Flat": "mujoco/deploy/policy/ago_z1/policy.pt",
    "AGO-Z1-WBC": "mujoco/deploy/policy/ago_z1/wbc/policy.pt",
    "GO1-WX250S-Flat": "mujoco/deploy/policy/go1_wx250s/policy.pt",
}


def main():
    """Run pre-trained policy in Isaac Sim."""
    root_dir = Path(__file__).resolve().parent.parent

    policy_path = args_cli.policy
    if policy_path is None:
        rel_path = TASK_TO_POLICY.get(args_cli.task, "mujoco/deploy/policy/go2_piper/policy.pt")
        policy_path = str(root_dir / rel_path)

    print("\n" + "=" * 70)
    print(f"  Isaac Sim Deployment: Task [{args_cli.task}] with {args_cli.num_envs} Parallel Envs")
    print(f"  Loading Policy: {policy_path}")
    print("=" * 70 + "\n")

    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    # Load TorchScript Policy
    policy = torch.jit.load(policy_path, map_location=args_cli.device)
    policy.eval()

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Environment created! Device: {env.unwrapped.device}")
    obs_dict, _ = env.reset()

    step_count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            policy_obs = obs_dict["policy"]
            actions = policy(policy_obs)
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

            step_count += 1
            if step_count % 250 == 0:
                mean_rew = rewards.mean().item()
                print(f"[Sim Running] Step: {step_count} (Avg Step Reward: {mean_rew:.3f})")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR]: {e}")
        raise e
    finally:
        simulation_app.close()
