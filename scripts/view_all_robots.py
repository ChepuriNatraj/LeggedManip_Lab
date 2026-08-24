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

"""Script to spawn and view all 7 Legged Manipulator robots side-by-side in Isaac Sim."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="View all 7 Legged Manipulator robots in Isaac Sim.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext

from LeggedManip_Lab.assets.ago_z1.ago_z1_articulation_cfg import AGO_Z1_CFG
from LeggedManip_Lab.assets.b1_z1.b1_z1_articulation_cfg import B1_Z1_CFG
from LeggedManip_Lab.assets.b2_z1.b2_z1_articulation_cfg import B2_Z1_CFG
from LeggedManip_Lab.assets.go1_arx5.go1_arx5_articulation_cfg import GO1_ARX5_CFG
from LeggedManip_Lab.assets.go1_wx250s.go1_wx250s_articulation_cfg import GO1_WX250S_CFG
from LeggedManip_Lab.assets.go2_arx5.go2_arx5_articulation_cfg import GO2_ARX5_CFG
from LeggedManip_Lab.assets.go2_piper.go2_piper_articulation_cfg import GO2_PIPER_CFG


def design_scene():
    """Design the scene with ground, lighting, and all 7 robot models."""
    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Dome light & distant light
    cfg_light = sim_utils.DomeLightCfg(
        intensity=2500.0,
        color=(0.95, 0.95, 1.0),
    )
    cfg_light.func("/World/Light", cfg_light)

    # List of all 7 robots with their spawn configs and offsets
    robots_data = [
        ("AGO-Z1", AGO_Z1_CFG, (0.0, -4.5, 0.55)),
        ("B1-Z1", B1_Z1_CFG, (0.0, -3.0, 0.55)),
        ("B2-Z1", B2_Z1_CFG, (0.0, -1.5, 0.55)),
        ("GO1-ARX5", GO1_ARX5_CFG, (0.0, 0.0, 0.55)),
        ("GO1-WX250S", GO1_WX250S_CFG, (0.0, 1.5, 0.55)),
        ("GO2-ARX5", GO2_ARX5_CFG, (0.0, 3.0, 0.55)),
        ("GO2-PIPER", GO2_PIPER_CFG, (0.0, 4.5, 0.35)),
    ]

    robot_articulations = []
    print("\n" + "=" * 70)
    print("  LeggedManip_Lab: Spawning all 7 Robot Platforms in Isaac Sim")
    print("=" * 70)

    for name, base_cfg, pos in robots_data:
        cfg = copy.deepcopy(base_cfg)
        prim_path = f"/World/Robots/{name.replace('-', '_')}"
        cfg.prim_path = prim_path
        cfg.init_state.pos = pos

        # Create articulation
        robot = Articulation(cfg)
        robot_articulations.append((name, robot, pos))
        print(f"  [+] Loaded {name:<12} at Position (X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.2f}) -> {prim_path}")

    print("=" * 70 + "\n")
    return robot_articulations


def main():
    """Main simulation loop."""
    sim_cfg = SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5.0, 0.0, 3.5], target=[0.0, 0.0, 0.5])

    # Design the scene
    robot_articulations = design_scene()

    # Reset simulation context
    sim.reset()

    print("[INFO]: Setting default joint positions for all robots...")
    for name, robot, _ in robot_articulations:
        default_joint_pos = robot.data.default_joint_pos.clone()
        default_joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        robot.reset()

    print("\n[INFO]: All 7 robots are active in Isaac Sim!")
    print("[INFO]: You can orbit (Left Click + Drag), pan (Middle Click + Drag), and zoom (Scroll).")
    print("[INFO]: Press Ctrl+C or close the window to exit.\n")

    step_count = 0
    while simulation_app.is_running():
        # Keep joint targets at default positions to hold pose
        for name, robot, _ in robot_articulations:
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.write_data_to_sim()

        # Step physics and rendering
        sim.step()

        step_count += 1
        if step_count % 500 == 0:
            print(f"[Sim Running] Step: {step_count} (Time: {step_count * sim_cfg.dt:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error]: {e}")
        raise e
    finally:
        simulation_app.close()
