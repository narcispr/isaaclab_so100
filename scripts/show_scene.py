# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import torch
from isaaclab_so100.tasks.manager_based.isaaclab_so100.so100_valve_scene_cfg import SO100ValveSceneCfg  # isort: skip

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
         # Perform step
        sim.step()
        time.sleep(sim_dt)
        # Update buffers
        scene.update(sim_dt)
        # quat = scene["handle_frame"].data.target_quat_source
        
        # # Convert quaternion to Euler angles and print
        # euler_angles = torch.atan2(
        #     2 * (quat[..., 0] * quat[..., 1] + quat[..., 2] * quat[..., 3]),
        #     1 - 2 * (quat[..., 1]**2 + quat[..., 2]**2)
        # )
        # print("Euler Angles:", euler_angles)

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([0, -2.0, 3.2], [0.0, -0.1, 0.25])
    # design scene
    scene_cfg = SO100ValveSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()


