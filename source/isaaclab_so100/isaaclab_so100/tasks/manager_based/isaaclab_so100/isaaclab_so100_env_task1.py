# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs import mdp 

from isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp.observations import object_position_in_robot_root_frame
from isaaclab_so100.tasks.manager_based.isaaclab_so100.so100_env_cfg import SO100SceneCfg

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    
    # Set actions for the specific robot type (SO100)
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        scale=0.5,
        use_default_offset=True
    )

    # Set gripper action with wider range for better visibility
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["Gripper"],
        open_command_expr={"Gripper": 0.5},  # Fully open
        close_command_expr={"Gripper": 0.0}  # fully closed
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        object_pos_rel = ObsTerm(func=object_position_in_robot_root_frame)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on startup
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )




# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     # (1) Constant running reward
#     alive = RewTerm(func=mdp.is_alive, weight=1.0)
#     # (2) Failure penalty
#     terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
#     # (3) Primary task: keep pole upright
#     pole_pos = RewTerm(
#         func=mdp.joint_pos_target_l2,
#         weight=-1.0,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
#     )
#     # (4) Shaping tasks: lower cart velocity
#     cart_vel = RewTerm(
#         func=mdp.joint_vel_l1,
#         weight=-0.01,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
#     )
#     # (5) Shaping tasks: lower pole angular velocity
#     pole_vel = RewTerm(
#         func=mdp.joint_vel_l1,
#         weight=-0.005,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
#     )


# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     # (1) Time out
#     time_out = DoneTerm(func=mdp.time_out, time_out=True)
#     # (2) Cart out of bounds
#     cart_out_of_bounds = DoneTerm(
#         func=mdp.joint_pos_out_of_manual_limit,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
#     )


@configclass
class SO100EnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = SO100SceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    """Main function."""
    # parse the arguments
    env_cfg = SO100EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            print("[Env 0]: Joint efforts: ", joint_efforts[0].tolist())
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0].tolist())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()