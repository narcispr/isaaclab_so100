# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp import observations

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)



def object_lifted_distance(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2]


def ee_to_object_distance(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """
    Computes the Euclidean distance from the gripper (end-effector) to the object,
    both expressed in the robot's base frame.
    """
    # Get gripper position in robot base frame
    ee_pose = observations.gripper_position_in_robot_base(env)
    ee_pos = ee_pose[:, :3]  # Extract position (x, y, z)

    # Get object position in robot base frame
    object_pos = observations.object_position_in_robot_root_frame(env)

    # Compute Euclidean distance
    distance = torch.norm(ee_pos - object_pos, dim=-1)  # shape: (num_envs,)
    # return 1 - torch.tanh(distance / std)
    return -distance

def ee_close_to_object(
    env: ManagerBasedRLEnv, std: float, threshold: float = 0.05
) -> torch.Tensor:
    """
    Computes a reward based on the distance from the end-effector to the object.
    If the distance is below the threshold, it returns a positive reward.
    """
    distance = -ee_to_object_distance(env, std)
    return torch.where(distance < threshold, 1.0, 0.0)

# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward the agent for reaching the object using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     # Target object position: (num_envs, 3)
#     cube_pos_w = object.data.root_pos_w
#     # End-effector position: (num_envs, 3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]
#     # Distance of the end-effector to the object: (num_envs,)
#     object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

#     return 1 - torch.tanh(object_ee_distance / std)


# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     # rewarded if the object is lifted above the threshold
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# def object_ee_distance_and_lifted(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Combined reward for reaching the object AND lifting it."""
#     # Get reaching reward
#     reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
#     # Get lifting reward
#     lift_reward = object_is_lifted(env, minimal_height, object_cfg)
#     # Combine rewards multiplicatively
#     return reach_reward * lift_reward
