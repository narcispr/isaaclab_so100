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


def ee_to_object_distance(env: ManagerBasedRLEnv, std: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """
    Computes the Euclidean distance from the gripper (end-effector) to the object,
    both expressed in the robot's base frame.
    """
    # Get gripper position in robot base frame
    ee_pose = observations.gripper_position_in_robot_base(env)
    ee_pos = ee_pose[:, :3]  # Extract position (x, y, z)

    # Get object position in robot base frame
    object_pos = observations.object_position_in_robot_root_frame(env, object_cfg=object_cfg)

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

def handle_rotation(
    env: ManagerBasedRLEnv,
    handle_frame_cfg: SceneEntityCfg = SceneEntityCfg("handle_frame"),
) -> torch.Tensor:
    """Computes the change in rotation of the handle since the last step.

    This reward is designed to encourage the agent to continuously rotate the valve
    in a positive direction. It tracks the change in the rotation angle from the previous
    step, rewarding positive changes and penalizing negative changes.
    """
    handle_frame: FrameTransformer = env.scene[handle_frame_cfg.name]
    quat = handle_frame.data.target_quat_source[:, 0, :]  # (num_envs, 4) --> Quaternion

    # Convert quaternion to rotation angle
    z = quat[:, 2]
    w = quat[:, 3]

    current_rot = 2.0 * torch.atan2(z, w)  # shape: (num_envs,)

    # initialize previous rotation at the first step
    if not hasattr(env, "previous_handle_rot"):
        env.previous_handle_rot = current_rot.clone()

    # compute reward for rotations
    # we use the previous step's rotation to compute the delta
    delta_rot = current_rot - env.previous_handle_rot

    # for envs that have just been reset, the reward is 0
    # because there is no previous step in the same episode
    delta_rot[env.reset_buf] = 0.0

    # update previous rotation for the next step
    env.previous_handle_rot = current_rot.clone()

    return delta_rot
    
def gripper_to_closest_handle_end_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Computes the distance from the gripper to the closest handle end."""
    # Get gripper position in robot base frame (x, y, z)
    gripper_pose = observations.gripper_position_in_robot_base(env)
    gripper_pos = gripper_pose[:, :3]  # Shape: (num_envs, 3)

    # Get handle ends positions in robot base frame
    # This returns a flat tensor of shape (num_envs, 12)
    handle_ends_flat = observations.handle_ends_positions_in_robot_base(env)
    # Reshape to (num_envs, 4, 3) to work with the 4 points
    handle_ends_pos = handle_ends_flat.view(env.num_envs, 4, 3)

    # Calculate the distance from the gripper to each of the 4 handle ends
    # gripper_pos needs to be unsqueezed to be broadcastable with handle_ends_pos
    # gripper_pos shape: (num_envs, 3) -> (num_envs, 1, 3)
    # handle_ends_pos shape: (num_envs, 4, 3)
    # The subtraction will be broadcasted, resulting shape: (num_envs, 4, 3)
    distances = torch.norm(handle_ends_pos - gripper_pos.unsqueeze(1), dim=-1) # Shape: (num_envs, 4)

    # Find the minimum distance for each environment
    min_distance, _ = torch.min(distances, dim=-1) # Shape: (num_envs,)

    return min_distance

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