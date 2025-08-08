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
    if not hasattr(env, "initial_handle_yaw"):
        print("ERROR! Initial handle yaw not set!")
        env.initial_handle_yaw = current_rot.clone()

    # compute reward for rotations
    # we use the initial step's rotation to compute the delta
    delta_rot = current_rot - env.initial_handle_yaw
    print(f"[INFO] Current rotation: {current_rot}, previous rotation: {env.initial_handle_yaw}, Delta rotation: {delta_rot}")
    # update the initial rotation for the next step
    env.initial_handle_yaw = current_rot.clone()

    return torch.abs(delta_rot)
    
def gripper_to_closest_handle_end_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Computes the distance from the gripper to the closest handle end."""
    # The observation function now gives us the positions relative to the gripper.
    # The gripper is at (0,0,0) in its own frame, so the distance is just the norm.
    handle_ends_flat = observations.handle_ends_positions_in_gripper_frame(env)
    handle_ends_pos = handle_ends_flat.view(env.num_envs, 4, 3)

    # Calculate the norm of each vector (which is the distance from the origin/gripper)
    distances = torch.norm(handle_ends_pos, dim=-1) # Shape: (num_envs, 4)

    # Find the minimum distance for each environment
    min_distance, _ = torch.min(distances, dim=-1) # Shape: (num_envs,)

    return min_distance
