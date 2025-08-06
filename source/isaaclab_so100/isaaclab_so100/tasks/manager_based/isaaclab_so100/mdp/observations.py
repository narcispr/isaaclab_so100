# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply
from isaaclab.sensors.frame_transformer import FrameTransformer
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),) -> torch.Tensor:
    
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # computes the position of the object relative to the robot’s base, by 
    # subtracting the robot’s position/orientation from the object’s world position.
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7], 
        object.data.root_pos_w[:, :3]
    )
    return object_pos_b


def gripper_position_in_robot_base(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    """
    Returns the position of the gripper relative to the robot's base frame.
    """
    frame_transformer: FrameTransformer = env.scene["ee_frame"]

    p = frame_transformer.data.target_pos_source[:, 1, :]  # (num_envs, 3) --> Index 1 is the gripper frame
    quat = frame_transformer.data.target_quat_source[:, 1, :]

    return torch.cat([p, quat], dim=-1)  # (num_envs, 7) --> Position + Quaternion 

def handle_ends_positions_in_robot_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The positions of the four handle ends, transformed into the robot's base frame."""
    # Get the pose of the handle's center in the world frame
    handle_frame: FrameTransformer = env.scene["handle_frame"]
    handle_pos_w = handle_frame.data.target_pos_w[:, 0, :]
    handle_quat_w = handle_frame.data.target_quat_w[:, 0, :]

    # Define the local offsets from the handle's center based on the URDF
    # The handle is 0.20m long, so the ends are at +/- 0.10m
    device = env.device
    offsets = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, -0.1, 0.0],
        ],
        device=device,
    )
    # We need to replicate the offsets for each environment
    num_envs = env.num_envs
    offsets = offsets.repeat(num_envs, 1, 1)  # Shape: (num_envs, 4, 3)

    # Rotate the local offsets by the handle's orientation
    rotated_offsets = quat_apply(handle_quat_w.unsqueeze(1), offsets)

    # Add the rotated offsets to the handle's world position
    # to get the world positions of the ends
    ends_pos_w = handle_pos_w.unsqueeze(1) + rotated_offsets

    # Transform the world positions of the ends into the robot's base frame
    robot: RigidObject = env.scene["robot"]
    ends_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3].unsqueeze(1),
        robot.data.root_state_w[:, 3:7].unsqueeze(1),
        ends_pos_w,
    )

    # Reshape to a flat tensor for the observation
    return ends_pos_b.reshape(num_envs, -1) 