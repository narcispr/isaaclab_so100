# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
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