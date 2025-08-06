# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp import rewards

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_close_to_object_termination(
    env: ManagerBasedRLEnv, std: float, threshold: float = 0.02
) -> torch.Tensor:
    """
    Computes a reward based on the distance from the end-effector to the object.
    If the distance is below the threshold, it returns a positive reward.
    """
    distance = -rewards.ee_to_object_distance(env, std)
    # print(f"If distance {distance} is below threshold {threshold}, then return True")
    return torch.where(distance < threshold, True, False)


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, True, False)

def valve_rotated_past_threshold(
    env: ManagerBasedRLEnv,
    angle_threshold: float = 0.785,  # 45 degrees in radians
    handle_frame_cfg: SceneEntityCfg = SceneEntityCfg("handle_frame"),
) -> torch.Tensor:
    """Terminate when the valve has been rotated past a certain threshold from its initial position."""
    handle_frame: FrameTransformer = env.scene[handle_frame_cfg.name]
    quat = handle_frame.data.target_quat_source[:, 0, :]

    # Convert quaternion to yaw angle
    z = quat[:, 2]
    w = quat[:, 3]
    current_yaw = 2.0 * torch.atan2(z, w)

    # Initialize initial_yaw at the first step if it doesn't exist
    if not hasattr(env, "initial_handle_yaw"):
        env.initial_handle_yaw = current_yaw.clone()

    # Calculate the absolute difference in yaw from the initial position
    delta_yaw = torch.abs(current_yaw - env.initial_handle_yaw)

    # Terminate if the rotation exceeds the threshold
    return delta_yaw > angle_threshold