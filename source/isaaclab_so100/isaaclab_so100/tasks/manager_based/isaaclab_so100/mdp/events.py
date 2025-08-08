# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform
from isaaclab.assets.articulation import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def euler_angles_to_quat(euler_angles: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """
    Convert euler angles to quaternions (x, y, z, w).
    """
    # get half angles
    h_roll = euler_angles[..., 0] / 2
    h_pitch = euler_angles[..., 1] / 2
    h_yaw = euler_angles[..., 2] / 2
    # compute sin and cos
    c_r, s_r = torch.cos(h_roll), torch.sin(h_roll)
    c_p, s_p = torch.cos(h_pitch), torch.sin(h_pitch)
    c_y, s_y = torch.cos(h_yaw), torch.sin(h_yaw)
    # compute quaternion
    quat = torch.zeros((euler_angles.shape[0], 4), device=euler_angles.device)
    if convention == "XYZ":
        quat[:, 0] = s_r * c_p * c_y - c_r * s_p * s_y  # x
        quat[:, 1] = c_r * s_p * c_y + s_r * c_p * s_y  # y
        quat[:, 2] = c_r * c_p * s_y - s_r * s_p * c_y  # z
        quat[:, 3] = c_r * c_p * c_y + s_r * s_p * s_y  # w
    else:
        raise NotImplementedError(f"Convention {convention} is not supported.")
    return quat


def reset_initial_valve_angle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    # handle_frame_cfg: SceneEntityCfg,
):
    
    # print("RESET!")
    asset: Articulation = env.scene['valve']
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.initial_handle_yaw = torch.zeros_like(joint_pos[:, 0])  # Initialize the initial yaw to zero for the reset environments

    """Stores the initial yaw of the valve handle at the start of the episode."""
    # handle_frame: FrameTransformer = env.scene[handle_frame_cfg.name]
    # quat = handle_frame.data.target_quat_source[:, 0, :]

    # # Convert quaternion to yaw angle
    # z = quat[:, 2]
    # w = quat[:, 3]
    # current_yaw = 2.0 * torch.atan2(z, w)

    # Initialize initial_yaw at the first step
    # if not hasattr(env, "initial_handle_yaw"):
    #     env.initial_handle_yaw = torch.zeros_like(current_yaw)

    # # On reset, update the initial yaw for the reset environments
    # env.initial_handle_yaw[env_ids] = current_yaw[env_ids]

