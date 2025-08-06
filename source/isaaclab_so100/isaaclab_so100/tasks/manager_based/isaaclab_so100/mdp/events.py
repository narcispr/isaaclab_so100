# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_initial_valve_angle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    handle_frame_cfg: SceneEntityCfg,
):
    """Stores the initial yaw of the valve handle at the start of the episode."""
    handle_frame: FrameTransformer = env.scene[handle_frame_cfg.name]
    quat = handle_frame.data.target_quat_source[:, 0, :]

    # Convert quaternion to yaw angle
    z = quat[:, 2]
    w = quat[:, 3]
    current_yaw = 2.0 * torch.atan2(z, w)

    # Initialize initial_yaw at the first step
    if not hasattr(env, "initial_handle_yaw"):
        env.initial_handle_yaw = torch.zeros_like(current_yaw)

    # On reset, update the initial yaw for the reset environments
    env.initial_handle_yaw[env_ids] = current_yaw[env_ids]
