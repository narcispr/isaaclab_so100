# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def handle_ends_positions_in_gripper_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The positions of the four handle ends, relative to the gripper frame."""
    # The FrameTransformer is configured to compute this directly
    frame_transformer: FrameTransformer = env.scene["handle_ends_wrt_gripper"]
    # The data is available in target_pos_source, which has shape (num_envs, num_targets, 3)
    # We just need to flatten it for the observation vector
    return frame_transformer.data.target_pos_source.view(env.num_envs, -1)
