# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# import isaaclab_so100.tasks.manager_based.isaaclab_so100.isaaclab_so100_base_env as isaaclab_so100_base_env

##
# Register Gym environments.
##

# Register the SO-100 Cube Lift environment
# gym.register(
#     id="Isaac-So100-CubeTouch-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.isaaclab_so100.tasks.manager_based.isaaclab_so100.isaaclab_so100_base_env:SO100CubeLiftEnvCfg"
#     },
#     disable_env_checker=True,
# )