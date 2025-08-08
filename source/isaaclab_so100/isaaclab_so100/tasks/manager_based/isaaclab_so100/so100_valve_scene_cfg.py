# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab_so100.tasks.manager_based.isaaclab_so100.so100_robot_cfg import SO100_CFG  # isort: skip
from isaaclab_so100.tasks.manager_based.isaaclab_so100.valve_cfg import VALVE_CFG  # isort: skip

@configclass
class SO100ValveSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = SO100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # valve
    valve: ArticulationCfg = VALVE_CFG.replace(prim_path="{ENV_REGEX_NS}/Valve",
                                               init_state=ArticulationCfg.InitialStateCfg(
                                                    pos=(0.0, 0.0, 0.0),  # Initial position of the valve
                                                    rot=(1.0, 0.0, 0.0, 0.0)  # Initial orientation of the valve (identity quaternion)
                                               )
                                              )
    
    
    # Frame transformer to get the handle ends relative to the gripper
    handle_ends_wrt_gripper = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Fixed_Gripper",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Valve/valve/handle_end_x_pos"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Valve/valve/handle_end_x_neg"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Valve/valve/handle_end_y_pos"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Valve/valve/handle_end_y_neg"),
        ],
    )

    # Add a frame transformer to indicate Valve handle pose wrt valve base
    handle_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Valve/valve/valve_base",  # the source frame (world in this case)
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/valve/valve_handle",
                name="handle",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            )
        ],
        update_period=0.0,
        debug_vis=True,  # still draw visuals
    )

