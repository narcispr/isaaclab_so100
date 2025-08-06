# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs import mdp

from isaaclab_so100.tasks.manager_based.isaaclab_so100.so100_valve_scene_cfg import SO100ValveSceneCfg
import isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp.observations as so100_observations
import isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp.rewards as so100_rewards
import isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp.terminations as so100_terminations

##
# Scene definition
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    # Reset EE pose command to a uniform distribution?? Why it is not done inside Events??
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        gripper_pos = ObsTerm(func=so100_observations.gripper_position_in_robot_base)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        handle_ends_positions = ObsTerm(func=so100_observations.handle_ends_positions_in_robot_base)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset the valve to a random position on the table
    reset_valve_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (0.0, 0.0), "yaw": (-3.14159, 3.14159)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("valve"),
        },
    )

    # reset the robot's joints to a random position
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Reaching reward
    rotate_handle = RewTerm(
        func=so100_rewards.handle_rotation,
        params={"handle_frame_cfg": SceneEntityCfg("handle_frame")},
        weight=100.0
    )
    reaching_object = RewTerm(func=so100_rewards.ee_to_object_distance, params={"std": 0.05, "object_cfg":SceneEntityCfg("valve")}, weight=1.0)

    # Lifting reward
    # lifting_object = RewTerm(func=so100_rewards.object_lifted_distance, weight=1.0)

    # Hight Reward to reaching desired height
    # object_is_lifted = RewTerm(func=so100_rewards.object_is_lifted, params={"minimal_height": 0.2}, weight=100.0)

    # Reward for being close to the object
    # ee_close_to_object = RewTerm(func=so100_rewards.ee_close_to_object, params={"std": 0.05, "threshold": 0.02}, weight=25.0)

    # Action penalty to encourage smooth movements
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # Joint velocity penalty to prevent erratic movements
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-1e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    # )

    # object_reached = DoneTerm(func=so100_terminations.ee_close_to_object_termination, params={"std": 0.05, "threshold": 0.025})
    # object_lifted = DoneTerm(func=so100_terminations.object_is_lifted, params={"minimal_height": 0.2, "object_cfg": SceneEntityCfg("object")})

# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     # Stage 1: Focus on reaching
#     # Start with higher reaching reward, then gradually decrease it
#     reaching_reward = CurrTerm(
#         func=mdp.modify_reward_weight, 
#         params={"term_name": "reaching_object", "weight": 1.0, "num_steps": 6000}
#     )

#     # Stage 2: Transition to lifting
#     # Start with lower lifting reward, gradually increase to encourage lifting behavior
#     lifting_reward = CurrTerm(
#         func=mdp.modify_reward_weight, 
#         params={"term_name": "lifting_object", "weight": 35.0, "num_steps": 8000}
#     )

    # # Stage 4: Stabilize the policy
    # # Gradually increase action penalties to encourage smoother, more stable movements
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, 
    #     params={"term_name": "action_rate", "weight": -5e-4, "num_steps": 12000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, 
    #     params={"term_name": "joint_vel", "weight": -5e-4, "num_steps": 12000}
    # )


##
# Environment configuration
##


@configclass
class SO100ValveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: SO100ValveSceneCfg = SO100ValveSceneCfg(num_envs=4, env_spacing=1.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # Set actions for the specific robot type (SO100)
    actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        scale=0.5,
        use_default_offset=True
    )
    
    # Set gripper action with wider range for better visibility
    actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["Gripper"],
        open_command_expr={"Gripper": 0.5},  # Fully open
        close_command_expr={"Gripper": 0.0}  # flly closed
    )
    
    commands: CommandsCfg = CommandsCfg()
    # Set the body name for the end effector
    commands.object_pose.body_name = "Fixed_Gripper"
    # Disable debug visualization for the target pose command
    commands.object_pose.debug_vis = False
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 4.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        
        # Configure camera for closer view during video recording
        self.viewer.eye = (0.0, -1.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.1)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        