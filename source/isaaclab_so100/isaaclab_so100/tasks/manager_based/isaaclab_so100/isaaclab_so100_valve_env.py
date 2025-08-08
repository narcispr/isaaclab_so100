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
import isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp.events as so100_events

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
        handle_ends_positions = ObsTerm(func=so100_observations.handle_ends_positions_in_gripper_frame)
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
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, -0.3), "z": (0.0, 0.0), "yaw": (-0.76, 0.76)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("valve"),
        },
    )

    # reset the valve's joints to a random position
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve_joint"),
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
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
    # store the initial valve angle on reset
    reset_valve_angle = EventTerm(
        func=so100_events.reset_initial_valve_angle,
        mode="reset",
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Stage 1: Reaching reward to encourage the agent to move towards the handle
    reaching_handle = RewTerm(func=so100_rewards.gripper_to_closest_handle_end_distance, weight=-1.0)

    # Stage 2: Rotation reward, initialized to 0 and ramped up by the curriculum
    rotate_handle = RewTerm(
        func=so100_rewards.handle_rotation,
        params={"handle_frame_cfg": SceneEntityCfg("handle_frame")},
        weight=0.1
    )

    # Stage 3: Penalties for smooth movements, initialized to 0 and ramped up by the curriculum
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # gripper_close_to_handle = DoneTerm(func=so100_terminations.gripper_close_to_any_handle_end, params={"threshold": 0.05})
    # valve_rotated = DoneTerm(func=so100_terminations.valve_rotated_past_threshold)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Stage 2: After N steps, ramp up the rotation reward from 0.0 to 100.0
    rotate_handle_curriculum = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "rotate_handle", "weight": 0.5, "num_steps": 10000}
    )

    # Stage 3: After N steps, ramp up the action penalties to encourage movements
    action_rate_curriculum = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-4, "num_steps": 80000}
    )
    joint_vel_curriculum = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-4, "num_steps": 80000}
    )


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
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
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
        