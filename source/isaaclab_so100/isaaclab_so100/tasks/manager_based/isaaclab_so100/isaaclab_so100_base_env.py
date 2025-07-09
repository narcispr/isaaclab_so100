# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import dataclasses

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



# ^  ^  ^
# |  |  |
#  REMOVE



# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import mdp
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_so100.tasks.manager_based.isaaclab_so100.so100_scene_cfg import SO100SceneCfg
import isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp.observations as so100_observations
import isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp.rewards as so100_rewards
# import isaaclab_so100.tasks.manager_based.isaaclab_so100.mdp.terminations as so100_terminations


# Just for runnit here:
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

##
# Scene definition
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
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
        object_position = ObsTerm(func=so100_observations.object_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Reaching reward with lower weight
    reaching_object = RewTerm(func=so100_rewards.object_ee_distance, params={"std": 0.05}, weight=2)

    # Lifting reward with higher weight
    lifting_object = RewTerm(func=so100_rewards.object_is_lifted, params={"minimal_height": 0.02}, weight=25.0)

    # Action penalty to encourage smooth movements
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # Joint velocity penalty to prevent erratic movements
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


# @configclass
class CurriculumCfg:
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

    # Stage 4: Stabilize the policy
    # Gradually increase action penalties to encourage smoother, more stable movements
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "action_rate", "weight": -5e-4, "num_steps": 12000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "joint_vel", "weight": -5e-4, "num_steps": 12000}
    )


##
# Environment configuration
##


@configclass
class SO100LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: SO100SceneCfg = SO100SceneCfg(num_envs=4, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
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
        self.viewer.eye = (1.0, 1.0, 0.8)
        self.viewer.lookat = (0.5, 0.0, 0.2)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0



@configclass
class SO100CubeLiftEnvCfg(SO100LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (SO100)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=0.5,
            use_default_offset=True
        )
        
        # Set gripper action with wider range for better visibility
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Gripper"],
            open_command_expr={"Gripper": 0.5},  # Fully open
            close_command_expr={"Gripper": 0.0}  # flly closed
        )
        
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "Fixed_Gripper"
        # Disable debug visualization for the target pose command
        self.commands.object_pose.debug_vis = False

        

def main():
    """Main function."""
    
    print("Starting SO100 Lift Environment...")

    # parse the arguments
    env_cfg = SO100CubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedRLEnv(env_cfg)
    
    # simulate physics
    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            # print joint efforts
            joint_efforts = torch.tensor([[2.0, 0, 0, 0, 0, 0.0]], device="cuda:0")
            print("[Env 0]: joint_efforts: ", joint_efforts[0])
            obs = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: obs: ", obs[0]["policy"])
            print("[Env 0] reward:", obs[1].item())
            # update counter
            count += 1

    # close the environment
    env.close()

if __name__ == "__main__":
    # run the main function
    print("Running main...")
    main()
    # close sim app
    simulation_app.close()
