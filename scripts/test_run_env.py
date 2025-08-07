import argparse
import dataclasses
import torch 

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


from isaaclab_so100.tasks.manager_based.isaaclab_so100.isaaclab_so100_valve_env import SO100ValveEnvCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets.articulation import Articulation

if __name__ == "__main__":
    env_cfg = SO100ValveEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedRLEnv(env_cfg)
    while simulation_app.is_running():
        print("RESET!")
        asset: Articulation = env.scene['valve']
        env_ids: torch.Tensor = torch.arange(0, env_cfg.scene.num_envs, device=env_cfg.sim.device)
        joint_pos = asset.data.default_joint_pos[env_ids].clone()
        joint_vel = asset.data.default_joint_vel[env_ids].clone()
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset environment
        env.reset()
        # run the environment
        obs = None
        while obs is None or not obs[3][0]: # done param in obs
            # sample random actions
            joint_efforts = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device="cuda:0")
            obs = env.step(joint_efforts)
            # print keys in obs
            print(f"Reward: {obs[1]}")
            # print(f"Obs: {obs[0]}")
            