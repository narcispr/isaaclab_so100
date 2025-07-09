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


from isaaclab_so100.tasks.manager_based.isaaclab_so100.isaaclab_so100_base_env import SO100CubeLiftEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

if __name__ == "__main__":
    env_cfg = SO100CubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedRLEnv(env_cfg)
    while simulation_app.is_running():
        # reset environment
        env.reset()
        # run the environment
        obs = None
        while obs is None or not torch.any(obs[2]):
            # sample random actions
            joint_efforts = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]], device="cuda:0")
            obs = env.step(joint_efforts)
            # print current orientation of pole
            