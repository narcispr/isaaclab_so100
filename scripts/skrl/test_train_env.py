import argparse
import dataclasses
import torch 
import yaml

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
print(f"Arguments: {args_cli}")
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab_so100.tasks.manager_based.isaaclab_so100.isaaclab_so100_touch_cube_env import SO100TouchCubeEnvCfg
from isaaclab_so100.tasks.manager_based.isaaclab_so100.isaaclab_so100_lift_cube_env import SO100LiftCubeEnvCfg
from isaaclab_so100.tasks.manager_based.isaaclab_so100.isaaclab_so100_valve_env import SO100ValveEnvCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

if __name__ == "__main__":
    # env_cfg = SO100TouchCubeEnvCfg()
    # env_cfg = SO100LiftCubeEnvCfg()
    env_cfg = SO100ValveEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedRLEnv(env_cfg)
    
    # wrap around environment for skrl
    # load the experiment config and instantiate the runner
    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # same as: `wrap_env(env, wrapper="auto")`
    cfg = Runner.load_cfg_from_yaml("/home/narcis/SOARM100/isaaclab_so100/scripts/skrl/config.yaml")
    
    runner = Runner(env, cfg)

    # run training
    runner.run("train")

    # close the simulator
    env.close()