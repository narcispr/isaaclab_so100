import argparse
import dataclasses
import torch 
import yaml
import time

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


# 🔁 Let the app run 1 tick to finish initializing
simulation_app.update()

# -----------------------------------------------
# Configure rendering
import carb

settings = carb.settings.get_settings()

# 🔁 Switch to RTX Path Traced renderer
settings.set("/app/renderer", "RayTracedPathTracing")  # or just "PathTracing" in some versions

# ✅ Enable Path Tracing rendering
settings.set("/rtx/pathtraced/enabled", True)

# 🎯 Set samples per pixel to 2
settings.set("/rtx/pathtraced/spp", 2)

# (Optional) Disable denoisers meant for real-time
settings.set("/rtx/denoiser/enable", True)
settings.set("/rtx/rayTracedLighting/enabled", True)

# (Optional) Anti-aliasing – often not needed with path tracing
settings.set("/rtx/post/aa/enableTAA", False)
# -----------------------------------------------



from isaaclab_so100.tasks.manager_based.isaaclab_so100.isaaclab_so100_touch_cube_env import SO100TouchCubeEnvCfg
from isaaclab_so100.tasks.manager_based.isaaclab_so100.isaaclab_so100_lift_cube_env import SO100LiftCubeEnvCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

if __name__ == "__main__":
    # env_cfg = SO100TouchCubeEnvCfg()
    env_cfg = SO100LiftCubeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedRLEnv(env_cfg)
    
     # wrap around environment for skrl
     # load the experiment config and instantiate the runner
    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # same as: `wrap_env(env, wrapper="auto")`
    cfg = Runner.load_cfg_from_yaml("/home/narcis/SOARM100/isaaclab_so100/scripts/skrl/config.yaml")
    
    
    runner = Runner(env, cfg)

    runner.agent.load("/home/narcis/SOARM100/isaaclab_so100/scripts/skrl/so100_lift_cube_PPO/25-07-10_14-41-33-203367_PPO/checkpoints/best_agent.pt")
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    dt = env.step_dt
    print(f"[INFO] Environment step time: {dt:.3f} seconds")
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
            
        # time delay for real-time evaluation
        sleep_time = max(dt - (time.time() - start_time), 0.0)
        time.sleep(sleep_time)

    # close the simulator
    env.close()