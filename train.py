import gymnasium as gym

import envs

from stable_baselines3 import PPO, SAC

import wandb
from wandb.integration.sb3 import WandbCallback


config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 200_0000,
    "env_id": "AlignHole-v0",
}
run = wandb.init(
    project="vRL",
    name="SAC-200w",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

model = SAC(
    config["policy_type"],
    config["env_id"],
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    buffer_size=8_0000,
)
model.learn(
    total_timesteps=config["total_timesteps"], callback=WandbCallback(verbose=2)
)

model.save("./model/SAC_200w_Greyscale")

run.finish()


# env = gym.make(
#     "AlignHole-v0",
#     max_position_action=0.01,
#     max_orientation_action=1,
#     render_mode="rgb_array",
#     display_ui=False,
# )

# model = PPO("CnnPolicy", env, verbose=1)

# model.learn(total_timesteps=100_0000)

# model.save("./model/PPO_100w_Greyscale")
