import gymnasium as gym

import envs

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt


"""
evaluate_policy
"""
# create env
env = gym.make(
    "AlignHole-v0",
    max_position_action=0.01,
    max_orientation_action=1,
    render_mode="rgb_array",
)

# random model
random_model = PPO("CnnPolicy", env)
# trained model
model = PPO.load("./model/PPO_100w_Greyscale", env)

# # evaluate policy
# mean_reward_random_model, std_reward_random_model = evaluate_policy(
#     random_model,
#     env,
#     n_eval_episodes=10,
#     deterministic=True,
# )
# mean_reward, std_reward = evaluate_policy(
#     model,
#     env,
#     n_eval_episodes=10,
#     deterministic=True,
# )

# print("---------------------------------------------------")
# print(
#     f"mean reward of random model ={mean_reward_random_model:.2f} +/- {std_reward_random_model}"
# )
# print(f"mean reward of trained model ={mean_reward:.2f} +/- {std_reward}")
# print("---------------------------------------------------")


"""
display action
"""
# test model
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    # predict
    action, _ = model.predict(obs, deterministic=True)

    # interact
    obs, reward, done, info = vec_env.step(action)

    # reset
    # vec_env.reset()

    # display image
    img = vec_env.render()
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Step: {i}")
    plt.pause(0.1)

    print("---------------------------------------------------")
    print(f"Step: {i}")
    print(f"Action: {action}")
    # print(f"Info: {info}")
    print("---------------------------------------------------")
