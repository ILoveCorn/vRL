"""
Test custom gymnasium environment
2025-05-09
"""

import envs
from envs.AlignHole_v0 import AlignHoleEnv

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import matplotlib.pyplot as plt

import numpy as np

import time

##################
# Gym wrapped env
##################
# env = gym.make("AlignHole-v0", display_ui=True)

# check_env(env)

# not working w/ or w/o UI


##########
# Raw env
##########
env = AlignHoleEnv(display_ui=False)

check_env(env)

# reset & step
# for i in range(5):
#     obs1, info = env.reset()

#     obs2, _, _, _, info = env.step(np.array([0, 0, 0.1, 0, 0]))

#     plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
#     plt.imshow(obs1)
#     plt.axis("off")

#     plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
#     plt.imshow(obs2)
#     plt.axis("off")

#     plt.show()

# env.close()
