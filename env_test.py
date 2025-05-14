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

from PIL import Image


def add_separator(image, thickness=5, color=(0, 0, 0)):
    """Adds a vertical separator to the right of the image."""
    separator = np.zeros((image.shape[0], thickness, image.shape[2]), dtype=np.uint8)
    separator[:, :, :] = color  # Set color
    return np.concatenate([image, separator], axis=1)


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

# check_env(env)

img1s = []

# reset & step
for i in range(5):
    obs1, info = env.reset()
    img1 = env.img

    obs2, _, _, _, info = env.step(np.array([0, 0, 0, 1, 0]))
    img2 = env.img

    # concatenate images to show action result
    img12 = np.concatenate([add_separator(img1), img2], axis=1)

    plt.imshow(img12)
    plt.title("action=[0, 0, 0, 5, 0]")
    plt.axis("off")
    plt.subplots_adjust(
        left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
    )  # Adjust the subplot parameters

    # Save the figure to a file
    plt.savefig("media/action_rx.png")

    plt.show()

    # add seperator
    if i != 4:
        img1s.append(add_separator(img1))
    else:
        img1s.append(img1)

env.close()

# concatenate images horizontally
concatenated_img1s = np.concatenate(img1s, axis=1)

# save the concatenated image
Image.fromarray(concatenated_img1s.astype(np.uint8)).save("media/reset_demo.png")
