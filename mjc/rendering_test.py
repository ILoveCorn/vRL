# MuJoCo rendering test
# 2025-05-05

import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Make model and data
model = mujoco.MjModel.from_xml_path("mjc/align_hole.xml")
data = mujoco.MjData(model)

# Render the model
with mujoco.Renderer(model, 800, 800) as renderer:
    # Forward simulation step
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="hand_camera")
    # renderer.update_scene(data)

    # Get rendered image
    img = renderer.render()

    # Display the image using matplotlib
    plt.imshow(img)
    plt.axis("off")
    plt.show()
