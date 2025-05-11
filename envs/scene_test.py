"""
Test scene with pyrep
2025-05-08
"""

from os.path import dirname, join, abspath

import math

from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

import matplotlib.pyplot as plt

SCENE_FILE = join(dirname(abspath(__file__)), "scene/hole_alignment.ttt")

# Simulator
pr = PyRep()
pr.launch(SCENE_FILE, headless=True)

# Simulation timestep
dt = pr.get_simulation_timestep()

# Camera
cam = VisionSensor("Camera")

# FOV
fov = cam.get_perspective_angle()

# Part position
part_position = Shape("Part").get_position()

# Start simulation
pr.start()

while sim.simGetSimulationTime() <= 5:
    # Step simulation
    pr.step()

    # Get camera image
    img = cam.capture_rgb()

    # Control camera position
    cam_position = cam.get_position()
    # cam_position[0] += 0.01 * dt
    # cam_position[1] += 0.01 * dt
    # cam_position[2] += 0.01 * dt
    cam.set_position(cam_position)

    # Control camera orientation
    if sim.simGetSimulationTime() <= 5:
        dpitch = 0 / 180 * math.pi * dt
        dyaw = 5 / 180 * math.pi * dt
    else:
        dpitch = 0 / 180 * math.pi * dt
        dyaw = 5 / 180 * math.pi * dt
    cam.rotate([dpitch, dyaw, 0])

    # NOTE: camera's x.y axes are opposite to image's u.v axes

    # Show image
    plt.imshow(img)
    plt.axis("off")
    # plt.show(block=False)
    plt.pause(0.1)
    plt.clf()


# Stop simulation
pr.stop()
pr.shutdown()
plt.close()
