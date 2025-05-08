"""
Test scene with pyrep
2025-05-08
"""

from os.path import dirname, join, abspath
import math
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.vision_sensor import VisionSensor
import matplotlib.pyplot as plt

SCENE_FILE = join(dirname(dirname(abspath(__file__))), "scene/hole_alignment.ttt")

# Simulator
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)

# Simulation timestep
dt = pr.get_simulation_timestep()

# Camera
cam = VisionSensor("Camera")

# Start simulation
pr.start()

while sim.simGetSimulationTime() <= 10:
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
        dpitch = 5 / 180 * math.pi * dt
        dyaw = 0 / 180 * math.pi * dt
    cam.rotate([dpitch, dyaw, 0])

    # Show image
    plt.imshow(img)
    plt.axis("off")
    plt.pause(0.01)
    plt.clf()


# Stop simulation
pr.stop()
pr.shutdown()
plt.close()
