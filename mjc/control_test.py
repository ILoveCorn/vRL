import mujoco
import numpy as np
import matplotlib.pyplot as plt
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("test/mjc/align_hole.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Control input
        nu = model.nu
        ctrl = np.zeros(nu)

        # Generate sinusoidal signals for ctrl[0] and ctrl[3]
        f = 0.5  # Frequency
        # ctrl[0] = 0.1 * np.sin(2 * np.pi * f * data.time)
        ctrl[0] = 0.1
        # ctrl[3] = 0.1 * np.sin(2 * np.pi * f * data.time)

        # Apply control
        data.ctrl = ctrl

        # Step the physics
        mujoco.mj_step(model, data)

        # Print actuator force
        # print("Actuator force:", data.actuator_force)

        # Sync the viewer with the physics state
        viewer.sync()
