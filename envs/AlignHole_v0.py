"""
Hole alignment environment using pyrep

---------------
Version history
---------------
v0  2025-05-08: initial version with fixed part
"""

from typing import Union

from os.path import dirname, join, abspath

import numpy as np

from gymnasium import utils, Env
from gymnasium.spaces import Box

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape


class AlignHoleEnv(Env, utils.EzPickle):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        timestep: float = 0.05,
        display_ui: bool = False,
        render_mode: Union[None, str] = "rgb_array",
        scene_file: str = "hole_alignment.ttt",
        image_size: int = 256,
        hole_axis: list = [0, -1, 1],
        orientation_err_weight: float = 1,
        max_position_action: float = 0.1,
        max_orientation_action: float = 5,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            timestep,
            display_ui,
            render_mode,
            scene_file,
            image_size,
            hole_axis,
            orientation_err_weight,
            max_position_action,
            max_orientation_action,
            **kwargs,
        )

        # set render mode
        self.render_mode = render_mode

        hole_axis = np.array(hole_axis)
        # normalize hole axis
        self.hole_axis = hole_axis / np.linalg.norm(hole_axis)
        self.orientation_err_weight = orientation_err_weight

        # normalized action [dx, dy, dz, dpitch, dyaw] in camera frame
        self.action_space = Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float64)
        self.max_position_action = max_position_action
        self.max_orientation_action = max_orientation_action

        # RGB observation
        self.observation_space = Box(
            low=0, high=255, shape=(image_size, image_size, 1), dtype=np.uint8
        )

        self.SCENE_FILE = join(dirname(abspath(__file__)), "scene/", scene_file)

        # initialize the environment
        self.pr = PyRep()
        self.pr.launch(self.SCENE_FILE, headless=not display_ui)

        # set simulation timestep
        self.pr.set_simulation_timestep(timestep)

        # set image resolution
        self.cam = VisionSensor("Camera")
        self.cam.set_resolution([image_size, image_size])

        # get camera fov
        self.fov = self.cam.get_perspective_angle()

        # get hole position
        self.hole_position = Shape("Part").get_position()

        # start simulation
        self.pr.start()

    def step(self, action):
        # note: camera's x.y axes are opposite to image's u.v axes

        # get camera pose
        self.cam_transformation = self.cam.get_matrix()

        # update camera pose
        self.cam.set_position(
            self.cam_transformation[:3, :3] @ (self.max_position_action * action[:3])
            + self.cam_transformation[:3, 3]
        )
        self.cam.rotate(
            self.max_orientation_action
            * np.array([action[3], action[4], 0])
            / 180
            * np.pi
        )

        # step simulation
        self.pr.step()

        # capture image
        self.img = np.clip((self.cam.capture_rgb() * 255.0).astype(np.uint8), 0, 255)
        # RGB to greyscale
        observation = np.clip(
            np.dot(self.img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8), 0, 255
        )
        observation = np.expand_dims(observation, axis=-1)

        # alignment error
        position_err = self._position_err()
        orientation_err = self._orientation_err()

        # calculate reward
        # linear reward
        reward = -position_err - self.orientation_err_weight * orientation_err

        # exponential reward
        # reward = np.exp(reward)

        # sparse reward
        # if -reward <= 10:
        #     reward += 10

        # terminate if hole is outside the visible cone
        terminated = position_err >= self.fov / 2

        # other info
        info = {"Position error": position_err, "Orientation error": orientation_err}

        return observation, reward, terminated, False, info

    def _position_err(self):
        rz_cam = self.cam_transformation[:3, 2]
        v_cam_to_hole = self.hole_position - self.cam_transformation[:3, 3]

        return (
            np.acos(
                np.clip(
                    rz_cam.dot(v_cam_to_hole)
                    / np.linalg.norm(rz_cam)
                    / np.linalg.norm(v_cam_to_hole),
                    -1.0,
                    1.0,
                )
            )
            / np.pi
            * 180
        )

    def _orientation_err(self):
        rz_cam = self.cam_transformation[:3, 2]

        angle = (
            np.acos(
                np.clip(rz_cam.dot(self.hole_axis) / np.linalg.norm(rz_cam), -1.0, 1.0)
            )
            / np.pi
            * 180
        )

        return angle if angle < 90 else 180 - angle

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        # random camera pose [x, y, z, rx, ry] in spherical coordinates
        r = self.np_random.uniform(low=0.05, high=0.2)
        theta = self.np_random.uniform(low=0, high=2 * np.pi)
        phi = self.np_random.uniform(low=0, high=22.5 / 180 * np.pi)
        rz_cam = -np.array(
            [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        )

        # calculate camera pose
        if rz_cam[0] != 0 or rz_cam[1] != 0:
            rx_cam = np.array([-rz_cam[1], rz_cam[0], 0])
            rx_cam = rx_cam / np.linalg.norm(rx_cam)
        else:
            rx_cam = np.array([1, 0, 0])
        ry_cam = np.linalg.cross(rz_cam, rx_cam)

        cam_position = self.hole_position - r * rz_cam

        cam_transformation = np.concatenate(
            [
                np.column_stack([rx_cam, ry_cam, rz_cam, cam_position]),
                np.array([0, 0, 0, 1]).reshape(1, -1),
            ]
        )

        # set camera pose
        self.cam.set_matrix(cam_transformation)

        # get camera pose
        self.cam_transformation = self.cam.get_matrix()

        # initial observation
        self.pr.step()
        self.img = np.clip((self.cam.capture_rgb() * 255.0).astype(np.uint8), 0, 255)
        # RGB to greyscale
        observation = np.clip(
            np.dot(self.img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8), 0, 255
        )
        observation = np.expand_dims(observation, axis=-1)

        # initial info
        info = {
            "Position error": self._position_err(),
            "Orientation error": self._orientation_err(),
        }

        return observation, info

    def render(self):
        return self.img

    def close(self):
        self.pr.stop()
        self.pr.shutdown()
