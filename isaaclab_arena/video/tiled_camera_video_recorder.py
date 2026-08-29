# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym wrapper recording one mp4 per camera in ``obs['camera_obs']``, tiled across environments.

The sibling ``CameraObsVideoRecorder`` writes a file per (environment, camera, episode), which suits
an evaluation rollout of a handful of environments. A training run of a hundred and twenty-eight
would write thousands of clips nobody opens. This lays the same frames out as one grid per camera
instead, so a single video shows what every robot is looking at at the same moment.

Read off the scene's sensors rather than out of the observations, unlike its sibling. Routing the
images through the observation manager would give the environment a second observation group, and
Isaac Lab's RSL-RL wrapper answers that by handing the learner a tensor dictionary in place of a
policy tensor, which is not what the learner's own input checks expect. Recording is a bystander to
training and should not change what the policy is trained on.

Recorded in windows rather than continuously, on the same pattern as gymnasium's ``RecordVideo``:
pulling a batch of camera images back from the GPU on every step of a training run would cost more
than the training does.
"""

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import os
import torch
from typing import Any

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from isaaclab_arena.video.camera_observation_video_recorder import sanitize_camera_name, to_uint8

RGB_OUTPUT_KEY = "rgb"
"""Camera output recorded. Depth and segmentation need colouring before they are an image."""

GUTTER_PIXELS = 2
"""Width of the dark seam drawn between neighbouring tiles.

Without it two adjacent sensor views of the same dim interior read as one wide image.
"""


def tile_frames(frames: np.ndarray) -> np.ndarray:
    """Lay a batch of camera images out as a single grid image.

    Args:
        frames: The batch to lay out, shaped ``(count, height, width, channels)``.

    Returns:
        The grid, shaped ``(rows * height, columns * width, channels)`` plus the seams between tiles.
    """
    count, height, width, channels = frames.shape
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    canvas = np.zeros(
        (
            rows * height + (rows - 1) * GUTTER_PIXELS,
            columns * width + (columns - 1) * GUTTER_PIXELS,
            channels,
        ),
        dtype=np.uint8,
    )
    for index in range(count):
        row, column = divmod(index, columns)
        top = row * (height + GUTTER_PIXELS)
        left = column * (width + GUTTER_PIXELS)
        canvas[top : top + height, left : left + width] = frames[index]
    return canvas


def rgb_cameras(env: gym.Env) -> dict[str, Any]:
    """Return the scene's cameras that publish colour images, keyed by their scene name.

    Args:
        env: The wrapped environment, whose ``unwrapped`` scene holds the sensors.
    """
    sensors = getattr(env.unwrapped.scene, "sensors", {})
    return {
        name: sensor
        for name, sensor in sensors.items()
        if RGB_OUTPUT_KEY in getattr(getattr(sensor, "data", None), "output", {})
    }


class TiledCameraObsVideoRecorder(gym.Wrapper):
    """Record one mp4 per scene camera, each frame tiled across environments."""

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        name_prefix: str = "sensor-tiles",
        interval: int = 2000,
        length: int = 200,
        max_tiles: int = 144,
        fps: int | None = None,
    ):
        """
        Args:
            env: The environment to wrap.
            video_folder: Directory the mp4s are written to, created if missing.
            name_prefix: Leading part of each filename, before the camera name and step.
            interval: Environment steps between the start of one recording window and the next.
            length: Steps recorded in each window.
            max_tiles: Most environments to show. Beyond this the grid's tiles are too small to read
                and the copy off the GPU stops being cheap, so the surplus environments are dropped.
            fps: Frame rate of the mp4s. Defaults to the environment's own render rate.
        """
        super().__init__(env)
        assert length <= interval, f"a {length}-step window does not fit in an interval of {interval}"
        os.makedirs(video_folder, exist_ok=True)
        self._video_folder = video_folder
        self._name_prefix = name_prefix
        self._interval = interval
        self._length = length
        self._max_tiles = max_tiles
        self._fps = fps if fps is not None else int(env.metadata.get("render_fps", 30))
        self._step_index = 0
        self._window_start: int | None = None
        self._writers: dict[str, FFMPEG_VideoWriter] = {}

    def step(self, action):
        result = self.env.step(action)
        if self._window_start is None and self._step_index % self._interval == 0:
            self._window_start = self._step_index
        if self._window_start is not None:
            self._record()
            if self._step_index - self._window_start + 1 >= self._length:
                self._close_writers()
                self._window_start = None
        self._step_index += 1
        return result

    def _record(self) -> None:
        """Append one tiled frame per camera to the videos of the window in progress."""
        for camera_name, camera in rgb_cameras(self.env).items():
            batch = camera.data.output[RGB_OUTPUT_KEY][: self._max_tiles]
            if isinstance(batch, torch.Tensor):
                batch = batch.detach().cpu()
            tiled = tile_frames(to_uint8(batch)[..., :3])
            assert (
                tiled.ndim == 3 and tiled.shape[2] == 3
            ), f"camera {camera_name!r} tiled to shape {tiled.shape}, and the encoder takes (H, W, 3) RGB"
            writer = self._writers.get(camera_name)
            if writer is None:
                name = f"{self._name_prefix}-{sanitize_camera_name(camera_name)}-step-{self._window_start}.mp4"
                height, width, _ = tiled.shape
                # One thread, as with the per-episode recorder: frames arrive slower than they encode.
                writer = FFMPEG_VideoWriter(
                    os.path.join(self._video_folder, name), size=(width, height), fps=self._fps, threads=1
                )
                self._writers[camera_name] = writer
            writer.write_frame(tiled)

    def _close_writers(self) -> None:
        """Finalise the mp4 of every camera recorded in the window that just ended."""
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def close(self) -> None:
        self._close_writers()
        self.env.close()
