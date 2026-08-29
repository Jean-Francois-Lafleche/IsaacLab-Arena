# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym wrapper recording the run from fixed vantage points of our own choosing.

Isaac Lab's own recorder gives one video, from the viewer. A grid of a hundred and twenty-eight rooms
needs two: a wide shot showing the whole grid training, in which a single room is a few dozen pixels,
and a close shot of a handful of rooms in which the task is actually legible.

The obvious way to add the second -- a camera sensor in the scene -- does not work here. Isaac Lab
prepares one render context for the whole simulation and fixes its instance count from the first
camera initialised, so a single scene-wide camera standing beside the robots' own per-environment
cameras is rejected outright. Replicating it per environment instead would render the same wide shot
a hundred and twenty-eight times over.

So these are not sensors. Each view drives its own render product against a plain USD camera, through
the same capture class Isaac Lab records the viewer with, which sidesteps the sensor machinery and its
instance count entirely and costs one render product per view.
"""

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import os
from dataclasses import dataclass

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

VIEW_ROOT = "/World/scene_views"
"""Where the cameras these views render through are authored, outside the environment hierarchy."""

FOCAL_LENGTH_MM = 24.0
"""Focal length the views' cameras are given.

Only the ratio of this to the aperture sets the field of view, so the value itself is arbitrary; it is
fixed so that a view's aperture can be solved from the field of view it asks for.
"""

NEAR_CLIP_M = 0.5
"""Nearest distance a view's camera renders.

Far enough out to be past nothing in particular: these cameras stand tens of metres back, and a near
plane at the default fraction of a millimetre costs depth precision across the whole grid.
"""

FAR_CLIP_M = 1.0e5
"""Furthest distance a view's camera renders, wide enough to cover a grid of any size."""


# Slots as well as frozen: Isaac Lab walks a configuration's contents to rewrite the strings in it,
# assigning into any dataclass it finds along the way, and without a ``__dict__`` a view is not one it
# can descend into.
@dataclass(frozen=True, slots=True)
class SceneView:
    """One vantage point to record the run from.

    Attributes:
        name: Short name, used in the filename of the videos this view produces.
        eye: Where the camera stands, in world coordinates.
        lookat: The point it is aimed at, in world coordinates.
        horizontal_fov_deg: Horizontal field of view, which decides how much of the grid it takes in.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    name: str
    eye: tuple[float, float, float]
    lookat: tuple[float, float, float]
    horizontal_fov_deg: float
    width: int
    height: int


def define_view_camera(stage, view: SceneView) -> str:
    """Author a camera for a view and return its prim path.

    Args:
        stage: The stage to author on.
        view: The view whose framing the camera is given.
    """
    from pxr import Gf, UsdGeom

    path = f"{VIEW_ROOT}/{view.name}"
    camera = UsdGeom.Camera.Define(stage, path)
    aperture = 2.0 * FOCAL_LENGTH_MM * math.tan(math.radians(view.horizontal_fov_deg) / 2.0)
    camera.CreateFocalLengthAttr(FOCAL_LENGTH_MM)
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture * view.height / view.width)
    camera.CreateClippingRangeAttr(Gf.Vec2f(NEAR_CLIP_M, FAR_CLIP_M))
    return path


class SceneViewVideoRecorder(gym.Wrapper):
    """Record one mp4 per configured view, in windows, alongside whatever else is recording."""

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        views: tuple[SceneView, ...],
        name_prefix: str = "scene-view",
        interval: int = 2000,
        length: int = 200,
        fps: int | None = None,
    ):
        """
        Args:
            env: The environment to wrap.
            video_folder: Directory the mp4s are written to, created if missing.
            name_prefix: Leading part of each filename, before the view name and step.
            views: The vantage points to record from.
            interval: Environment steps between the start of one recording window and the next.
            length: Steps recorded in each window.
            fps: Frame rate of the mp4s. Defaults to the environment's own render rate.
        """
        super().__init__(env)
        assert views, "a scene view recorder with no views records nothing"
        assert length <= interval, f"a {length}-step window does not fit in an interval of {interval}"
        os.makedirs(video_folder, exist_ok=True)
        self._video_folder = video_folder
        self._views = views
        self._name_prefix = name_prefix
        self._interval = interval
        self._length = length
        self._fps = fps if fps is not None else int(env.metadata.get("render_fps", 30))
        self._step_index = 0
        self._window_start: int | None = None
        self._captures: dict[str, object] = {}
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

    def _capture_for(self, view: SceneView):
        """Return the capture that renders a view, building it and its camera on first use.

        Deferred to the first recorded frame because the stage a camera is authored on does not exist
        when the wrapper is constructed.

        Args:
            view: The view to render.
        """
        capture = self._captures.get(view.name)
        if capture is None:
            from isaaclab_physx.video_recording.isaacsim_kit_perspective_video import (
                create_isaacsim_kit_perspective_video,
            )
            from isaaclab_physx.video_recording.isaacsim_kit_perspective_video_cfg import IsaacsimKitPerspectiveVideoCfg

            path = define_view_camera(self.env.unwrapped.scene.stage, view)
            capture = create_isaacsim_kit_perspective_video(
                IsaacsimKitPerspectiveVideoCfg(
                    camera_prim_path=path,
                    eye=view.eye,
                    lookat=view.lookat,
                    window_width=view.width,
                    window_height=view.height,
                )
            )
            self._captures[view.name] = capture
        return capture

    def _record(self) -> None:
        """Append one frame per view to the videos of the window in progress."""
        for view in self._views:
            frame = self._capture_for(view).render_rgb_array()
            if frame is None:
                continue
            frame = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[:, :, :3])
            writer = self._writers.get(view.name)
            if writer is None:
                name = f"{self._name_prefix}-{view.name}-step-{self._window_start}.mp4"
                height, width, _ = frame.shape
                # One thread, as with the other recorders: frames arrive slower than they encode.
                writer = FFMPEG_VideoWriter(
                    os.path.join(self._video_folder, name), size=(width, height), fps=self._fps, threads=1
                )
                self._writers[view.name] = writer
            writer.write_frame(frame)

    def _close_writers(self) -> None:
        """Finalise the mp4 of every view recorded in the window that just ended."""
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def close(self) -> None:
        self._close_writers()
        self.env.close()
