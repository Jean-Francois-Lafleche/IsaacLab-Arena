# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
    from isaaclab_arena.scene.object_reference import ObjectReference

VIDEO_RESOLUTION = (1920, 1080)

KIT_PERSPECTIVE_FOV_DEG = 60.0
"""Horizontal field of view of ``/OmniverseKit_Persp``, which is what both the viewer and the
recorder render through. Kit's default focal length and aperture are chosen to give exactly this,
and neither ``ViewerCfg`` nor the recorder exposes a way to change or read it back."""

SCENE_HEIGHT = 3.0
"""Height of one environment's contents, for working out how much vertical frame they need."""

CAMERA_PITCH_DEG = 62.0
"""How far below horizontal the camera looks.

The generated rooms have walls but no ceiling, so this is what decides how much of each interior is
visible: a wall of height h hides h / tan(pitch) of the floor behind it, which at this angle is about
a third of a room rather than the half a shallower angle would cost. Steeper still would show the
interiors best of all but flatten the grid into a floor plan.
"""

OVERVIEW_CAMERA_NAME = "overview"
"""Name of the close view, which also names the video file the recorder writes for it."""

OVERVIEW_LENS_FOV_DEG = 50.0
"""Horizontal field of view of the close view, in degrees.

Sets how much perspective the shot has: a narrower lens stands further back and flattens the rooms,
a wider one leans over them."""

OVERVIEW_RESOLUTION = (1280, 720)
"""Frame size of the close view, in pixels."""

CAMERA_AZIMUTH_DEG = -135.0
"""Compass bearing of the eye from what it looks at, measured from +x towards +y.

On a diagonal, because that is the direction a square grid presents its narrowest face from and so
the one that fits most environments into a wide frame. This diagonal in particular puts the eye
behind the robots: they stand at the low-x edge of their rooms facing the worktop at high x, so a
camera at high x would look at the backs of the cabinets and the fronts of the robots, with the
drawer each one is working on hidden behind its own arm.
"""

FRAMING_MARGIN = 1.02
"""Fraction of extra distance beyond the exact fit, so the widest environments are not clipped.

Small, because the fit is to the frame's width and any slack there shows as dead space down both
sides of the shot."""


@dataclass
class LiftObjectLiveEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the lift-object environment on a generated scene."""

    live_variation_pool_dir: str | None = None
    """Set by the shared Arena parser; the pool supplies both the scene and its per-reset variations."""

    lift_prim: str | None = None
    """Pool prim to lift, as it appears in the manifest inventory. Defaults to the first liftable prim."""

    embodiment: str = "franka_joint_pos"
    """Use joint control by default because it is more reliable than IK for RL training."""

    embodiment_cameras: bool = False
    """Spawn the robot's own cameras.

    Kept separate from ``enable_cameras`` because recording training video forces cameras on at the
    app level, and that alone should not add a set of robot sensors to every environment.
    """

    framed_env_count: int | None = None
    """How many environments the viewer frames. ``None`` frames all of them.

    A full grid of 128 shows that training is running but not what any robot is doing; a block of
    about nine is close enough to see whether the scene variation is sane.

    Not named after video, despite being what a recording frames: Isaac Lab's launcher owns the
    ``--video`` flags, and a ``--video_*`` flag of ours gets its value eaten before Arena's parser
    sees it.
    """

    rl_training_mode: bool = False


def _select_lift_prim(references: list[ObjectReference], lift_prim: str | None) -> ObjectReference:
    """Pick the pool prim the task lifts.

    Args:
        references: Object references in manifest inventory order.
        lift_prim: Requested prim, matched against the reference names. ``None`` takes the first
            reference whose name marks it as a graspable object, falling back to the first reference.

    Returns:
        The reference the task should lift.
    """
    assert references, "the variation pool exposes no variable prims to lift"
    if lift_prim is not None:
        wanted = lift_prim.strip("/").replace("/", "_")
        matches = [ref for ref in references if ref.name == wanted or ref.name.endswith(wanted)]
        assert matches, f"no pool prim matches --lift_prim {lift_prim!r}; available: {[r.name for r in references]}"
        return matches[0]
    graspable = [ref for ref in references if "object" in ref.name.lower()]
    return graspable[0] if graspable else references[0]


def _environment_block_centre(num_envs: int, env_spacing: float, block_size: int) -> tuple[float, float]:
    """Return the centre of a block of ``block_size`` environments, in world coordinates.

    The block is taken from the corner of the grid the camera stands over, so that the environments
    it is not framing lie beyond it and fall off the top of a steeply pitched frame, rather than
    across the middle of the shot. A camera aimed at the origin would miss the block entirely, since
    Isaac Lab's grid is centred there and the block is not.

    Args:
        num_envs: Total environments in the grid.
        env_spacing: Distance between neighbouring environment origins.
        block_size: Environments the block covers.
    """
    rows = math.floor(math.sqrt(num_envs))
    columns = math.ceil(num_envs / rows)
    block_rows = min(rows, math.ceil(math.sqrt(block_size)))
    block_columns = min(columns, math.ceil(block_size / block_rows))
    azimuth = math.radians(CAMERA_AZIMUTH_DEG)
    return (
        math.copysign((rows - block_rows) / 2 * env_spacing, math.cos(azimuth)),
        math.copysign((columns - block_columns) / 2 * env_spacing, math.sin(azimuth)),
    )


def _camera_offset(distance: float, pitch: float) -> tuple[float, float, float]:
    """Return where the eye sits relative to what it looks at.

    Args:
        distance: How far the eye is from the point it looks at, in metres.
        pitch: Angle below horizontal that the camera looks along, in radians.
    """
    azimuth = math.radians(CAMERA_AZIMUTH_DEG)
    ground = distance * math.cos(pitch)
    return ground * math.cos(azimuth), ground * math.sin(azimuth), distance * math.sin(pitch)


def _camera_basis(forward: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    """Return the right and up axes of a camera looking along ``forward`` with the horizon level.

    Args:
        forward: Unit vector from the eye towards what it looks at.
    """
    length = math.hypot(forward[1], forward[0])
    right = (forward[1] / length, -forward[0] / length, 0.0)
    up = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    return right, up


def _frame_reach(span: float, distance: float, pitch: float, aim: float) -> tuple[float, float, float]:
    """Return how far a block of environments reaches into the frame, in units of the frame's depth.

    A value of ``math.tan(fov / 2)`` is exactly the frame's edge.

    Args:
        span: Side of the square block, in metres.
        distance: How far the eye is from the block's centre, in metres.
        pitch: Angle below horizontal that the camera looks along, in radians.
        aim: How far above the block's centre the camera points, along its own up axis, in metres.

    Returns:
        The furthest the block reaches to either side, then the furthest it reaches down the frame
        and the furthest up. The two vertical reaches are separate because a block seen along its
        diagonal projects lopsided: its near corner falls much further below the point the camera
        aims at than its far corner rises above it.
    """
    eye = _camera_offset(distance, pitch)
    _, level_up = _camera_basis(tuple(-component / distance for component in eye))
    target = tuple(aim * component for component in level_up)
    to_target = [component - origin for component, origin in zip(target, eye)]
    length = math.sqrt(sum(component**2 for component in to_target))
    forward = tuple(component / length for component in to_target)
    right, up = _camera_basis(forward)

    across, lowest, highest = 0.0, 0.0, 0.0
    half_span = span / 2.0
    for corner_x in (-half_span, half_span):
        for corner_y in (-half_span, half_span):
            for corner_z in (-SCENE_HEIGHT / 2.0, SCENE_HEIGHT / 2.0):
                corner = (corner_x, corner_y, corner_z)
                offset = [position - origin for position, origin in zip(corner, eye)]
                depth = sum(a * b for a, b in zip(offset, forward))
                assert depth > 0.0, f"the block reaches behind a camera {distance} m from its centre"
                vertical = sum(a * b for a, b in zip(offset, up)) / depth
                across = max(across, abs(sum(a * b for a, b in zip(offset, right))) / depth)
                lowest, highest = min(lowest, vertical), max(highest, vertical)
    return across, lowest, highest


def _aim_above_centre(span: float, distance: float, pitch: float) -> float:
    """Return how far above a block's centre to point, so that its projection sits centred in frame.

    Args:
        span: Side of the square block, in metres.
        distance: How far the eye is from the block's centre, in metres.
        pitch: Angle below horizontal that the camera looks along, in radians.
    """
    # Bisected rather than stepped towards, because aiming higher moves the block's near corner down
    # the frame further than it moves the far corner, by the ratio of their distances, so a step of
    # the size the imbalance suggests overshoots and the correction runs away.
    below, above = -distance / 3.0, distance / 3.0
    for _ in range(40):
        aim = (below + above) / 2.0
        _, lowest, highest = _frame_reach(span, distance, pitch, aim)
        below, above = (aim, above) if lowest + highest > 0.0 else (below, aim)
    return (below + above) / 2.0


def _width_filling_distance(span: float, pitch: float, half_frame_x: float) -> float:
    """Return how far back a camera sits for a block of environments to fill the width of its frame.

    Found by projecting the block's corners and bisecting, rather than by scaling its size: seen
    along its diagonal the block recedes far enough that its near and far corners sit at very
    different distances, and treating it as flat misjudges the fit by nearly half.

    Only the width is fitted. Whether the overrun down the frame is then clipped or framed depends on
    the camera: a fixed lens on a widescreen frame has to clip, because seen along its diagonal from a
    steep angle a square grid projects almost square -- the foreshortening that should squash it
    vertically is undone by the near corner sitting much closer to the eye than the far one, and no
    pitch shallow enough to keep the roofless rooms visible stretches that into a widescreen shape.

    Args:
        span: Side of the block, in metres.
        pitch: Angle below horizontal that the camera looks along, in radians.
        half_frame_x: Tangent of half the camera's horizontal field of view.
    """

    def fits(distance: float) -> bool:
        across, _, _ = _frame_reach(span, distance, pitch, _aim_above_centre(span, distance, pitch))
        return across <= half_frame_x

    near, far = span, 10.0 * span + 100.0
    assert fits(far), f"a block {span} m across does not fit even from {far} m"
    for _ in range(40):
        middle = (near + far) / 2.0
        near, far = (near, middle) if fits(middle) else (middle, far)
    return far


def _block_view(
    num_envs: int,
    env_spacing: float,
    framed: int,
    pitch: float,
    half_frame_x: float,
    centre: tuple[float, float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return where a camera stands and what it points at, to fill its frame with ``framed`` rooms.

    Args:
        num_envs: Total environments in the grid.
        env_spacing: Distance between neighbouring environment origins.
        framed: Environments the camera frames.
        pitch: Angle below horizontal that the camera looks along, in radians.
        half_frame_x: Tangent of half the camera's horizontal field of view.
        centre: What the framed block is centred on, in world coordinates. Defaults to the corner of
            the grid nearest the camera, which keeps the environments it is not framing behind the
            block rather than across the shot.

    Returns:
        The eye position and the point it looks at, both in world coordinates, and how far the eye
        stands from the block's centre.
    """
    span = math.ceil(math.sqrt(framed)) * env_spacing
    centre_x, centre_y = centre if centre is not None else _environment_block_centre(num_envs, env_spacing, framed)
    distance = FRAMING_MARGIN * _width_filling_distance(span, pitch, half_frame_x)
    aim = _aim_above_centre(span, distance, pitch)
    offset = _camera_offset(distance, pitch)
    _, level_up = _camera_basis(tuple(-component / distance for component in offset))
    eye = (centre_x + offset[0], centre_y + offset[1], offset[2])
    lookat = (centre_x + aim * level_up[0], centre_y + aim * level_up[1], aim * level_up[2])
    return eye, lookat, distance


def _world_convention_quaternion(
    eye: tuple[float, float, float], lookat: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """Return the orientation of a camera at ``eye`` looking at ``lookat``, as ``(w, x, y, z)``.

    In Isaac Lab's ``world`` camera convention the camera looks along its own +x with +z up, so the
    rotation is just the camera's axes written as the columns of a matrix.

    Args:
        eye: Where the camera stands, in world coordinates.
        lookat: The point it looks at, in world coordinates.
    """
    separation = [target - origin for target, origin in zip(lookat, eye)]
    length = math.sqrt(sum(component**2 for component in separation))
    assert length > 0.0, "a camera cannot look at the point it stands on"
    forward = tuple(component / length for component in separation)
    right, up = _camera_basis(forward)
    left = tuple(-component for component in right)
    trace = forward[0] + left[1] + up[2]
    assert trace > -1.0, f"a camera looking along {forward} turns further than this conversion covers"
    w = math.sqrt(1.0 + trace) / 2.0
    return (
        w,
        (up[1] - left[2]) / (4.0 * w),
        (forward[2] - up[0]) / (4.0 * w),
        (left[0] - forward[1]) / (4.0 * w),
    )


def _overview_view(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg, framed: int, pitch: float):
    """Return a view that frames ``framed`` rooms closely enough to make out the task.

    The recorded scene video renders the viewer, which frames the whole grid at a size where a room is
    a few dozen pixels. This is a second, closer view of the same run.

    Taken from the middle of the grid rather than its near corner. The rooms this leaves out then lie
    on all sides instead of behind, which costs a little of the frame to their near edges but shows
    the block as part of the field it belongs to rather than as an outlying corner of it.

    Args:
        env_cfg: The environment configuration the grid's shape is read from.
        framed: Environments the view frames.
        pitch: Angle below horizontal that the camera looks along, in radians.
    """
    from isaaclab_arena.video.scene_view_video_recorder import SceneView

    eye, lookat, _ = _block_view(
        env_cfg.scene.num_envs,
        env_cfg.scene.env_spacing,
        framed,
        pitch,
        math.tan(math.radians(OVERVIEW_LENS_FOV_DEG) / 2.0),
        centre=(0.0, 0.0),
    )
    return SceneView(
        name=OVERVIEW_CAMERA_NAME,
        eye=eye,
        lookat=lookat,
        horizontal_fov_deg=OVERVIEW_LENS_FOV_DEG,
        width=OVERVIEW_RESOLUTION[0],
        height=OVERVIEW_RESOLUTION[1],
    )


def _size_physics_buffers(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> None:
    """Enlarge the GPU contact buffers to fit a cluttered table in every environment.

    A generated scene puts a whole table setting in front of each robot, and dozens of resting props
    per environment generate far more contacts than PhysX's default buffers hold. PhysX responds to
    the overflow by dropping contacts, which shows up as props sinking through the table they were
    resting on -- so this is a correctness setting, not a performance one.

    Sized for a hundred-plus environments of SimReady tableware, whose SDF colliders are what make
    the stack this large. Trading them for convex decompositions would cost a tenth of the memory but
    leaves plates and bowls unable to rest against each other.

    Args:
        env_cfg: The environment configuration to modify in place.
    """
    from isaaclab_physx.physics import PhysxCfg

    if env_cfg.sim.physics is None:
        env_cfg.sim.physics = PhysxCfg()
    env_cfg.sim.physics.gpu_collision_stack_size = 3 * 2**30
    # A hundred and twenty-eight tables of SDF tableware asked PhysX for 8.76M contacts against the
    # 8.39M that 2**23 provides, and the overflow is silent apart from one log line: the contacts
    # past the limit are simply dropped, so a plate loses the surface under it mid-episode. Doubling
    # leaves room for the densest variations the pool produces rather than sitting a few percent
    # under the peak.
    env_cfg.sim.physics.gpu_max_rigid_contact_count = 2**24
    env_cfg.sim.physics.gpu_max_rigid_patch_count = 2**24


def _expose_for_lit_interiors(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> None:
    """Let the renderer meter the shot instead of rendering a lit room as white.

    A generated interior is lit at physical intensities -- a sky dome for fill, a sun two orders of
    magnitude above it, and lamps indoors -- and the offscreen path renders at a fixed exposure, so a
    capture of a sunlit kitchen comes back with the walls, the worktops and the casework all clipped to
    the same white. That is a metering problem, not a lighting one: the same scene tone-maps correctly
    in an offline render, and dimming the lights to suit one renderer would break the other.

    Args:
        env_cfg: The environment configuration to modify in place.
    """
    settings = dict(env_cfg.sim.render.carb_settings or {})
    settings["/rtx/post/histogram/enabled"] = True
    env_cfg.sim.render.carb_settings = settings


def frame_environments(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    framed_env_count: int | None = None,
    sensor_tile_video: bool = False,
    overview_env_count: int | None = None,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Run at 50 Hz, size the physics buffers for the clutter, and frame the environments to watch.

    Recorded training video renders the viewer camera, so this is what decides whether a capture shows
    the whole grid or a legible handful of environments.

    Args:
        env_cfg: The environment configuration to modify in place.
        framed_env_count: Environments the viewer frames. ``None`` frames the whole grid.
        sensor_tile_video: Also record the robot's own cameras, tiled across environments.
        overview_env_count: Also record a close view of this many rooms, from a camera of our own.
            ``None`` records only the viewer's wide shot.

    Returns:
        The same configuration, so this can be used directly as an ``env_cfg_callback``.
    """
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        SceneViewVideoCfg,
        SensorTileVideoCfg,
        set_control_rate_50hz,
    )

    set_control_rate_50hz(env_cfg)
    if sensor_tile_video:
        env_cfg.sensor_tile_video = SensorTileVideoCfg()
    _size_physics_buffers(env_cfg)
    _expose_for_lit_interiors(env_cfg)
    num_envs, spacing = env_cfg.scene.num_envs, env_cfg.scene.env_spacing
    framed = min(num_envs, framed_env_count or num_envs)
    pitch = math.radians(CAMERA_PITCH_DEG)
    if overview_env_count:
        env_cfg.scene_view_video = SceneViewVideoCfg(
            views=(_overview_view(env_cfg, min(num_envs, overview_env_count), pitch),)
        )
    eye, lookat, _ = _block_view(
        num_envs, spacing, framed, pitch, math.tan(math.radians(KIT_PERSPECTIVE_FOV_DEG) / 2.0)
    )
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = eye
    env_cfg.viewer.lookat = lookat
    env_cfg.viewer.resolution = VIDEO_RESOLUTION
    # The recorder keeps its own camera and frame size, so framing the viewer alone would leave a
    # recorded video pointing somewhere else at a lower resolution.
    env_cfg.video_recorder.eye = eye
    env_cfg.video_recorder.lookat = lookat
    env_cfg.video_recorder.window_width, env_cfg.video_recorder.window_height = VIDEO_RESOLUTION
    return env_cfg


@register_environment
class LiftObjectLiveEnvironment(ArenaEnvironmentFactory[LiftObjectLiveEnvironmentCfg]):
    """Lift an object out of a generated scene whose layout is redrawn from a pool on every reset."""

    name: str = "lift_object_live"
    _legacy_argparse_cfg_type = LiftObjectLiveEnvironmentCfg

    def build(self, cfg: LiftObjectLiveEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from pathlib import Path

        import isaaclab_arena_examples.policy.base_rsl_rl_policy as base_rsl_rl_policy
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.live_scene.generated_background import (
            GENERATED_SCENE_BACKGROUND_REGISTRY_NAME,
            build_object_references,
        )
        from isaaclab_arena.live_scene.pool_manifest import load_pool_manifest, resolve_scene_usd
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.lift_object_task import LiftObjectTaskRL

        assert (
            cfg.live_variation_pool_dir is not None
        ), "lift_object_live builds its scene from a generated variation pool; pass --live_variation_pool_dir <dir>"
        pool_dir = Path(cfg.live_variation_pool_dir)
        manifest = load_pool_manifest(pool_dir)

        background = self.asset_registry.get_asset_by_name(GENERATED_SCENE_BACKGROUND_REGISTRY_NAME)(
            usd_path=str(resolve_scene_usd(pool_dir, manifest))
        )
        references = build_object_references(manifest, background)
        lift_object = _select_lift_prim(references, cfg.lift_prim)

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.embodiment_cameras, concatenate_observation_terms=True
        )

        # The generated scene carries its own floor and lighting, so only the robot is added to it.
        scene = Scene(assets=[background, *references])

        task = LiftObjectTaskRL(
            lift_object,
            background,
            embodiment,
            minimum_height_to_lift=0.04,
            episode_length_s=5.0,
            rl_training_mode=cfg.rl_training_mode,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=f"{base_rsl_rl_policy.__name__}:RLPolicyCfg",
            env_cfg_callback=functools.partial(frame_environments, framed_env_count=cfg.framed_env_count),
        )
