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
    """Return the centre of the first ``block_size`` environments, in world coordinates.

    Mirrors Isaac Lab's grid layout, where environment 0 sits at the high-x, low-y corner rather than
    at the origin, so a camera aimed at the origin would miss the block entirely.

    Args:
        num_envs: Total environments in the grid.
        env_spacing: Distance between neighbouring environment origins.
        block_size: Environments the block covers.
    """
    rows = math.floor(math.sqrt(num_envs))
    columns = math.ceil(num_envs / rows)
    block_rows = min(rows, math.ceil(math.sqrt(block_size)))
    block_columns = min(columns, math.ceil(block_size / block_rows))
    # Environment (i, j) sits at x = -(i - (rows - 1) / 2) * spacing, y = (j - (columns - 1) / 2) * spacing.
    centre_x = -((block_rows - 1) / 2 - (rows - 1) / 2) * env_spacing
    centre_y = ((block_columns - 1) / 2 - (columns - 1) / 2) * env_spacing
    return centre_x, centre_y


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
    env_cfg.sim.physics.gpu_max_rigid_contact_count = 2**23
    env_cfg.sim.physics.gpu_max_rigid_patch_count = 2**23


def _frame_environments(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg, framed_env_count: int | None = None
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Run at 50 Hz, size the physics buffers for the clutter, and frame the environments to watch.

    Recorded training video renders the viewer camera, so this is what decides whether a capture shows
    the whole grid or a legible handful of environments.

    Args:
        env_cfg: The environment configuration to modify in place.
        framed_env_count: Environments to frame. ``None`` frames the whole grid.

    Returns:
        The same configuration, so this can be used directly as an ``env_cfg_callback``.
    """
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz

    set_control_rate_50hz(env_cfg)
    _size_physics_buffers(env_cfg)
    num_envs, spacing = env_cfg.scene.num_envs, env_cfg.scene.env_spacing
    framed = min(num_envs, framed_env_count or num_envs)
    span = math.ceil(math.sqrt(framed)) * spacing
    centre_x, centre_y = _environment_block_centre(num_envs, spacing, framed)
    lookat = (centre_x, centre_y, 0.0)
    # Steep and close. A shallow angle would frame the block correctly and still show the rest of the
    # grid receding behind it, because the block sits at a corner and the camera looks back inwards.
    eye = (centre_x + 0.45 * span, centre_y - 0.45 * span, 0.85 * span)
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
            env_cfg_callback=functools.partial(_frame_environments, framed_env_count=cfg.framed_env_count),
        )
