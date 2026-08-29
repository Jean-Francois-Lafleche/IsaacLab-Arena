# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Take a mug out of a kitchen drawer and put it on the countertop, in a kitchen redrawn every reset.

How far the drawer starts open is the pool's difficulty, drawn per variation and published as the
``drawer_openness`` scalar; the pool's manifest also says which joint that drives. The mug is carried
with the drawer, so it starts inside whatever the openness works out to.

By default a curriculum takes the scalar over, starting every drawer fully open and closing it as the
policy earns it: nothing about a shut drawer rewards a policy that has never opened one. Pass
``--live_variation_scalar_override drawer_openness=X`` to pin one difficulty instead, or set
``anneal_difficulty=False`` to train on the pool's own draw.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena_environments.lift_object_live_environment import frame_environments

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.live_scene.pool_manifest import PoolManifest

PICK_OBJECT_ROLE = "pick_object"
"""Manifest role of the prim the task retrieves."""

WORKTOP_Z = 0.0
"""Height of the kitchen worktop in the env-local frame.

The generated scene drops its floor by exactly the height of a base unit, so the worktop lands level
with the arm's own base plane. That convention is what lets the task name a destination height without
measuring the scene.
"""

FLOOR_Z = -0.9
"""Env-local height of the kitchen floor, one base unit below the worktop."""

GROUND_Z = FLOOR_Z - 0.005 - 0.1
"""Centre of the slab that fills the gaps between rooms, for a slab a fifth of a metre thick.

Five millimetres under the kitchen floors rather than level with them, because two opaque faces at
the same height z-fight.
"""

GOAL_X = 0.5
"""Where across the worktop's depth the mug is to be set down, in the env-local frame.

Towards the front edge rather than the middle: the arm has to reach over an open drawer to get there,
and the back of the counter is behind the drawer it came out of.
"""


@dataclass
class PickPlaceKitchenLiveEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the kitchen mug environment on a generated scene."""

    live_variation_pool_dir: str | None = None
    """Set by the shared Arena parser; the pool supplies both the scene and its per-reset variations."""

    embodiment: str = "franka_joint_pos"
    """Use joint control by default because it is more reliable than IK for RL training."""

    embodiment_cameras: bool = False
    """Spawn the robot's own cameras, independently of the app-level camera flag.

    Implied by ``sensor_tile_video``, which has nothing to record without them.
    """

    sensor_tile_video: bool = False
    """Record what the robot's own cameras see, as one video per camera tiled across environments.

    Separate from the scene video, which shows the rooms from outside: this is the policy's own view.
    """

    framed_env_count: int | None = None
    """How many environments the viewer and any recording frame. ``None`` frames all of them."""

    overview_env_count: int | None = None
    """Also record a close view of this many rooms, alongside the viewer's wide shot.

    The wide shot renders a room a few dozen pixels across, which shows the grid training but not the
    task. ``None`` records only the wide shot.
    """

    anneal_difficulty: bool = True
    """Pin the drawer fully open and close it as the policy succeeds, rather than taking the pool's draw.

    Yields to ``--live_variation_scalar_override drawer_openness=X``, which is the more specific
    instruction of the two.
    """

    rl_training_mode: bool = False


def _prims_with_role(manifest: PoolManifest, role: str) -> list[str]:
    """Return the inventory paths carrying a role, in manifest order.

    Args:
        manifest: Pool manifest to read.
        role: Role to select.
    """
    paths = [prim.path for prim in manifest.prims if prim.role == role]
    assert (
        paths
    ), f"the variation pool has no prim with role {role!r}; it has {sorted({prim.role for prim in manifest.prims})}"
    return paths


@register_environment
class PickPlaceKitchenLiveEnvironment(ArenaEnvironmentFactory[PickPlaceKitchenLiveEnvironmentCfg]):
    """Retrieve a mug from a drawer in a kitchen drawn fresh from a variation pool on every reset."""

    name: str = "pick_place_kitchen_live"
    _legacy_argparse_cfg_type = PickPlaceKitchenLiveEnvironmentCfg

    def build(self, cfg: PickPlaceKitchenLiveEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from pathlib import Path

        import isaaclab_arena_examples.policy.base_rsl_rl_policy as base_rsl_rl_policy
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.live_scene.generated_background import (
            GENERATED_SCENE_BACKGROUND_REGISTRY_NAME,
            build_object_references,
        )
        from isaaclab_arena.live_scene.pool_manifest import load_pool_manifest, resolve_scene_usd
        from isaaclab_arena.live_scene.scene_entities import object_reference_name
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.retrieve_from_drawer_task import RetrieveFromDrawerTaskRL
        from isaaclab_arena.utils.pose import Pose

        assert cfg.live_variation_pool_dir is not None, (
            "pick_place_kitchen_live builds its scene from a generated variation pool; "
            "pass --live_variation_pool_dir <dir>"
        )
        pool_dir = Path(cfg.live_variation_pool_dir)
        manifest = load_pool_manifest(pool_dir)

        background = self.asset_registry.get_asset_by_name(GENERATED_SCENE_BACKGROUND_REGISTRY_NAME)(
            usd_path=str(resolve_scene_usd(pool_dir, manifest))
        )
        # The pool's own joint declarations say which prim carries which joint, so the openable
        # references follow from the manifest rather than from a name written down here.
        openable = {path: joint.joint_name for joint in manifest.scalar_joints for path in joint.prims}
        references = build_object_references(manifest, background, openable_joint_names=openable)

        mug_paths = _prims_with_role(manifest, PICK_OBJECT_ROLE)
        assert len(mug_paths) == 1, f"expected one prim to pick, found {mug_paths}"
        by_path = {prim.path: reference for prim, reference in zip(manifest.prims, references)}
        mug = by_path[mug_paths[0]]

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.embodiment_cameras or cfg.sensor_tile_video, concatenate_observation_terms=True
        )
        # The recorder reads the cameras off the scene, so recording alone does not put them in front
        # of the policy: an observation group the policy was not trained with would change the run.
        embodiment.camera_observations = cfg.embodiment_cameras

        # The generated kitchen carries its own floor, walls and lighting. What it does not carry is
        # anything between one room and the next: the rooms are narrower than the environment spacing,
        # and a camera high enough to show the grid looks through those gaps at the parked props
        # twenty-five metres below. One slab across the whole grid closes that view.
        ground = self.asset_registry.get_asset_by_name("visual_ground_plane")(
            initial_pose=Pose(position_xyz=(0.0, 0.0, GROUND_Z))
        )
        scene = Scene(assets=[background, ground, *references])

        # One joint declaration, naming every candidate cabinet: the pool parks all but one of them
        # per variation, and the reward works out which per environment.
        assert (
            len(manifest.scalar_joints) == 1
        ), f"expected the pool to declare one scalar joint, found {len(manifest.scalar_joints)}"
        drawer = manifest.scalar_joints[0]
        task = RetrieveFromDrawerTaskRL(
            retrieved_object=mug,
            embodiment=embodiment,
            drawer_articulations=tuple(object_reference_name(path) for path in drawer.prims),
            drawer_joint_name=drawer.joint_name,
            drawer_travel=drawer.travel,
            surface_z=WORKTOP_Z,
            goal_x=GOAL_X,
            floor_z=FLOOR_Z,
            difficulty_scalar=drawer.scalar if cfg.anneal_difficulty else None,
            rl_training_mode=cfg.rl_training_mode,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=f"{base_rsl_rl_policy.__name__}:RLPolicyCfg",
            env_cfg_callback=functools.partial(
                frame_environments,
                framed_env_count=cfg.framed_env_count,
                sensor_tile_video=cfg.sensor_tile_video,
                overview_env_count=cfg.overview_env_count,
            ),
        )
