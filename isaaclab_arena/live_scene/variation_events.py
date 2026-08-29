# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reset event that applies a pooled scene variation to the resetting environments."""

from __future__ import annotations

import dataclasses
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.live_scene.scene_entities import object_reference_name
from isaaclab_arena.live_scene.support_placement import SupportPlacement, SupportPlacementHandle, place_on_supports
from isaaclab_arena.tasks.episode_start_pose import record_episode_start_poses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.live_scene.pool_manifest import PoolManifest
    from isaaclab_arena.live_scene.variation_pool import LiveVariationPool, LiveVariationPoolHandle

LIVE_VARIATION_RESET_EVENT_NAME = "live_variation_reset"
"""Name of the reset event term that owns the live variation pool."""


@dataclasses.dataclass
class ScalarDrivenJoint:
    """A joint a reset drives from one of the pool's per-variation scalars.

    A pose row cannot carry articulation state, so a pool publishes a scalar for it and the reset
    turns that into a joint position. ``override`` lets a curriculum take control of the value while
    still drawing everything else from the pool.

    One target names every articulation the scalar applies to, because a pool holds its alternatives
    as a parked superset: nine candidate cabinets exist and a variation places one. Driving the
    parked eight costs one batched write each and is invisible, whereas splitting them into nine
    targets would carry ``carried`` nine times over.

    Attributes:
        scalar: Pool scalar to read, which must be one the manifest declares.
        articulations: Scene entities holding the joint, all driven to the same value.
        joint_name: Joint to drive, resolved against each articulation once and cached.
        travel: Joint position at a scalar of 1. Metres for a prismatic joint, radians for a
            revolute one.
        axis_env_local: Unit direction the moving part travels in, in env-local coordinates. Only
            used to carry ``carried`` entities when an override moves the joint away from what the
            pool authored.
        carried: Scene entities the pool placed resting in or on the moving part. Their pooled poses
            were measured against the joint at the pool's own value, so an override has to move them
            by the same amount or they are left behind in mid-air.
        override: Value to use instead of the pool's, or ``None`` to follow the pool. Mutable at
            runtime: a curriculum reaches this object through the event term's params.
    """

    scalar: str
    articulations: tuple[str, ...]
    joint_name: str
    travel: float
    axis_env_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    carried: tuple[str, ...] = ()
    override: float | None = None
    _joint_ids: dict[str, list[int]] = dataclasses.field(default_factory=dict, init=False, repr=False)

    def joint_ids(self, name: str, articulation) -> list[int]:
        """Return an articulation's joint indices for this joint, resolving once per articulation.

        Args:
            name: Scene entity name, used as the cache key and in failure messages.
            articulation: The scene entity itself.
        """
        if name not in self._joint_ids:
            found, names = articulation.find_joints(self.joint_name)
            assert (
                len(found) == 1
            ), f"joint pattern {self.joint_name!r} matched {names} on {name}; a scalar drives exactly one joint"
            self._joint_ids[name] = list(found)
        return self._joint_ids[name]


def build_scalar_driven_joints(
    manifest: PoolManifest,
    overrides: dict[str, float] | None = None,
    scene_keys: set[str] | None = None,
) -> tuple[ScalarDrivenJoint, ...]:
    """Turn a manifest's joint declarations into reset targets.

    The pool describes its own joints, so an environment needs no per-task wiring: it only decides
    whether to pin a scalar. Resolved at build time so a stale declaration fails before the scene
    loads rather than at the first reset.

    Args:
        manifest: Manifest of the pool being consumed.
        overrides: Values to pin, keyed by scalar name. A name the pool does not publish is an error
            rather than a no-op, because silently ignoring it would leave a curriculum with no effect.
        scene_keys: Scene entity names available, for checking the declarations resolve. Skipped when
            omitted.

    Returns:
        One target per declared joint, in manifest order.
    """
    overrides = overrides or {}
    published = set(manifest.shard.scalars)
    unknown = sorted(set(overrides) - published)
    assert not unknown, f"cannot override scalars the pool does not publish: {unknown}; it has {sorted(published)}"

    targets: list[ScalarDrivenJoint] = []
    for joint in manifest.scalar_joints:
        articulations = tuple(object_reference_name(path) for path in joint.prims)
        carried = tuple(object_reference_name(path) for path in joint.carried)
        if scene_keys is not None:
            missing = sorted((set(articulations) | set(carried)) - scene_keys)
            assert (
                not missing
            ), f"the pool's {joint.joint_name!r} joint names prims the scene does not expose: {missing}"
        targets.append(
            ScalarDrivenJoint(
                scalar=joint.scalar,
                articulations=articulations,
                joint_name=joint.joint_name,
                travel=joint.travel,
                axis_env_local=joint.axis_env_local,
                carried=carried,
                override=overrides.get(joint.scalar),
            )
        )
    return tuple(targets)


def get_live_variation_pool(env) -> LiveVariationPool | None:
    """Return the variation pool stored on the env reset event, or ``None`` when absent.

    Lets a runtime caller reach the pool from the env alone, for example to read starvation counters.

    Args:
        env: The gym-wrapped Isaac Lab env; the base env is reached via ``env.unwrapped``.
    """
    try:
        term_cfg = env.unwrapped.event_manager.get_term_cfg(LIVE_VARIATION_RESET_EVENT_NAME)
    except ValueError:
        return None
    handle = term_cfg.params.get("variation_pool")
    assert handle is not None, f"'{LIVE_VARIATION_RESET_EVENT_NAME}' event is missing its variation_pool parameter."
    return handle.pool


def apply_scene_variation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    variation_pool: LiveVariationPoolHandle,
    write_plan: list[tuple[int, str]],
    scalar_driven_joints: tuple[ScalarDrivenJoint, ...] = (),
    support_placements: SupportPlacementHandle | None = None,
) -> None:
    """Draw one variation per resetting environment and write it to the simulator.

    The whole path is tensor writes: a gather out of the pool's device-resident ring buffer, one
    batched pose write per prim covering every resetting environment at once, and no USD authoring.
    Parked prims already carry the pool's parking offset, so they cost nothing extra here.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        variation_pool: Opaque handle to the live variation pool.
        write_plan: ``(inventory index, scene entity name)`` pairs for the prims Arena can write.
            Built once at build time so the reset path does no lookup or filtering.
        scalar_driven_joints: Joints to drive from the pool's per-variation scalars.
        support_placements: Handle to the objects Arena poses itself, on a surface the pool only
            names. These override whatever the pool authored for them.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    pool = variation_pool.pool
    num_scene_envs = env.scene.env_origins.shape[0]
    assert (
        pool.num_envs == num_scene_envs
    ), f"Variation pool has {pool.num_envs} envs, but scene has {num_scene_envs} env origins."

    reset_env_ids = env_ids.tolist()
    # One draw for both channels: two draws would advance the shared stream twice and hand an
    # environment a joint target belonging to a different variation than its poses.
    poses, scalars = pool.gather(reset_env_ids)
    poses = poses.to(device=env.device)
    scalars = scalars.to(device=env.device)

    poses = poses.clone()
    _carry_with_joints(pool, poses, scalars, write_plan, scalar_driven_joints)
    if support_placements is not None:
        _place_on_supports(pool, poses, scalars, support_placements.placements, scalar_driven_joints)

    # A task measuring progress against where an object started needs the spawn pose, and this is the
    # only place that has it without reading it back out of the simulator. One scatter of the whole
    # block keeps that off the per-prim loop below. Recorded after any carry, so it is where the
    # object actually starts rather than where the pool would have put it.
    record_episode_start_poses(env, env_ids, write_plan, poses)
    # Pool poses are env-local; the simulator wants world coordinates.
    poses[:, :, :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    zero_velocity = torch.zeros((len(reset_env_ids), 6), device=env.device)

    for inventory_index, scene_name in write_plan:
        scene_asset = env.scene[scene_name]
        scene_asset.write_root_pose_to_sim(poses[:, inventory_index, :], env_ids=env_ids)
        scene_asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    # After the root poses, matching the order Isaac Lab resets its own assets in: a joint is a pose
    # relative to a root that has just moved.
    _drive_joints(env, env_ids, pool, scalars, scalar_driven_joints)


def _place_on_supports(
    pool: LiveVariationPool,
    poses: torch.Tensor,
    scalars: torch.Tensor,
    support_placements: tuple[SupportPlacement, ...],
    scalar_driven_joints: tuple[ScalarDrivenJoint, ...],
) -> None:
    """Repose the objects Arena places itself, overwriting what the pool authored for them.

    Args:
        pool: Pool the variation came from, for resolving scalar columns.
        poses: ``(n_envs, n_prims, 7)`` env-local poses, modified in place.
        scalars: ``(n_envs, n_scalars)`` drawn alongside ``poses``.
        support_placements: Objects to place, each naming the surfaces it may rest on.
        scalar_driven_joints: Joints being driven, for the value a support's own joint ends at.
    """
    by_scalar = {target.scalar: target for target in scalar_driven_joints}
    for placement in support_placements:
        openness = None
        if placement.carry_scalar is not None:
            target = by_scalar.get(placement.carry_scalar)
            assert target is not None, (
                f"placement carries with the {placement.carry_scalar!r} scalar, which drives no "
                f"joint; the pool drives {sorted(by_scalar)}"
            )
            pooled = scalars[:, pool.scalar_column(placement.carry_scalar)]
            # The value the joint actually ends at, so the surface is measured where the drawer
            # stops rather than where the pool would have left it.
            openness = pooled if target.override is None else torch.full_like(pooled, float(target.override))
        place_on_supports(poses, placement, openness)


def _carry_with_joints(
    pool: LiveVariationPool,
    poses: torch.Tensor,
    scalars: torch.Tensor,
    write_plan: list[tuple[int, str]],
    scalar_driven_joints: tuple[ScalarDrivenJoint, ...],
) -> None:
    """Shift whatever the pool left resting in a joint that the caller is about to drive elsewhere.

    Mutates ``poses`` in place, before the caller records episode-start poses or writes anything, so
    a carried object is written once at its final position rather than moved after the fact.

    Args:
        pool: The variation pool, for resolving scalar names to columns.
        poses: ``(n_envs, n_prims, 7)`` env-local poses, modified in place.
        scalars: ``(n_envs, n_scalars)`` drawn alongside ``poses``.
        write_plan: ``(inventory index, scene entity name)`` pairs.
        scalar_driven_joints: Joints whose scalars may be overridden.
    """
    if not scalar_driven_joints:
        return
    index_of = {scene_name: inventory_index for inventory_index, scene_name in write_plan}
    for target in scalar_driven_joints:
        if target.override is None or not target.carried:
            # The pool measured a carried object against the joint at its own value, so only moving
            # the joint somewhere else leaves the object behind in mid-air.
            continue
        pooled = scalars[:, pool.scalar_column(target.scalar)]
        axis = torch.tensor(target.axis_env_local, dtype=poses.dtype, device=poses.device)
        shift = ((float(target.override) - pooled) * target.travel).unsqueeze(1) * axis
        for scene_name in target.carried:
            assert scene_name in index_of, (
                f"{scene_name!r} is carried by the {target.joint_name} joint but is not in the write "
                "plan, so the pool does not pose it and it cannot be carried"
            )
            poses[:, index_of[scene_name], :3] += shift


def _drive_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pool: LiveVariationPool,
    scalars: torch.Tensor,
    scalar_driven_joints: tuple[ScalarDrivenJoint, ...],
) -> None:
    """Set each scalar-driven joint to the value its environment drew, at rest.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        pool: The variation pool, for resolving scalar names to columns.
        scalars: ``(n_envs, n_scalars)`` drawn alongside the poses.
        scalar_driven_joints: Joints to drive.
    """
    for target in scalar_driven_joints:
        pooled = scalars[:, pool.scalar_column(target.scalar)]
        wanted = pooled if target.override is None else torch.full_like(pooled, float(target.override))
        position = (wanted * target.travel).unsqueeze(1)
        velocity = torch.zeros_like(position)
        for name in target.articulations:
            articulation = env.scene[name]
            articulation.write_joint_state_to_sim(
                position,
                velocity,
                joint_ids=target.joint_ids(name, articulation),
                env_ids=env_ids,
            )
