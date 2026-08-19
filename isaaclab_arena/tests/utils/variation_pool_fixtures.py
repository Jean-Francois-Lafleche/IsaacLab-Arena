# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build a variation pool on disk without the generator, so tests exercise only Arena's half.

The pool this produces is shaped exactly like a real one: a base scene whose topology is frozen, a
manifest whose ordered inventory fixes the shard column order, and shards written through the same
atomic path the producer uses.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.live_scene import pool_contract, shard_writer
from isaaclab_arena.live_scene.pool_manifest import PoolManifest, PoolPrimSpec, ShardLayout

FIXTURE_PARKING_OFFSET = (0.0, 0.0, -25.0)
"""Far enough below the floor that a parked prim is out of every camera frustum."""

FIXTURE_PRIM_PATHS = ("Props/mug_00", "Props/mug_01")
"""Two movable props, so a test can prove two environments receive different poses."""


def build_base_scene(usd_path: str | Path, prim_paths: tuple[str, ...] = FIXTURE_PRIM_PATHS) -> Path:
    """Write a base scene containing one rigid body per inventory prim.

    Args:
        usd_path: Destination file. The extension decides the USD format.
        prim_paths: Prim paths relative to the default prim.

    Returns:
        The written path.
    """
    # Imported lazily because unit tests build manifests and shards without ever touching USD.
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    usd_path = Path(usd_path)
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(usd_path.as_posix())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    for index, relative_path in enumerate(prim_paths):
        body = UsdGeom.Xform.Define(stage, f"/Root/{relative_path}")
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.25 * index, 0.5))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.2)
        mesh = UsdGeom.Cube.Define(stage, f"/Root/{relative_path}/geometry")
        mesh.CreateSizeAttr(0.05)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    assert stage.GetRootLayer().Save(), f"failed to write fixture base scene to {usd_path}"
    return usd_path


def build_manifest(
    prim_paths: tuple[str, ...] = FIXTURE_PRIM_PATHS,
    variations_per_shard: int = 2,
    num_envs_hint: int = 2,
    parkable_paths: tuple[str, ...] = (),
    scene_usd: str = pool_contract.DEFAULT_SCENE_FILENAME,
) -> PoolManifest:
    """Return a manifest describing the fixture scene.

    Args:
        prim_paths: Prim paths relative to the default prim, in shard column order.
        variations_per_shard: Rows the producer packs per shard.
        num_envs_hint: Environment count the scene's asset multiplicity was sized for.
        parkable_paths: Subset of ``prim_paths`` a variation may park.
        scene_usd: Base scene filename relative to the pool directory.

    Returns:
        A manifest with a matching inventory digest.
    """
    prims = [
        PoolPrimSpec(
            path=path,
            role="prop",
            object_type=ObjectType.RIGID,
            parkable=path in parkable_paths,
        )
        for path in prim_paths
    ]
    manifest = PoolManifest(
        schema_version=pool_contract.SCHEMA_VERSION,
        scene_usd=scene_usd,
        num_envs_hint=num_envs_hint,
        parking_offset_xyz=FIXTURE_PARKING_OFFSET,
        shard=ShardLayout(variations_per_shard=variations_per_shard),
        prims=prims,
    )
    manifest.inventory_digest = manifest.compute_inventory_digest()
    return manifest


def make_poses(
    manifest: PoolManifest,
    num_variations: int,
    first_variation_index: int = 0,
    parked_variation_indices: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic poses and a parked mask for a run of variations.

    Positions encode the global variation index and the prim's inventory position, so a test can read
    back which variation an environment received.

    Args:
        manifest: Manifest whose inventory fixes the column order.
        num_variations: Rows to generate.
        first_variation_index: Global index of the first row, which its positions encode.
        parked_variation_indices: Rows, relative to this run, whose parkable prims are parked.

    Returns:
        A ``(poses, parked)`` tuple.
    """
    num_prims = manifest.num_prims
    poses = np.zeros((num_variations, num_prims, pool_contract.POSE_COLUMNS), dtype=np.float32)
    parked = np.zeros((num_variations, num_prims), dtype=bool)
    parkable = set(manifest.parkable_indices())
    for row in range(num_variations):
        variation_index = first_variation_index + row
        for prim_index in range(num_prims):
            poses[row, prim_index, 0] = 0.1 * (variation_index + 1)
            poses[row, prim_index, 1] = 0.1 * (prim_index + 1)
            poses[row, prim_index, 2] = 0.5
            # Identity rotation in Arena's (x, y, z, w) order.
            poses[row, prim_index, 6] = 1.0
        if row in parked_variation_indices:
            for prim_index in parkable:
                parked[row, prim_index] = True
                poses[row, prim_index, :3] = np.asarray(manifest.parking_offset_xyz, dtype=np.float32)
    return poses, parked


def build_fixture_pool(
    pool_dir: str | Path,
    num_variations: int = 2,
    prim_paths: tuple[str, ...] = FIXTURE_PRIM_PATHS,
    parkable_paths: tuple[str, ...] = (),
    parked_variation_indices: tuple[int, ...] = (),
    with_base_scene: bool = False,
) -> PoolManifest:
    """Write a complete single-shard pool.

    Args:
        pool_dir: Directory to create the pool in.
        num_variations: Variations to pack into the one shard.
        prim_paths: Prim paths relative to the default prim, in shard column order.
        parkable_paths: Subset of ``prim_paths`` a variation may park.
        parked_variation_indices: Variations whose parkable prims are parked.
        with_base_scene: Also write the base scene USD. Requires ``pxr``, so sim tests opt in and
            unit tests leave it off.

    Returns:
        The manifest that was written.
    """
    pool_dir = Path(pool_dir)
    manifest = build_manifest(
        prim_paths=prim_paths,
        variations_per_shard=num_variations,
        parkable_paths=parkable_paths,
    )
    shard_writer.write_manifest(pool_dir, manifest)
    poses, parked = make_poses(
        manifest,
        num_variations,
        parked_variation_indices=parked_variation_indices,
    )
    shard_writer.write_shard(pool_dir, sequence=0, poses=poses, parked=parked)
    if with_base_scene:
        build_base_scene(pool_dir / manifest.scene_usd, prim_paths=prim_paths)
    return manifest


def append_fixture_shard(
    pool_dir: str | Path,
    manifest: PoolManifest,
    num_variations: int,
    first_variation_index: int,
) -> Any:
    """Append one more shard to an existing pool, as a running producer would.

    Args:
        pool_dir: Pool directory to append to.
        manifest: Manifest of that pool.
        num_variations: Rows to pack into the new shard.
        first_variation_index: Global index of the shard's first row.

    Returns:
        Path of the poses file that was written.
    """
    poses, parked = make_poses(manifest, num_variations, first_variation_index=first_variation_index)
    sequence = shard_writer.next_shard_sequence(pool_dir)
    return shard_writer.write_shard(pool_dir, sequence=sequence, poses=poses, parked=parked)


def variation_index_from_pose_row(pose_row: Any) -> int:
    """Return the variation index encoded in a pose row by :func:`make_poses`.

    Args:
        pose_row: A ``(7,)`` sequence of pose components.
    """
    return int(round(float(pose_row[0]) / 0.1)) - 1
