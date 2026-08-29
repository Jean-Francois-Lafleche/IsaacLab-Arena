# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Discover and memory-map variation shards.

This module is deliberately restricted to the filesystem and ``numpy``. Both release the GIL for
the work that matters here, so the ingest thread in
:mod:`isaaclab_arena.live_scene.variation_pool` genuinely overlaps the simulation thread. Importing
``pxr`` here would defeat that, because USD's Python bindings do not reliably release the GIL.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.live_scene import pool_contract
from isaaclab_arena.live_scene.pool_manifest import PoolManifest


@dataclass(frozen=True)
class VariationShard:
    """One shard's arrays, memory-mapped.

    Attributes are views into the mapped files, so slicing a row costs nothing until it is copied.
    """

    sequence: int
    """Producer-assigned ordering key."""

    poses: np.ndarray
    """``float32`` array of shape ``(n_variations, n_prims, 7)`` in env-local coordinates."""

    parked: np.ndarray
    """``bool`` array of shape ``(n_variations, n_prims)``, true where a prim is parked."""

    scalars: np.ndarray
    """``float32`` array of shape ``(n_variations, n_scalars)``, in the manifest's scalar order.
    Width zero for a pool that declares no scalars, so the shape is always usable."""

    @property
    def num_variations(self) -> int:
        """Rows on the first axis."""
        return int(self.poses.shape[0])


def discover_shard_paths(pool_dir: str | Path, after_sequence: int = -1) -> list[Path]:
    """Return complete shard poses files with a sequence above ``after_sequence``, oldest first.

    Only the poses file is globbed because the producer renames it last, so its presence means the
    whole shard is readable.

    Args:
        pool_dir: Directory the producer writes into.
        after_sequence: Return only shards newer than this sequence. Defaults to every shard.
    """
    directory = pool_contract.variations_dir(pool_dir)
    if not directory.is_dir():
        return []
    paths = [path for path in directory.glob(f"*{pool_contract.POSES_SUFFIX}") if path.is_file()]
    keyed = [(pool_contract.shard_sequence_from_path(path), path) for path in paths]
    return [path for sequence, path in sorted(keyed) if sequence > after_sequence]


def load_shard(poses_path: str | Path, manifest: PoolManifest) -> VariationShard:
    """Memory-map a shard and validate it against the manifest.

    Args:
        poses_path: Path to the shard's ``.poses.npy`` file.
        manifest: Manifest of the pool the shard belongs to.

    Returns:
        The mapped shard.
    """
    poses_path = Path(poses_path)
    parked_path = pool_contract.parked_path_for_poses(poses_path)
    assert parked_path.is_file(), (
        f"shard {poses_path.name} has no parked mask at {parked_path.name}; the producer must rename "
        "the parked file before the poses file"
    )

    poses = np.load(poses_path, mmap_mode="r")
    parked = np.load(parked_path, mmap_mode="r")

    expected_prims = manifest.num_prims
    assert poses.ndim == 3, f"shard {poses_path.name} poses must be 3-D, got shape {poses.shape}"
    assert poses.shape[1] == expected_prims, (
        f"shard {poses_path.name} is packed for {poses.shape[1]} prims but the manifest inventory has "
        f"{expected_prims}; a pool whose inventory changed needs a new pool directory"
    )
    assert (
        poses.shape[2] == pool_contract.POSE_COLUMNS
    ), f"shard {poses_path.name} poses must have {pool_contract.POSE_COLUMNS} columns, got {poses.shape[2]}"
    assert (
        poses.dtype == np.float32
    ), f"shard {poses_path.name} poses must be {pool_contract.POSE_DTYPE}, got {poses.dtype}"
    assert (
        parked.shape == poses.shape[:2]
    ), f"shard {poses_path.name} parked mask shape {parked.shape} does not match its poses {poses.shape[:2]}"
    assert parked.dtype == np.bool_, f"shard {poses_path.name} parked mask must be bool, got {parked.dtype}"

    expected_scalars = len(manifest.shard.scalars)
    if expected_scalars:
        scalars_path = pool_contract.scalars_path_for_poses(poses_path)
        assert scalars_path.is_file(), (
            f"shard {poses_path.name} declares scalars {manifest.shard.scalars} but has no "
            f"{scalars_path.name}; the producer must rename the scalars file before the poses file"
        )
        scalars = np.load(scalars_path, mmap_mode="r")
        assert scalars.shape == (poses.shape[0], expected_scalars), (
            f"shard {poses_path.name} scalars shape {scalars.shape} does not match "
            f"({poses.shape[0]}, {expected_scalars}) from its poses and the manifest"
        )
        assert (
            scalars.dtype == np.float32
        ), f"shard {poses_path.name} scalars must be {pool_contract.POSE_DTYPE}, got {scalars.dtype}"
    else:
        # An empty column block rather than None, so a consumer slices a row unconditionally.
        scalars = np.zeros((poses.shape[0], 0), dtype=np.float32)

    return VariationShard(
        sequence=pool_contract.shard_sequence_from_path(poses_path),
        poses=poses,
        parked=parked,
        scalars=scalars,
    )


def check_parked_poses_match_manifest(
    shard: VariationShard,
    manifest: PoolManifest,
    tolerance: float = 1e-4,
) -> None:
    """Assert every parked entry sits at the manifest's parking offset.

    Parking is what hides an inactive asset, so a parked prim left in the scene would be visible.
    This is a validation helper for tests and the producer's own gate, not part of the reset path.

    Args:
        shard: Shard to check.
        manifest: Manifest of the pool the shard belongs to.
        tolerance: Allowed absolute deviation per position component.
    """
    parked_rows = np.asarray(shard.parked)
    if not parked_rows.any():
        return
    positions = np.asarray(shard.poses)[..., :3]
    expected = np.asarray(manifest.parking_offset_xyz, dtype=np.float32)
    deviation = np.abs(positions[parked_rows] - expected)
    worst = float(deviation.max())
    assert worst <= tolerance, (
        f"shard {shard.sequence} has parked prims away from the manifest parking offset "
        f"{manifest.parking_offset_xyz} by up to {worst}"
    )


def unparkable_violations(shard: VariationShard, manifest: PoolManifest) -> list[str]:
    """Return descriptions of prims parked despite not being marked parkable.

    Args:
        shard: Shard to check.
        manifest: Manifest of the pool the shard belongs to.
    """
    parked_rows = np.asarray(shard.parked)
    violations: list[str] = []
    for index, prim in enumerate(manifest.prims):
        if prim.parkable:
            continue
        offending = np.nonzero(parked_rows[:, index])[0]
        if offending.size:
            violations.append(
                f"prim {prim.path!r} is not parkable but is parked in variations {offending.tolist()[:8]}"
            )
    return violations
