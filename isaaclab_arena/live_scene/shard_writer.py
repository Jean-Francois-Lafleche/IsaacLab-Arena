# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write variation shards and manifests into a pool directory.

Lives in Arena so the read and write halves of the contract sit next to each other and are tested
together. The variation producer uses the same functions, either by importing Arena or by vendoring
this module alongside :mod:`isaaclab_arena.live_scene.pool_contract`.

Every write lands on a ``.tmp`` sibling, is fsynced, then renamed. Renames are atomic on POSIX, so a
reader never observes a partial shard. The poses file is renamed last, which is what lets a reader
treat it as the readiness marker for the whole shard.
"""

from __future__ import annotations

import hashlib
import numpy as np
import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from isaaclab_arena.live_scene import pool_contract
from isaaclab_arena.live_scene.pool_manifest import PoolManifest


def _write_atomic(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Write through a temporary sibling and rename into place.

    Args:
        path: Final destination.
        write: Callback receiving the open binary handle for the temporary file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + pool_contract.TEMP_SUFFIX)
    with open(temp_path, "wb") as handle:
        write(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def write_manifest(pool_dir: str | Path, manifest: PoolManifest) -> Path:
    """Write a manifest into a pool directory.

    Args:
        pool_dir: Directory to write into.
        manifest: Manifest to serialize.

    Returns:
        The manifest path.
    """
    path = pool_contract.manifest_path(pool_dir)
    payload = manifest.to_json().encode()
    _write_atomic(path, lambda handle: handle.write(payload))
    return path


def write_shard(
    pool_dir: str | Path,
    sequence: int,
    poses: np.ndarray,
    parked: np.ndarray | None = None,
    scalars: np.ndarray | None = None,
) -> Path:
    """Write one shard's arrays into a pool directory.

    Args:
        pool_dir: Directory to write into.
        sequence: Ordering key; readers consume shards in ascending sequence.
        poses: ``float32`` array of shape ``(n_variations, n_prims, 7)`` in env-local coordinates.
        parked: ``bool`` array of shape ``(n_variations, n_prims)``. Defaults to all false.
        scalars: ``float32`` array of shape ``(n_variations, n_scalars)``, in the manifest's scalar
            order. ``None`` writes no scalars file, for a pool that declares none.

    Returns:
        Path of the poses file, which is written last.
    """
    poses = np.ascontiguousarray(poses, dtype=np.float32)
    assert poses.ndim == 3, f"poses must be 3-D, got shape {poses.shape}"
    assert (
        poses.shape[2] == pool_contract.POSE_COLUMNS
    ), f"poses must have {pool_contract.POSE_COLUMNS} columns, got {poses.shape[2]}"
    if parked is None:
        parked = np.zeros(poses.shape[:2], dtype=bool)
    parked = np.ascontiguousarray(parked, dtype=bool)
    assert parked.shape == poses.shape[:2], f"parked mask shape {parked.shape} does not match poses {poses.shape[:2]}"

    # Companions first: the poses file is the readiness marker, so it must land last.
    parked_path = pool_contract.shard_parked_path(pool_dir, sequence)
    _write_atomic(parked_path, lambda handle: np.save(handle, parked, allow_pickle=False))
    if scalars is not None:
        scalars = np.ascontiguousarray(scalars, dtype=np.float32)
        assert scalars.ndim == 2, f"scalars must be 2-D, got shape {scalars.shape}"
        assert scalars.shape[0] == poses.shape[0], f"scalars has {scalars.shape[0]} rows but poses has {poses.shape[0]}"
        _write_atomic(
            pool_contract.shard_scalars_path(pool_dir, sequence),
            lambda handle: np.save(handle, scalars, allow_pickle=False),
        )
    poses_path = pool_contract.shard_poses_path(pool_dir, sequence)
    _write_atomic(poses_path, lambda handle: np.save(handle, poses, allow_pickle=False))
    return poses_path


def next_shard_sequence(pool_dir: str | Path) -> int:
    """Return the sequence a new shard should use, one past the highest present.

    Args:
        pool_dir: Directory to inspect.
    """
    directory = pool_contract.variations_dir(pool_dir)
    if not directory.is_dir():
        return 0
    sequences = [
        pool_contract.shard_sequence_from_path(path) for path in directory.glob(f"*{pool_contract.POSES_SUFFIX}")
    ]
    return max(sequences) + 1 if sequences else 0


def file_digest(path: str | Path) -> str:
    """Return the sha256 hex digest of a file.

    Args:
        path: File to hash.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
