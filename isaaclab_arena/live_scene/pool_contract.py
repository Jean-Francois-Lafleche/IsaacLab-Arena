# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Versioned on-disk contract shared by the variation producer and Arena.

The producer writes into a *pool directory*::

    <pool>/manifest.json              # schema, prim inventory, shard layout
    <pool>/scene.usdc                 # base scene, superset of assets, topology frozen
    <pool>/variations/<seq>.poses.npy # hot path: float32 (n_variations, n_prims, 7)
    <pool>/variations/<seq>.parked.npy# bool (n_variations, n_prims), validation only
    <pool>/layers/<seq>/<id>.usdc     # archival variation layers, never read at runtime

Arena reads only ``manifest.json`` and the ``variations/`` shards at runtime. The archival
layers exist for replay and inspection outside Arena.

Both files of a shard are written to a ``.tmp`` sibling, fsynced, then renamed. ``parked`` is
renamed before ``poses``, so the presence of ``<seq>.poses.npy`` means the whole shard is
readable. Readers must therefore discover shards by globbing :data:`POSES_SUFFIX`.

This module imports nothing beyond the standard library so the producer can vendor it and both
sides stay byte-compatible. In particular it must never import ``pxr``, ``numpy``, or ``torch``.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = 1
"""Bumped whenever the layout below changes incompatibly. Arena refuses a pool it cannot read."""

MANIFEST_FILENAME = "manifest.json"
DEFAULT_SCENE_FILENAME = "scene.usdc"
VARIATIONS_DIRNAME = "variations"
LAYERS_DIRNAME = "layers"

POSES_SUFFIX = ".poses.npy"
"""Written last, so its presence marks a shard as complete."""

PARKED_SUFFIX = ".parked.npy"
TEMP_SUFFIX = ".tmp"

POSE_COLUMNS = 7
"""One pose row is 3 position components followed by 4 quaternion components."""

QUATERNION_ORDER = "xyzw"
"""Matches :meth:`isaaclab_arena.utils.pose.Pose.to_tensor`, so a shard row is written to the
simulator with no reordering. The manifest repeats this so a producer using a different
convention fails loudly instead of silently mirroring every rotation."""

POSE_DTYPE = "float32"
PARKED_DTYPE = "bool"


def manifest_path(pool_dir: str | Path) -> Path:
    """Return the manifest path for a pool directory."""
    return Path(pool_dir) / MANIFEST_FILENAME


def variations_dir(pool_dir: str | Path) -> Path:
    """Return the directory holding the hot-path shards."""
    return Path(pool_dir) / VARIATIONS_DIRNAME


def layers_dir(pool_dir: str | Path) -> Path:
    """Return the directory holding the archival variation layers."""
    return Path(pool_dir) / LAYERS_DIRNAME


def shard_poses_path(pool_dir: str | Path, sequence: int) -> Path:
    """Return the poses file path for shard ``sequence``."""
    return variations_dir(pool_dir) / f"{sequence:08d}{POSES_SUFFIX}"


def shard_parked_path(pool_dir: str | Path, sequence: int) -> Path:
    """Return the parked-mask file path for shard ``sequence``."""
    return variations_dir(pool_dir) / f"{sequence:08d}{PARKED_SUFFIX}"


def parked_path_for_poses(poses_path: str | Path) -> Path:
    """Return the parked-mask path that accompanies a poses file."""
    poses_path = Path(poses_path)
    assert poses_path.name.endswith(POSES_SUFFIX), f"{poses_path.name!r} is not a {POSES_SUFFIX} file"
    return poses_path.with_name(poses_path.name[: -len(POSES_SUFFIX)] + PARKED_SUFFIX)


def shard_sequence_from_path(poses_path: str | Path) -> int:
    """Return the sequence number encoded in a poses filename.

    Sequence numbers order ingestion, so a reader can consume shards in the order they were
    produced regardless of filesystem listing order.
    """
    poses_path = Path(poses_path)
    assert poses_path.name.endswith(POSES_SUFFIX), f"{poses_path.name!r} is not a {POSES_SUFFIX} file"
    stem = poses_path.name[: -len(POSES_SUFFIX)]
    assert stem.isdigit(), f"shard {poses_path.name!r} does not start with a numeric sequence"
    return int(stem)
