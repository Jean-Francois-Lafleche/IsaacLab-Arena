# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schema for a variation pool's ``manifest.json``.

Kept free of ``pxr``, ``numpy``, and ``torch`` imports so it can be read before Isaac Sim starts
and vendored by the producer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.live_scene import pool_contract


class PoolPrimSpec(BaseModel):
    """One variable prim in the base scene.

    The order of these entries inside :attr:`PoolManifest.prims` *is* the column order of a
    shard's second axis, so entries must never be reordered for an existing pool.
    """

    path: str = Field(
        min_length=1,
        description="Prim path relative to the base scene's default prim, e.g. 'Props/Mug_00'.",
    )

    role: str = Field(
        default="",
        description="Free-form label describing what the prim is for, e.g. 'pick_object'. Diagnostics only.",
    )

    object_type: ObjectType = Field(
        description="Physics type Arena spawns the prim as, which decides how a pose is written.",
    )

    parkable: bool = Field(
        default=False,
        description="Whether a variation may park this prim instead of placing it in the scene.",
    )

    @field_validator("path")
    @classmethod
    def _reject_absolute_path(cls, value: str) -> str:
        assert not value.startswith("/"), f"prim path {value!r} must be relative to the scene's default prim"
        return value.strip("/")


class ShardLayout(BaseModel):
    """How the producer packs variations into shard files."""

    variations_per_shard: int = Field(gt=0, description="Rows on a shard's first axis.")

    pose_columns: int = Field(
        default=pool_contract.POSE_COLUMNS,
        description="Components per pose row.",
    )

    pose_dtype: str = Field(default=pool_contract.POSE_DTYPE, description="numpy dtype of the poses array.")

    quaternion_order: str = Field(
        default=pool_contract.QUATERNION_ORDER,
        description="Component order of the pose row's quaternion.",
    )

    scalars: list[str] = Field(
        default_factory=list,
        description="Per-variation scalars, in the scalars shard's column order.",
    )
    """Named so a consumer resolves a scalar to a column once at load rather than by position."""

    @field_validator("scalars")
    @classmethod
    def _check_scalars_unique(cls, value: list[str]) -> list[str]:
        assert len(set(value)) == len(value), f"duplicate scalar names in the manifest: {value}"
        assert all(value), "a scalar name may not be empty"
        return value

    @field_validator("pose_columns")
    @classmethod
    def _check_columns(cls, value: int) -> int:
        assert value == pool_contract.POSE_COLUMNS, f"pose_columns must be {pool_contract.POSE_COLUMNS}, got {value}"
        return value

    @field_validator("pose_dtype")
    @classmethod
    def _check_dtype(cls, value: str) -> str:
        assert value == pool_contract.POSE_DTYPE, f"pose_dtype must be {pool_contract.POSE_DTYPE!r}, got {value!r}"
        return value

    @field_validator("quaternion_order")
    @classmethod
    def _check_quaternion_order(cls, value: str) -> str:
        # Arena writes a shard row straight to the simulator, so a mismatch here would mirror every
        # rotation in the pool rather than fail.
        assert value == pool_contract.QUATERNION_ORDER, (
            f"quaternion_order must be {pool_contract.QUATERNION_ORDER!r}, got {value!r}; "
            "the producer must emit poses in Arena's order"
        )
        return value


class ScalarJointSpec(BaseModel):
    """A joint the pool expects a reset to drive from one of its per-variation scalars.

    The producer publishes this because it is what knows the joint's name, its travel and which way
    it slides; an Arena environment that restated them would drift out of step with the pool.
    """

    scalar: str = Field(min_length=1, description="Scalar to read, which the manifest must declare.")

    prims: list[str] = Field(
        min_length=1,
        description=(
            "Inventory paths of the articulations carrying the joint. All of them: a pool "
            "holds alternatives as a parked superset, so which one a variation places is not known here."
        ),
    )

    joint_name: str = Field(min_length=1, description="Prim name of the joint inside each articulation.")

    travel: float = Field(
        description="Joint position at a scalar of 1: metres for a prismatic joint, radians for a revolute one.",
    )

    axis_env_local: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Env-local direction the moving part travels, needed only to carry 'carried' prims.",
    )

    carried: list[str] = Field(
        default_factory=list,
        description=(
            "Inventory paths resting in or on the moving part, measured against the joint at "
            "the scalar the pool chose. A consumer overriding the scalar has to move these with it."
        ),
    )


class PoolManifest(BaseModel):
    """Everything Arena needs to interpret a pool directory."""

    schema_version: int = Field(description="Contract version the pool was written against.")

    scene_usd: str = Field(
        default=pool_contract.DEFAULT_SCENE_FILENAME,
        description="Base scene filename, relative to the pool directory.",
    )

    scene_digest: str = Field(
        default="",
        description="Optional sha256 of the base scene, used to detect a pool rebuilt underneath a run.",
    )

    inventory_digest: str = Field(
        default="",
        description=(
            "Optional sha256 over the prim inventory. Changing the inventory requires a new pool "
            "directory, because existing shards are packed against the old column order."
        ),
    )

    num_envs_hint: int = Field(
        default=0,
        ge=0,
        description="Environment count the base scene's asset multiplicity was sized for. Diagnostics only.",
    )

    parking_offset_xyz: tuple[float, float, float] = Field(
        description=(
            "Env-local offset a parked prim is written to. Must sit below the floor and outside every "
            "camera frustum, because parking is what hides an inactive asset."
        ),
    )

    shard: ShardLayout = Field(description="Shard packing layout.")

    prims: list[PoolPrimSpec] = Field(min_length=1, description="Ordered inventory of variable prims.")

    scalar_joints: list[ScalarJointSpec] = Field(
        default_factory=list,
        description="Joints a reset should drive from the pool's scalars.",
    )

    @model_validator(mode="after")
    def _check_scalar_joints_resolve(self) -> PoolManifest:
        """Reject a joint naming a scalar or a prim the pool does not have.

        Left unchecked, a stale joint declaration fails at the first reset -- after the scene has
        loaded and the pool has filled -- rather than when the manifest is read.
        """
        paths = {prim.path for prim in self.prims}
        for joint in self.scalar_joints:
            assert joint.scalar in self.shard.scalars, (
                f"scalar joint {joint.joint_name!r} reads scalar {joint.scalar!r}, but the pool "
                f"declares {self.shard.scalars}"
            )
            unknown = sorted((set(joint.prims) | set(joint.carried)) - paths)
            assert not unknown, f"scalar joint {joint.joint_name!r} names prims outside the inventory: {unknown}"
        return self

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: int) -> int:
        assert value == pool_contract.SCHEMA_VERSION, (
            f"pool schema version {value} is not supported by this Arena build "
            f"(expected {pool_contract.SCHEMA_VERSION}); regenerate the pool"
        )
        return value

    @field_validator("prims")
    @classmethod
    def _check_unique_paths(cls, value: list[PoolPrimSpec]) -> list[PoolPrimSpec]:
        seen: set[str] = set()
        for prim in value:
            assert prim.path not in seen, f"duplicate prim path {prim.path!r} in the pool inventory"
            seen.add(prim.path)
        return value

    @property
    def num_prims(self) -> int:
        """Length of a shard's second axis."""
        return len(self.prims)

    def compute_inventory_digest(self) -> str:
        """Return the sha256 over the ordered inventory.

        Used to check a manifest's recorded :attr:`inventory_digest` when it carries one.
        """
        payload = json.dumps(
            [[prim.path, prim.object_type.value, prim.parkable] for prim in self.prims],
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def parkable_indices(self) -> list[int]:
        """Return inventory positions of prims a variation may park."""
        return [index for index, prim in enumerate(self.prims) if prim.parkable]

    def to_json(self) -> str:
        """Return the manifest as indented JSON."""
        return json.dumps(self.model_dump(mode="json"), indent=2)


def load_pool_manifest(pool_dir: str | Path) -> PoolManifest:
    """Read and validate the manifest in a pool directory.

    Args:
        pool_dir: Directory the producer writes into.

    Returns:
        The validated manifest.
    """
    path = pool_contract.manifest_path(pool_dir)
    assert path.is_file(), f"variation pool has no manifest at {path}"
    manifest = PoolManifest.model_validate_json(path.read_text())
    if manifest.inventory_digest:
        actual = manifest.compute_inventory_digest()
        assert actual == manifest.inventory_digest, (
            f"pool inventory digest mismatch in {path}: manifest records {manifest.inventory_digest} "
            f"but the inventory hashes to {actual}"
        )
    return manifest


def resolve_scene_usd(pool_dir: str | Path, manifest: PoolManifest) -> Path:
    """Return the absolute path to the pool's base scene.

    Args:
        pool_dir: Directory the producer writes into.
        manifest: Manifest already loaded from that directory.
    """
    path = Path(pool_dir) / manifest.scene_usd
    assert path.is_file(), f"variation pool manifest points at a missing base scene: {path}"
    return path
