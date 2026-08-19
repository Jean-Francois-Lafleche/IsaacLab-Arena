# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Background for an externally generated scene, plus the object references its pool addresses."""

from __future__ import annotations

from typing import Any

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.live_scene.pool_manifest import PoolManifest
from isaaclab_arena.live_scene.scene_entities import object_reference_name
from isaaclab_arena.utils.pose import Pose

GENERATED_SCENE_BACKGROUND_REGISTRY_NAME = "generated_scene_background"
"""Registry name for spawning a generated scene from a usd_path given in the spec's params."""

DEFAULT_GENERATED_OBJECT_MIN_Z = -0.5
"""Drop threshold for generated scenes, in metres. Overridable through params."""


@register_asset
class GeneratedSceneBackground(Background):
    """Spawn a generated base scene from an explicit USD path supplied in graph params.

    Mirrors :class:`~isaaclab_arena.assets.simready_object_library.SimReadyUsdObject`: the asset is
    registered once and carries no USD of its own, so a spec generated in one process loads in any
    other as long as the path is reachable.
    """

    name = GENERATED_SCENE_BACKGROUND_REGISTRY_NAME

    tags = ["background", "generated"]

    def __init__(
        self,
        usd_path: str,
        instance_name: str | None = None,
        prim_path: str | None = None,
        object_min_z: float = DEFAULT_GENERATED_OBJECT_MIN_Z,
        initial_pose: Pose | None = None,
        **kwargs: Any,
    ):
        assert usd_path, f"{GENERATED_SCENE_BACKGROUND_REGISTRY_NAME} requires params.usd_path"
        super().__init__(
            name=instance_name or self.name,
            tags=self.tags,
            usd_path=usd_path,
            object_min_z=object_min_z,
            prim_path=prim_path,
            initial_pose=initial_pose,
            **kwargs,
        )


def build_object_references(
    manifest: PoolManifest,
    background: Background,
    openable_joint_names: dict[str, str] | None = None,
) -> list[ObjectReference]:
    """Create one object reference per variable prim in the pool inventory.

    Returned references are ordered to match the manifest inventory, which is also the column order
    of a variation shard.

    Args:
        manifest: Pool manifest describing the variable prims.
        background: The generated scene these prims live inside.
        openable_joint_names: Optional ``{prim path: joint name}`` for articulated prims that a task
            opens or closes.

    Returns:
        The object references, in inventory order.
    """
    openable_joint_names = openable_joint_names or {}
    references: list[ObjectReference] = []
    for prim in manifest.prims:
        name = object_reference_name(prim.path)
        prim_path = f"{background.prim_path.rstrip('/')}/{prim.path}"
        joint_name = openable_joint_names.get(prim.path)
        if joint_name is not None:
            assert prim.object_type is ObjectType.ARTICULATION, (
                f"prim {prim.path!r} declares an openable joint but its object_type is "
                f"{prim.object_type.value}; it must be articulation"
            )
            references.append(
                OpenableObjectReference(
                    openable_joint_name=joint_name,
                    name=name,
                    prim_path=prim_path,
                    parent_asset=background,
                )
            )
            continue
        references.append(
            ObjectReference(
                name=name,
                prim_path=prim_path,
                parent_asset=background,
                object_type=prim.object_type,
            )
        )
    return references
