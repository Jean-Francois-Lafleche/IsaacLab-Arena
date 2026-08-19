# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Map a pool's prim inventory onto Arena scene entity names.

Kept free of ``pxr`` so the environment builder and unit tests can compute a write plan without
opening USD.
"""

from __future__ import annotations

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.live_scene.pool_manifest import PoolManifest


def object_reference_name(prim_path: str) -> str:
    """Return the scene entity name for a pool inventory prim path.

    Args:
        prim_path: Prim path relative to the base scene's default prim.
    """
    slug = prim_path.strip("/").replace("/", "_")
    assert slug, f"prim path {prim_path!r} yields no usable scene entity name"
    return slug


def build_variation_write_plan(manifest: PoolManifest) -> list[tuple[int, str]]:
    """Return the ``(inventory index, scene entity name)`` pairs the reset event writes.

    Only rigid and articulated prims accept a root pose write, so static prims are filtered out here
    rather than branched over on every reset.

    Args:
        manifest: Pool manifest describing the variable prims.
    """
    plan: list[tuple[int, str]] = []
    for index, prim in enumerate(manifest.prims):
        if prim.object_type is ObjectType.BASE:
            continue
        plan.append((index, object_reference_name(prim.path)))
    assert plan, (
        "pool inventory has no rigid or articulated prims, so no variation could ever be applied; "
        "mark the variable prims with object_type rigid or articulation"
    )
    return plan
