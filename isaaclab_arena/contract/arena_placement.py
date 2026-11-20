# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena's relation solver places what the contract's placements describe.

The contract says *what* is placed and *where it may go*: each ``layout.placement`` operator names its targets,
the support they stand on (``on``), optionally the region they must stay inside (``inside``), and a
distribution whose domain is how far each target may move from its authored pose, in the support's axes
(``x``, ``y``, ``yaw``). Here every such placement becomes an Arena placement problem:

- the support is an anchor (:class:`~isaaclab_arena.relations.relations.IsAnchor`): the floor of the
  ``inside`` region when there is one (a drawer's inside), otherwise the support prim's top;
- each target is placed ``On`` it, kept in the distribution's box by a ``PositionLimitsBox``;
- Arena's :class:`~isaaclab_arena.relations.pooled_object_placer.PooledObjectPlacer` solves a pool of
  layouts when the environment is built.

At every reset the placement's variation draws one of Arena's layouts and turns it into the contract's value
for the distribution: each target's offset from its authored pose, in the support's axes. The value is pinned
in the contract's request like any other Arena draw, so the contract applies it as given, moves it with what
its support does next (a drawer opening with the mug in it) and checks every required condition on the
result; a layout the contract rejects is drawn again. Arena's solver has no yaw range: a placement that may
turn its targets all the way round uses Arena's random yaw, a narrower one draws yaw from the variation's own
sampler.

Everything is computed on the template's nominal stage, in its frame: the frame of every environment.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any

PLACEMENT = "layout.placement"


def _scene_prim_type() -> type:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset

    class ScenePrim(PlaceableAsset):
        """A prim of the scene file, as Arena's solver sees it: a box about its origin, and a pose if fixed."""

        def __init__(self, name: str, bounding_box, initial_pose=None):
            super().__init__(name=name)
            self.bounding_box = bounding_box
            self.initial_pose = initial_pose
            self.relations = []

        def _build_reset_event(self):
            return None  # the contract writes every pose: Arena only solves

        def get_bounding_box(self):
            return self.bounding_box

        def get_corners_aabb(self, pos: torch.Tensor) -> torch.Tensor:
            return self.bounding_box.get_corners_at(pos)

        def is_initial_pose_set(self) -> bool:
            return self.initial_pose is not None

    return ScenePrim


def _world(stage, path: str):
    from pxr import Usd, UsdGeom

    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _axes(matrix) -> np.ndarray:
    """The orthonormal rotation of a world transform, as rows x, y, z of a 3x3 array (columns are world axes)."""
    rows = np.array([[matrix[i][j] for j in range(3)] for i in range(3)], dtype=float)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _yaw_deg(matrix) -> float:
    x = _axes(matrix)[0]
    return math.degrees(math.atan2(x[1], x[0]))


def _world_box(stage, path: str) -> tuple[np.ndarray, np.ndarray]:
    """A prim's world-aligned bounds; for an invisible region (a guide ``Cube``) its unit cube, transformed."""
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if prim.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
        box = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(-half, -half, -half), Gf.Vec3d(half, half, half)), _world(stage, path))
        rng = box.ComputeAlignedRange()
    else:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(rng.GetMin(), dtype=float), np.array(rng.GetMax(), dtype=float)


def _rotation(rotation_deg: tuple[float, ...]) -> np.ndarray:
    """A placement's fixed ``rotation`` ``(roll, pitch, yaw)``: ``Rz(yaw) · Ry(pitch) · Rx(roll)`` (layout@1)."""
    roll, pitch, yaw = (math.radians(a) for a in (tuple(rotation_deg) + (0.0, 0.0, 0.0))[:3])
    rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _local_box(stage, path: str, rotation_deg: tuple[float, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """A prim's bounds about its origin in world-aligned axes, turned by the placement's fixed ``rotation``.

    The rotation is about the prim's own axes, before it is placed (layout@1): the prim's world orientation
    becomes ``R_world · R_fixed``, and its own-frame bounds are turned by that.
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    # the range is in the prim's own frame (the box's matrix only undoes the prim's own transform)
    rng = cache.ComputeUntransformedBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    corners = [
        np.array((x, y, z), dtype=float)
        for x in (rng.GetMin()[0], rng.GetMax()[0])
        for y in (rng.GetMin()[1], rng.GetMax()[1])
        for z in (rng.GetMin()[2], rng.GetMax()[2])
    ]
    world = _world(stage, path)
    scale = np.linalg.norm(np.array([[world[i][j] for j in range(3)] for i in range(3)], dtype=float), axis=1)
    rotation = _axes(world).T @ (_rotation(rotation_deg) if rotation_deg else np.eye(3))
    turned = [rotation @ (c * scale) for c in corners]
    return np.min(turned, axis=0), np.max(turned, axis=0)


@dataclass
class PlacementProblem:
    """One contract placement as an Arena placement pool, and the conversion back to the contract's value."""

    operator: str
    distribution: str
    targets: tuple[str, ...]
    support_axes: np.ndarray
    authored: dict[str, tuple[np.ndarray, float]]
    yaw_range: tuple[float, float]
    full_yaw: bool
    assets: dict[str, Any] = field(default_factory=dict)
    pool: Any = None

    def pin_rows(self, layouts) -> list[torch.Tensor]:
        """Each layout as the distribution's raw value: per target ``x, y, yaw``, flattened in target order."""
        rows = []
        for layout in layouts:
            row = []
            for target in self.targets:
                asset = self.assets[target]
                p = np.array(layout.positions[asset], dtype=float)
                origin, yaw0 = self.authored[target]
                local = self.support_axes @ (p - origin)
                yaw = math.degrees(float(layout.orientations.get(asset, 0.0))) - yaw0 if self.full_yaw else 0.0
                yaw = (yaw + 180.0) % 360.0 - 180.0
                row += [float(local[0]), float(local[1]), yaw]
            rows.append(torch.tensor(row, dtype=torch.float32))
        return rows


def placement_problems(contract, stage, *, pool_size: int, seed: int) -> dict[str, PlacementProblem]:
    """Arena placement pools for the contract's ``layout.placement`` operators, keyed by distribution path.

    Only placements whose distribution is a per-target ``x, y, yaw`` box are handed to Arena; the others
    (and every ``populate``) stay the contract's.
    """
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimitsBox
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    ScenePrim = _scene_prim_type()
    by_path = {str(n.path): n for n in contract.nodes}
    problems: dict[str, PlacementProblem] = {}
    for k, op in enumerate(n for n in contract.nodes if n.type == PLACEMENT):
        edges = {e.name: list(e.targets) for e in op.edges if not e.form}
        dist = by_path.get((edges.get("distribution") or [None])[0])
        if dist is None or tuple(dist.fields.get("componentNames") or ()) != ("x", "y", "yaw"):
            continue
        if dist.fields.get("valuesPer") != "target" or not edges.get("on"):
            continue
        targets = tuple(dist.edge("targets").targets) if dist.edge("targets") else tuple(edges["targets"])
        lo, hi = (np.array(dist.fields["minimum"], dtype=float), np.array(dist.fields["maximum"], dtype=float))
        support = edges["on"][0]
        axes = _axes(_world(stage, support))
        # The anchor: the floor of the region the targets must stay inside, or the support's top.
        if edges.get("inside"):
            rlo, rhi = _world_box(stage, edges["inside"][0])
            top = rlo[2]
            anchor_lo, anchor_hi = np.array([rlo[0], rlo[1], top - 0.02]), np.array([rhi[0], rhi[1], top])
        else:
            anchor_lo, anchor_hi = _world_box(stage, support)
        anchor = ScenePrim(f"support_{k}", AxisAlignedBoundingBox(tuple(anchor_lo), tuple(anchor_hi)),
                           Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))  # fmt: skip
        anchor.add_relation(IsAnchor())
        rotation = tuple(op.fields.get("rotation") or ())
        problem = PlacementProblem(
            operator=str(op.path), distribution=str(dist.path), targets=targets, support_axes=axes, authored={},
            yaw_range=(float(lo[2]), float(hi[2])), full_yaw=float(hi[2] - lo[2]) >= 359.0,
        )  # fmt: skip
        objects = [anchor]
        for i, target in enumerate(targets):
            world = _world(stage, target)
            origin = np.array(world.ExtractTranslation(), dtype=float)
            problem.authored[target] = (origin, _yaw_deg(world))
            blo, bhi = _local_box(stage, target, rotation)
            asset = ScenePrim(f"target_{k}_{i}", AxisAlignedBoundingBox(tuple(blo), tuple(bhi)))
            asset.add_relation(On(anchor))
            # The distribution's box about the authored pose, in the support's axes, as world limits.
            corners = [origin + axes.T @ np.array([x, y, 0.0]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
            cmin, cmax = np.min(corners, axis=0), np.max(corners, axis=0)
            asset.add_relation(PositionLimitsBox(x_limits=(cmin[0], cmax[0]), y_limits=(cmin[1], cmax[1])))
            problem.assets[target] = asset
            objects.append(asset)
        params = ObjectPlacerParams(placement_seed=seed + k, random_yaw_init=problem.full_yaw)
        problem.pool = PooledObjectPlacer(objects, params, pool_size=pool_size)
        problems[problem.distribution] = problem
    return problems
