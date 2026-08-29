# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Let Arena, rather than the pool, decide where a task's objects start.

A generated pool poses the scene, and left to itself it also poses the objects the task is about.
That puts a task decision in the scene generator: change what the drawer presents and you have to
regenerate every variation. This moves the decision back to Arena. The pool says only which prim is
the surface -- through the env-graph sidecar it writes beside the scene -- and Arena solves a layout
on that surface with its own relation solver.

Two things make this more than a call to the solver. The surface moves: which cabinet is standing,
where along the wall it stands, and how far its drawer is pulled out are all redrawn per variation,
so a layout solved in world coordinates would be wrong for every episode. And the solver runs at
build time against static geometry, which is far too slow for a reset. So layouts are solved once
per candidate surface, in the frame of the prim that carries it, and a reset picks one and transforms
it by that prim's drawn pose -- a handful of tensor ops on poses already in hand.
"""

from __future__ import annotations

import json
import torch
from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.live_scene.pool_manifest import PoolManifest

POOL_ENV_GRAPH_NAME = "scene.arena.json"
"""Env-graph sidecar the producer writes beside a pool's base scene."""

SUPPORT_RELATION = "on"
"""Env-graph relation kind meaning the subject rests on the referenced support."""

PLACE_EVERY_SIDECAR_SUBJECT = "*"
"""Stands in for the whole list of objects the sidecar gives a support for.

Saves an environment restating a pool's drawer contents, which would go stale as soon as the
producer added to them."""

DEFAULT_EDGE_MARGIN = 0.03
"""How far in from a surface's edges an object's footprint is kept, in metres.

Smaller than a bare sampler would need, because the solver works in footprints rather than origins
and already accounts for each object's own size and yaw."""

DEFAULT_CLEARANCE = 0.005
"""Gap left between an object and the surface it rests on, in metres.

Small and positive: resting exactly on the surface leaves the two in contact at the first step, which
PhysX resolves by pushing them apart."""

DEFAULT_LAYOUTS_PER_SUPPORT = 16
"""Distinct layouts solved for each candidate surface.

The reset path picks one of these rather than solving, so this is the whole variety a surface ever
shows. Solved once at build, so the cost is paid before the run starts."""

LAYOUT_COLUMNS = 4
"""A solved layout row is a position followed by a yaw, all in the supporting prim's frame."""


@dataclass(frozen=True)
class SupportRegion:
    """A surface objects may rest on, measured in the frame of the prim carrying it.

    Attributes:
        owner_column: Shard column of the prim the region is measured against, whose drawn pose
            places the region in the world.
        x_min: Low edge of the usable rectangle along the owner's x axis, in metres.
        x_max: High edge of the usable rectangle along the owner's x axis, in metres.
        y_min: Low edge of the usable rectangle along the owner's y axis, in metres.
        y_max: High edge of the usable rectangle along the owner's y axis, in metres.
        z_top: Height of the resting surface in the owner's frame, in metres.
    """

    owner_column: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_top: float

    @property
    def centre(self) -> tuple[float, float]:
        """Middle of the usable rectangle, in the owner's frame."""
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def extent(self) -> tuple[float, float]:
        """Width and depth of the usable rectangle, in metres."""
        return self.x_max - self.x_min, self.y_max - self.y_min


@dataclass(frozen=True)
class SupportPlacement:
    """Pre-solved layouts for a set of objects sharing one set of candidate surfaces.

    Attributes:
        subject_columns: Shard columns of the objects placed, in the order they appear in a layout.
        regions: One region per candidate surface, in shard-column order of their owners.
        layouts: ``(n_regions, n_layouts, n_subjects, 4)`` of solved position and yaw, in the frame
            of each region's owner.
        parked_below_z: Env-local height under which a prim counts as parked rather than standing.
        carry_axis: Env-local direction the supports travel as their joint opens.
        carry_travel: Distance the supports travel over the joint's full range, in metres.
        carry_scalar: Pool scalar holding how far open that joint is, from zero to one.
    """

    subject_columns: tuple[int, ...]
    regions: tuple[SupportRegion, ...]
    layouts: torch.Tensor
    parked_below_z: float
    carry_axis: tuple[float, float, float] = (0.0, 0.0, 0.0)
    carry_travel: float = 0.0
    carry_scalar: str | None = None


class SupportPlacementHandle:
    """Opaque holder for solved placements used as an ``EventTermCfg`` param.

    Isaac Lab walks every term parameter to rewrite the strings in it, assigning into any dataclass it
    finds, and dumps the lot to YAML for the run's logs. A placement is deliberately frozen, so the
    first would raise, and its layouts are a tensor no log wants. This mirrors
    :class:`~isaaclab_arena.live_scene.variation_pool.LiveVariationPoolHandle`: ``__slots__`` means
    ``hasattr(handle, "__dict__")`` is false, so the walk stops here.
    """

    __slots__ = ("placements",)

    def __init__(self, placements: tuple[SupportPlacement, ...]) -> None:
        self.placements = placements

    def __deepcopy__(self, memo: dict[int, object]) -> SupportPlacementHandle:
        """Share the solved layouts across ``copy.deepcopy`` rather than duplicating the tensors."""
        memo[id(self)] = self
        return self

    def __getstate__(self) -> dict[str, list[int]]:
        """Serialize as a description of what is placed rather than the layouts themselves."""
        return {"placed_columns": [column for placement in self.placements for column in placement.subject_columns]}

    def __setstate__(self, state: dict[str, list[int]]) -> None:
        """Restore with nothing to place, since solved layouts do not cross a process boundary."""
        self.placements = ()


def env_graph_path(pool_dir: str | Path) -> Path:
    """Return where a pool's env-graph sidecar lives.

    Args:
        pool_dir: Pool directory.
    """
    return Path(pool_dir) / POOL_ENV_GRAPH_NAME


def _read_env_graph(pool_dir: str | Path) -> dict:
    """Return a pool's env-graph sidecar.

    Args:
        pool_dir: Pool directory holding the sidecar.
    """
    path = env_graph_path(pool_dir)
    assert path.is_file(), (
        f"no env-graph sidecar at {path}, so Arena cannot know which prims its objects rest on; "
        "the pool's producer writes one beside the scene"
    )
    return json.loads(path.read_text())


def load_support_prim_paths(pool_dir: str | Path, subject: str) -> tuple[str, ...]:
    """Return the prims the pool's sidecar says ``subject`` may rest on, in the order declared.

    More than one is normal and is why the sidecar is read at all: the pool holds every candidate
    support in a parked superset and stands exactly one of them per variation, so the choice can
    only be made per episode.

    Args:
        pool_dir: Pool directory holding the sidecar.
        subject: Env-graph id of the object being placed, which for a pool is its inventory path.
    """
    graph = _read_env_graph(pool_dir)
    by_id = {reference["id"]: reference["prim_path"] for reference in graph.get("object_references", ())}
    supports = [
        relation["reference"]
        for relation in graph.get("relations", ())
        if relation.get("kind") == SUPPORT_RELATION and relation.get("subject") == subject
    ]
    assert supports, (
        f"the env-graph at {env_graph_path(pool_dir)} declares no {SUPPORT_RELATION!r} relation for "
        f"{subject!r}, only for {sorted({r.get('subject') for r in graph.get('relations', ())})}"
    )
    missing = sorted(set(supports) - set(by_id))
    assert not missing, f"the env-graph at {env_graph_path(pool_dir)} names supports {missing} it does not describe"
    return tuple(by_id[support] for support in supports)


def load_placed_subjects(pool_dir: str | Path) -> tuple[str, ...]:
    """Return every object the pool's sidecar gives a support for, in the order declared.

    Lets an environment hand placement of a whole surface's contents to Arena without restating what
    the producer put there, which would go stale the moment the pool's inventory changed.

    Args:
        pool_dir: Pool directory holding the sidecar.
    """
    seen: list[str] = []
    for relation in _read_env_graph(pool_dir).get("relations", ()):
        subject = relation.get("subject")
        if relation.get("kind") == SUPPORT_RELATION and subject not in seen:
            seen.append(subject)
    assert seen, f"the env-graph at {env_graph_path(pool_dir)} gives no object a support to rest on"
    return tuple(seen)


def _owner_column(manifest: PoolManifest, support_path: str) -> int:
    """Return the shard column of the inventory prim a support hangs below.

    Args:
        manifest: Pool manifest, whose prim order is the shard's column order.
        support_path: Support prim path, relative to the scene's default prim.
    """
    candidates = [
        (index, prim.path)
        for index, prim in enumerate(manifest.prims)
        if support_path == prim.path or support_path.startswith(f"{prim.path}/")
    ]
    assert candidates, f"support {support_path!r} sits under no prim the pool poses, so it can never be placed"
    # The longest match, so a support under a nested inventory prim is owned by the nearest one.
    return max(candidates, key=lambda candidate: len(candidate[1]))[0]


def _local_range(stage, cache, prim, to_frame):
    """Return a prim's bounds expressed in another frame.

    Args:
        stage: Stage holding the prim.
        cache: Bounding-box cache to compute world bounds with.
        prim: The prim to measure.
        to_frame: Transform from world into the frame wanted.
    """
    from pxr import Gf

    world = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    assert not world.IsEmpty(), f"{prim.GetPath()} has no geometry to measure"
    local = Gf.Range3d()
    for corner in range(8):
        local.UnionWith(to_frame.Transform(world.GetCorner(corner)))
    return local


def measure_support_regions(scene_usd: str | Path, manifest: PoolManifest, support_paths: tuple[str, ...]):
    """Measure each support's resting surface in the frame of the prim that carries it.

    Read from the base scene rather than from the simulator, because a support only reaches the pose
    it will be used at once a variation has been drawn, and this runs before any of them have.

    Args:
        scene_usd: Base scene the pool's variations are deltas against.
        manifest: Pool manifest, for resolving which prim owns each support.
        support_paths: Support prims, relative to the scene's default prim.

    Returns:
        One region per support, in the order given.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(scene_usd))
    root = stage.GetDefaultPrim().GetPath()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])

    regions = []
    for support_path in support_paths:
        owner_column = _owner_column(manifest, support_path)
        owner = stage.GetPrimAtPath(root.AppendPath(manifest.prims[owner_column].path))
        support = stage.GetPrimAtPath(root.AppendPath(support_path))
        assert support.IsValid(), f"the env-graph names a support at {support_path!r} that the base scene lacks"
        to_owner = UsdGeom.Xformable(owner).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        local = _local_range(stage, cache, support, to_owner)
        low, high = local.GetMin(), local.GetMax()
        regions.append(
            SupportRegion(
                owner_column=owner_column,
                x_min=float(low[0]),
                x_max=float(high[0]),
                y_min=float(low[1]),
                y_max=float(high[1]),
                z_top=float(high[2]),
            )
        )
    return tuple(regions)


def measure_subject_bounds(scene_usd: str | Path, subject_paths: tuple[str, ...]):
    """Return each subject's bounds about its own origin, which is what the solver reasons over.

    Args:
        scene_usd: Base scene holding the subjects.
        subject_paths: Subject prims, relative to the scene's default prim.

    Returns:
        One ``AxisAlignedBoundingBox`` per subject, in the order given.
    """
    from pxr import Usd, UsdGeom

    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    stage = Usd.Stage.Open(str(scene_usd))
    root = stage.GetDefaultPrim().GetPath()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])

    bounds = []
    for subject_path in subject_paths:
        prim = stage.GetPrimAtPath(root.AppendPath(subject_path))
        assert prim.IsValid(), f"the base scene has no prim at {subject_path!r} to place"
        to_local = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        local = _local_range(stage, cache, prim, to_local)
        bounds.append(
            AxisAlignedBoundingBox(
                min_point=tuple(float(value) for value in local.GetMin()),
                max_point=tuple(float(value) for value in local.GetMax()),
            )
        )
    return tuple(bounds)


def _measured_placeable(name: str, bounding_box):
    """Return a stand-in asset carrying only the bounds the relation solver reasons over.

    The solver takes assets, but the things being placed here are prims inside the pool's base scene
    rather than Arena assets, and the surface they go on is a rectangle measured in a frame no asset
    exists in. Bounds are the whole of what the solver needs from either.

    Args:
        name: Name the solver reports the asset under.
        bounding_box: Bounds about the asset's own origin.
    """
    from isaaclab_arena.relations.placement_asset import PlaceableAsset

    class MeasuredPlaceable(PlaceableAsset):
        """An asset that is only ever a bounding box."""

        def get_bounding_box(self):
            """Return the bounds this stand-in was measured with."""
            return bounding_box

    return MeasuredPlaceable(name=name)


def solve_layouts_on_region(
    region: SupportRegion,
    subject_names: tuple[str, ...],
    subject_bounds,
    layouts: int,
    clearance: float,
    edge_margin: float,
) -> torch.Tensor:
    """Solve collision-free layouts for a set of objects on one surface.

    Args:
        region: The surface, in its owner's frame.
        subject_names: Names to solve under, for diagnostics.
        subject_bounds: Bounds of each subject, in the same order.
        layouts: How many distinct layouts to solve.
        clearance: Gap left between an object and the surface, in metres.
        edge_margin: How far in from the surface's edges a footprint is kept, in metres.

    Returns:
        ``(layouts, n_subjects, 4)`` of position and yaw, in the region owner's frame.
    """
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    width, depth = region.extent
    centre_x, centre_y = region.centre
    # A slab whose top face is the surface: the solver reads a support's height off the top of its
    # bounds, and its footprint off their extent, so this is the region expressed as an asset.
    surface = _measured_placeable(
        "support",
        AxisAlignedBoundingBox(
            min_point=(-width / 2.0, -depth / 2.0, -0.01), max_point=(width / 2.0, depth / 2.0, 0.0)
        ),
    )
    surface.set_initial_pose(Pose(position_xyz=(centre_x, centre_y, region.z_top)), create_reset_event=False)
    surface.add_relation(IsAnchor())

    subjects = []
    for name, bounds in zip(subject_names, subject_bounds):
        subject = _measured_placeable(name, bounds)
        subject.add_relation(On(surface, clearance_m=clearance, edge_margin_m=edge_margin))
        subjects.append(subject)

    placer = ObjectPlacer(
        ObjectPlacerParams(
            # The pool has already chosen what stands; the solver is only being asked where it goes.
            apply_positions_to_objects=False,
            # Yaw variety without a second pass: the solver collides against the box enclosing the
            # rotated object, so a layout stays valid at the angle it was solved for.
            random_yaw_init=True,
            # One solve per candidate surface, so the default running commentary would be printed
            # nine times over before a run has even started.
            solver_params=RelationSolverParams(verbose=False),
        )
    )
    ranked = placer.place_ranked_per_env([surface, *subjects], num_envs=1, results_per_env=layouts)
    assert len(ranked) == 1, f"asked for one environment's layouts and got {len(ranked)}"

    solved = torch.zeros((layouts, len(subjects), LAYOUT_COLUMNS), dtype=torch.float32)
    # Checked by name rather than through ``success``, which gates on the solver's required set and
    # lets an overlapping layout through. Two objects sharing a spot in a drawer is exactly the
    # failure this whole path exists to avoid, so it is required here whatever the solver's default.
    demanded = (PlacementCheck.NO_OVERLAP, PlacementCheck.ON_RELATION)
    usable = [
        result
        for result in ranked[0]
        if result.validation_results.do_all_required_validation_checks_pass(list(demanded))
    ]
    assert usable, (
        f"the solver laid out none of {list(subject_names)} on a {width:.2f} by {depth:.2f} m surface "
        "without overlaps; the objects may not fit, or the edge margin may be too wide for them"
    )
    for index in range(layouts):
        # Ranked best first, so cycling reuses the good layouts rather than padding with failures
        # when the surface is too tight to yield the full count.
        result = usable[index % len(usable)]
        for subject_index, subject in enumerate(subjects):
            solved[index, subject_index, :3] = torch.tensor(result.positions[subject])
            solved[index, subject_index, 3] = result.orientations.get(subject, 0.0)
    return solved


def build_support_placement(
    pool_dir: str | Path,
    manifest: PoolManifest,
    subjects: tuple[str, ...],
    layouts: int = DEFAULT_LAYOUTS_PER_SUPPORT,
    clearance: float = DEFAULT_CLEARANCE,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
) -> SupportPlacement:
    """Solve every candidate surface's layouts from the pool's own files.

    All the measuring and solving happens here, at build time, so the reset path only transforms
    numbers it already holds.

    Args:
        pool_dir: Pool directory, holding the base scene, manifest and env-graph sidecar.
        manifest: Manifest already loaded from that directory.
        subjects: Inventory paths of the objects to place, which must share a set of supports.
        layouts: Distinct layouts to solve per surface.
        clearance: Gap left between an object and the surface, in metres.
        edge_margin: How far in from a surface's edges a footprint is kept, in metres.
    """
    from isaaclab_arena.live_scene.pool_manifest import resolve_scene_usd

    columns = {prim.path: index for index, prim in enumerate(manifest.prims)}
    unknown = sorted(set(subjects) - set(columns))
    assert not unknown, f"{unknown} are not in the pool's inventory, so they have no pose to set"

    supports = {subject: load_support_prim_paths(pool_dir, subject) for subject in subjects}
    shared = set(supports.values())
    assert len(shared) == 1, (
        "every object placed together must rest on the same set of candidate supports, but the "
        f"sidecar gives {({subject: len(paths) for subject, paths in supports.items()})} distinct sets"
    )

    scene_usd = resolve_scene_usd(pool_dir, manifest)
    regions = measure_support_regions(scene_usd, manifest, shared.pop())
    bounds = measure_subject_bounds(scene_usd, subjects)

    # A support the pool also slides has to be followed, or the objects are placed where the drawer
    # was rather than where it stops.
    owners = {manifest.prims[region.owner_column].path for region in regions}
    carrying = [joint for joint in manifest.scalar_joints if owners & set(joint.prims)]
    assert len(carrying) <= 1, (
        f"{list(subjects)} rest on supports driven by more than one joint ("
        f"{[joint.joint_name for joint in carrying]}), which this cannot follow"
    )

    return SupportPlacement(
        subject_columns=tuple(columns[subject] for subject in subjects),
        regions=regions,
        layouts=torch.stack(
            [solve_layouts_on_region(region, subjects, bounds, layouts, clearance, edge_margin) for region in regions]
        ),
        # Halfway to the parking yard: standing supports are at room height and parked ones are far
        # below the floor, so anything in between is unambiguous.
        parked_below_z=manifest.parking_offset_xyz[2] / 2.0,
        carry_axis=carrying[0].axis_env_local if carrying else (0.0, 0.0, 0.0),
        carry_travel=carrying[0].travel if carrying else 0.0,
        carry_scalar=carrying[0].scalar if carrying else None,
    )


def _rotate(quaternions: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions, broadcasting over any leading dimensions.

    Args:
        quaternions: ``(..., 4)`` unit quaternions in the pool's ``(x, y, z, w)`` order.
        vectors: ``(..., 3)`` vectors to rotate.
    """
    imaginary, real = quaternions[..., :3], quaternions[..., 3:]
    cross = torch.cross(imaginary.expand_as(vectors), vectors, dim=-1)
    return vectors + 2.0 * (real * cross + torch.cross(imaginary.expand_as(vectors), cross, dim=-1))


def _yaw_of(quaternions: torch.Tensor) -> torch.Tensor:
    """Return the rotation about z of upright quaternions, in radians.

    Args:
        quaternions: ``(..., 4)`` unit quaternions in the pool's ``(x, y, z, w)`` order.
    """
    x, y, z, w = quaternions.unbind(-1)
    assert bool(
        (x.abs() < 1.0e-3).all() and (y.abs() < 1.0e-3).all()
    ), "a support is tipped over, so the objects on it cannot be placed by yaw alone"
    return 2.0 * torch.atan2(z, w)


def place_on_supports(
    poses: torch.Tensor,
    placement: SupportPlacement,
    openness: torch.Tensor | None,
) -> None:
    """Apply a solved layout to whichever of the placement's supports is standing.

    Mutates ``poses`` in place, in the same pass the pool's own poses are prepared in, so the objects
    are written once at their final positions rather than moved afterwards. Run this after any carry,
    so that objects the pool also posed end up where Arena put them rather than where the pool did.

    Objects the pool parked are left parked: a variation stands a subset of the drawer's contents,
    and a layout solved for all of them stays collision-free for any subset.

    Args:
        poses: ``(n_envs, n_prims, 7)`` env-local poses, modified in place.
        placement: The objects, their candidate supports and the layouts solved on each.
        openness: ``(n_envs,)`` fraction each support's joint is driven to, after any curriculum
            override. ``None`` when the supports do not move.
    """
    device, n_envs = poses.device, poses.shape[0]
    owners = torch.tensor([region.owner_column for region in placement.regions], device=device)
    subjects = torch.tensor(placement.subject_columns, device=device)
    layouts = placement.layouts.to(device=device, dtype=poses.dtype)

    # Exactly one support stands per environment and the rest are in the parking yard far below, so
    # a height test is enough to find it without the pool having to publish which one it chose.
    standing = poses[:, owners, 2] > placement.parked_below_z
    assert bool(
        standing.any(dim=1).all()
    ), "some environments drew a variation with none of the objects' supports standing, so there is nowhere to put them"
    chosen = torch.argmax(standing.to(torch.int8), dim=1)
    drawn = torch.randint(layouts.shape[1], (n_envs,), device=device)
    local = layouts[chosen, drawn]

    owner_poses = poses[torch.arange(n_envs, device=device), owners[chosen]]
    orientation = owner_poses[:, 3:7].unsqueeze(1)
    position = owner_poses[:, None, :3] + _rotate(orientation, local[..., :3])
    if openness is not None:
        axis = torch.tensor(placement.carry_axis, dtype=poses.dtype, device=device)
        position = position + (openness * placement.carry_travel)[:, None, None] * axis

    # Yaw only. A mug lying on its side is a different task, not a harder one, and the layouts were
    # solved flat on the surface.
    yaw = _yaw_of(owner_poses[:, 3:7]).unsqueeze(1) + local[..., 3]
    placed = torch.zeros((n_envs, len(placement.subject_columns), poses.shape[2]), dtype=poses.dtype, device=device)
    placed[..., :3] = position
    placed[..., 5] = torch.sin(yaw / 2.0)
    placed[..., 6] = torch.cos(yaw / 2.0)

    # A parked object keeps the pool's parking pose, so a variation showing less clutter still shows
    # less clutter after Arena has had its say.
    keep = (poses[:, subjects, 2] <= placement.parked_below_z).unsqueeze(-1)
    poses[:, subjects, :] = torch.where(keep, poses[:, subjects, :], placed)
