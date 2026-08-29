# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena laying out a drawer's contents itself, on a surface the pool only names."""

from __future__ import annotations

import json
import math
import torch

import pytest

from isaaclab_arena.live_scene.support_placement import (
    LAYOUT_COLUMNS,
    SupportPlacement,
    SupportRegion,
    env_graph_path,
    load_placed_subjects,
    load_support_prim_paths,
    place_on_supports,
)

PARKED_Z = -12.5
"""Halfway to a parking yard at -25, matching what the builder derives from a manifest."""


def write_sidecar(pool_dir, relations, references) -> None:
    """Write an env-graph sidecar into a pool directory.

    Args:
        pool_dir: Pool to write into.
        relations: Relation entries for the graph.
        references: ``(id, prim_path)`` pairs for the supports the relations name.
    """
    env_graph_path(pool_dir).write_text(
        json.dumps({
            "object_references": [
                {"id": name, "prim_path": path, "object_type": "base", "params": {}} for name, path in references
            ],
            "relations": relations,
        })
    )


def unit_square(owner_column: int, z_top: float = 0.7) -> SupportRegion:
    """Return a one-metre-square region centred on its owner's origin.

    Args:
        owner_column: Shard column of the prim carrying the region.
        z_top: Height of the surface in the owner's frame.
    """
    return SupportRegion(owner_column=owner_column, x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z_top=z_top)


def placement(regions, subject_columns=(0,), layouts=None, **overrides) -> SupportPlacement:
    """Build a placement whose layouts put each subject at a distinct spot on the surface.

    Args:
        regions: Candidate supports.
        subject_columns: Shard columns of the objects placed.
        layouts: ``(n_regions, n_layouts, n_subjects, 4)`` to use instead of the generated default.
        **overrides: Fields to set instead of the defaults.
    """
    regions = tuple(regions)
    if layouts is None:
        layouts = torch.zeros((len(regions), 3, len(subject_columns), LAYOUT_COLUMNS))
        for region_index, region in enumerate(regions):
            for layout in range(3):
                for subject in range(len(subject_columns)):
                    layouts[region_index, layout, subject, 0] = 0.1 * layout
                    layouts[region_index, layout, subject, 1] = 0.1 * subject
                    layouts[region_index, layout, subject, 2] = region.z_top + 0.05
    fields = {
        "subject_columns": tuple(subject_columns),
        "regions": regions,
        "layouts": layouts,
        "parked_below_z": PARKED_Z,
    }
    return SupportPlacement(**{**fields, **overrides})


def poses_with(
    standing: dict[int, tuple[float, float, float]],
    subjects: tuple[int, ...] = (0,),
    n_prims: int = 6,
    n_envs: int = 8,
) -> torch.Tensor:
    """Return a pose block whose named columns stand at a position and the rest are parked.

    Args:
        standing: Position per shard column for the supports that stand.
        subjects: Columns of the objects being placed, which the pool stands wherever it likes
            because Arena is about to overwrite them.
        n_prims: Columns in the block.
        n_envs: Rows in the block.
    """
    poses = torch.zeros((n_envs, n_prims, 7))
    poses[:, :, 2] = -25.0
    poses[:, :, 6] = 1.0
    for column, position in standing.items():
        poses[:, column, :3] = torch.tensor(position)
    for column in subjects:
        poses[:, column, 2] = 0.0
    return poses


class TestReadingTheSidecar:
    """The pool names the surface and what belongs on it; no geometry crosses the boundary."""

    def test_every_declared_support_is_returned_in_order(self, tmp_path):
        """A pool stands one of several candidate supports, so all of them have to come back."""
        write_sidecar(
            tmp_path,
            relations=[
                {"kind": "on", "subject": "mug_00", "reference": "drawer_a", "params": {}},
                {"kind": "on", "subject": "mug_00", "reference": "drawer_b", "params": {}},
            ],
            references=[("drawer_a", "cab_00/Floor"), ("drawer_b", "cab_01/Floor")],
        )
        assert load_support_prim_paths(tmp_path, "mug_00") == ("cab_00/Floor", "cab_01/Floor")

    def test_the_drawer_contents_are_discovered_rather_than_restated(self, tmp_path):
        """An environment naming the clutter itself would go stale when the producer added to it."""
        write_sidecar(
            tmp_path,
            relations=[
                {"kind": "on", "subject": "mug_00", "reference": "drawer_a", "params": {}},
                {"kind": "on", "subject": "spoon_00", "reference": "drawer_a", "params": {}},
                {"kind": "is_anchor", "subject": "drawer_a", "reference": None, "params": {}},
                {"kind": "on", "subject": "mug_00", "reference": "drawer_b", "params": {}},
            ],
            references=[("drawer_a", "cab_00/Floor"), ("drawer_b", "cab_01/Floor")],
        )
        assert load_placed_subjects(tmp_path) == ("mug_00", "spoon_00"), "each subject once, in declaration order"

    def test_a_missing_sidecar_is_an_error_rather_than_an_empty_result(self, tmp_path):
        """Silently placing nothing would leave the drawer contents wherever the pool left them."""
        with pytest.raises(AssertionError, match="no env-graph sidecar"):
            load_support_prim_paths(tmp_path, "mug_00")

    def test_a_subject_with_no_support_is_an_error(self, tmp_path):
        """Naming an object Arena should place, that the pool gives nowhere to put, is a mistake."""
        write_sidecar(
            tmp_path,
            relations=[{"kind": "on", "subject": "kettle_00", "reference": "worktop", "params": {}}],
            references=[("worktop", "counter_00/Top")],
        )
        with pytest.raises(AssertionError, match="declares no 'on' relation"):
            load_support_prim_paths(tmp_path, "mug_00")


class TestChoosingASupport:
    """Which candidate is standing changes per variation, so it is read off the drawn poses."""

    def test_the_objects_land_on_the_standing_support(self):
        """The parked candidates are far below the floor and must never be placed onto."""
        poses = poses_with({4: (1.0, 2.0, 0.0)})
        place_on_supports(poses, placement([unit_square(3), unit_square(4)]), openness=None)
        assert torch.allclose(poses[:, 0, 2], torch.full((8,), 0.75)), "seated at the standing support's height"
        assert bool(((poses[:, 0, 0] - 1.0).abs() <= 0.5).all()), "inside the standing support's footprint"

    def test_each_environment_uses_its_own_standing_support(self):
        """Environments draw independently, so two may stand different candidates on the same reset."""
        poses = poses_with({3: (0.0, 0.0, 0.0), 4: (5.0, 0.0, 0.0)})
        poses[:4, 4, 2] = -25.0
        poses[4:, 3, 2] = -25.0
        place_on_supports(poses, placement([unit_square(3), unit_square(4)]), openness=None)
        assert bool((poses[:4, 0, 0].abs() <= 0.5).all()), "first half placed on the support standing for them"
        assert bool(((poses[4:, 0, 0] - 5.0).abs() <= 0.5).all()), "second half placed on theirs"

    def test_a_variation_standing_no_support_is_an_error(self):
        """Placing the objects in the parking yard would strand the episode with nothing to pick."""
        with pytest.raises(AssertionError, match="none of the objects' supports standing"):
            place_on_supports(poses_with({}), placement([unit_square(3)]), openness=None)


class TestApplyingALayout:
    """Layouts are solved in the support's frame, so its drawn pose has to be applied."""

    def test_the_support_rotation_turns_the_layout_with_it(self):
        """A cabinet against a side wall is turned, and its drawer's contents turn with it."""
        poses = poses_with({3: (0.0, 0.0, 0.0)})
        # A quarter turn about z, in the pool's (x, y, z, w) order.
        poses[:, 3, 3:7] = torch.tensor([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
        offset = torch.zeros((1, 1, 1, LAYOUT_COLUMNS))
        offset[0, 0, 0, 0] = 2.0
        place_on_supports(poses, placement([unit_square(3)], layouts=offset), openness=None)
        assert bool((poses[:, 0, 0].abs() < 0.01).all()), "the support's own +x now points along the world +y"
        assert bool((poses[:, 0, 1] > 1.9).all()), "so the layout sits out along +y"

    def test_openness_slides_the_contents_out_with_the_drawer(self):
        """The base scene is measured with the drawer shut, so the whole driven travel applies."""
        shut, opened = poses_with({3: (0.0, 0.0, 0.0)}), poses_with({3: (0.0, 0.0, 0.0)})
        sliding = placement(
            [unit_square(3)], carry_axis=(-1.0, 0.0, 0.0), carry_travel=0.4, carry_scalar="drawer_openness"
        )
        # The same draw either side, so what is left between them is the carry and not the layout.
        torch.manual_seed(0)
        place_on_supports(shut, sliding, openness=torch.zeros(8))
        torch.manual_seed(0)
        place_on_supports(opened, sliding, openness=torch.ones(8))
        assert torch.allclose(opened[:, 0, 0], shut[:, 0, 0] - 0.4), "slid by the full travel"
        assert torch.allclose(opened[:, 0, 1:3], shut[:, 0, 1:3]), "and only along the joint's axis"

    def test_a_different_layout_is_drawn_per_environment(self):
        """The point of solving several is that the drawer is not arranged identically every episode."""
        poses = poses_with({3: (0.0, 0.0, 0.0)}, n_envs=64)
        place_on_supports(poses, placement([unit_square(3)]), openness=None)
        assert poses[:, 0, 0].std() > 0.01, "environments drew different layouts"

    def test_the_objects_are_upright_whatever_yaw_the_layout_gives(self):
        """A mug on its side is a different task, and the layouts were solved flat on the surface."""
        poses = poses_with({3: (0.0, 0.0, 0.0)}, n_envs=32)
        turned = torch.zeros((1, 1, 1, LAYOUT_COLUMNS))
        turned[0, 0, 0, 3] = 1.2
        place_on_supports(poses, placement([unit_square(3)], layouts=turned), openness=None)
        assert torch.allclose(poses[:, 0, 3:5], torch.zeros((32, 2))), "no roll or pitch"
        assert torch.allclose(poses[:, 0, 3:7].norm(dim=1), torch.ones(32)), "and still a unit quaternion"
        assert torch.allclose(2.0 * torch.atan2(poses[:, 0, 5], poses[:, 0, 6]), torch.full((32,), 1.2)), "solved yaw"


class TestASubsetOfTheContents:
    """A variation stands some of the drawer's superset and parks the rest."""

    def test_parked_objects_keep_their_parking_pose(self):
        """A layout solved for the full set stays valid for a subset, so parked ones just stay put."""
        poses = poses_with({3: (0.0, 0.0, 0.0)}, subjects=(0,))
        parked_before = poses[:, 1, :].clone()
        place_on_supports(poses, placement([unit_square(3)], subject_columns=(0, 1)), openness=None)
        assert torch.allclose(poses[:, 1, :], parked_before), "the parked object was left in the yard"
        assert bool((poses[:, 0, 2] > 0.0).all()), "while the standing one was placed on the surface"

    def test_every_standing_object_is_placed(self):
        """All of the drawer's standing contents come from Arena, not just the task's own object."""
        poses = poses_with({4: (0.0, 0.0, 0.0)}, subjects=(0, 1, 2))
        place_on_supports(poses, placement([unit_square(4)], subject_columns=(0, 1, 2)), openness=None)
        assert bool((poses[:, :3, 2] > 0.0).all()), "none left in the parking yard"
        spread = poses[0, :3, 1]
        assert len(torch.unique(spread)) == 3, f"the three objects share a spot rather than a layout: {spread}"
