# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A contract's distributions as Arena variations: the default samplers follow the contract's priors.

Builds a small contract in memory with ``usd_randomizer.contract`` (skipped when it is not importable) and
checks each family's mapping: a uniform keeps its bounds, an integer uniform becomes a choice over its range,
a categorical keeps its choices and weights, a bernoulli its probability, a per-target uniform repeats its
bounds per target and pins per target, and a distribution a placement searches is off by default.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("usd_randomizer") is None, reason="usd_randomizer (the contract package) is not importable"
)


def _contract():
    from pxr import Gf, Usd, UsdGeom
    from usd_randomizer import contract as rc

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.05))
    UsdGeom.Cube.Define(stage, "/World/Box").AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
    w = rc.author(stage)
    w.float("depth", -0.6, 0.4)
    w.int("count", 0, 2)
    w.choice("style", ["a", "b", "c"], weights=[2.0, 1.0, 1.0])
    w.bool("present", probability=0.6)
    w.floats("offset", ["x", "y"], [-0.1, -0.2], [0.1, 0.2], values_per="target", targets=["/World/Box"])
    pose = w.floats("pose", ["x", "y", "yaw"], [-0.2, -0.2, -90.0], [0.2, 0.2, 90.0], values_per="target",
                    targets=["/World/Box"])  # fmt: skip
    w.placement("place", pose, on="/World/Table", targets=["/World/Box"])
    (contract,), problems = rc.read(stage)
    assert contract.ok and not problems, (contract.errors, problems)
    return contract


def test_each_family_keeps_the_contracts_prior():
    from isaaclab_arena.contract.variations import distribution_specs

    specs, left = distribution_specs(_contract())
    by_name = {s.name: s for s in specs}
    assert not left

    depth = by_name["depth"]
    assert depth.kind == "uniform"
    assert (depth.cfg.sampler_cfg.low, depth.cfg.sampler_cfg.high) == ([-0.6], [0.4])

    assert by_name["count"].kind == "choice" and by_name["count"].cfg.choices == [0, 1, 2]
    # weights 2:1:1 kept by repetition
    assert sorted(by_name["style"].cfg.choices) == ["a", "a", "b", "c"]
    present = by_name["present"].cfg.choices
    assert present.count(True) / len(present) == pytest.approx(0.6)

    offset = by_name["offset"]
    assert offset.targets == ("/World/Box",)
    assert (offset.cfg.sampler_cfg.low, offset.cfg.sampler_cfg.high) == ([-0.1, -0.2], [0.1, 0.2])

    # the placement searches its pose: Arena leaves it to the contract unless enabled
    assert by_name["pose"].cfg.enabled is False
    assert all(by_name[n].cfg.enabled for n in ("depth", "count", "style", "present", "offset"))


def test_a_draw_becomes_the_contracts_pin_and_is_recorded_once():
    import torch

    from isaaclab_arena.contract.variations import ContractVariation, distribution_specs

    specs, _ = distribution_specs(_contract())
    variations = {s.name: ContractVariation(s) for s in specs}
    heard = []
    variations["offset"].add_sample_listener(lambda sample, env_ids: heard.append((sample, env_ids)))

    raws = variations["offset"].draw(3)
    pins = [variations["offset"].pin_value(r) for r in raws]
    assert all(set(p) == {"/World/Box"} and len(p["/World/Box"]) == 2 for p in pins)
    assert all(-0.1 <= p["/World/Box"][0] <= 0.1 and -0.2 <= p["/World/Box"][1] <= 0.2 for p in pins)
    assert not heard  # drawing alone does not report: the reset event may redraw
    variations["offset"].record(raws, torch.tensor([0, 1, 2]))
    assert len(heard) == 1 and heard[0][0].shape == (3, 2)

    style = variations["style"]
    assert all(v in ("a", "b", "c") for v in (style.pin_value(r) for r in style.draw(5)))
    assert isinstance(variations["depth"].pin_value(variations["depth"].draw(1)[0]), float)


def test_a_contract_drawn_scene_is_recorded_as_drawn():
    """When the contract draws an environment itself, the recorded values are the ones its scene was built from."""
    from isaaclab_arena.contract.variations import ContractVariation, distribution_specs

    specs, _ = distribution_specs(_contract())
    variations = {s.name: ContractVariation(s) for s in specs}
    by_leaf = {s.name: s.path for s in specs}
    truth = {by_leaf["depth"]: {"": 0.25}, by_leaf["style"]: {"": "c"}, by_leaf["present"]: {"": False},
             by_leaf["offset"]: {"/World/Box": (0.05, -0.15)}}  # fmt: skip

    class Draws:
        def value(self, path, target=""):
            return truth[path][target]

    for name in ("depth", "style", "present", "offset"):
        v = variations[name]
        arena = v.draw(1)[0]
        raw = v.raw_from_draws(Draws(), arena)
        expected = next(iter(truth[v.spec.path].values()))
        got = v.pin_value(raw)
        if isinstance(got, dict):
            got = got["/World/Box"]
        if isinstance(expected, (str, bool)):
            assert got == expected, name
        else:
            assert got == pytest.approx(expected), name
    # a distribution the sample left inactive keeps Arena's draw
    count = variations["count"]
    arena = count.draw(1)[0]
    assert count.raw_from_draws(Draws(), arena) == arena  # KeyError -> fallback


def test_a_function_of_other_draws_stays_the_contracts():
    """A function distribution (a curve of another draw) is not an Arena variation: it follows what Arena draws."""
    from pxr import Usd, UsdGeom
    from usd_randomizer import contract as rc

    from isaaclab_arena.contract.variations import distribution_specs

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Sun")
    w = rc.author(stage)
    zenith = w.float("zenith", 50.0, 72.0)
    w.distribution("temperature", "curve", distribution=zenith, curveIn=[50.0, 72.0], curveOut=[6500.0, 5000.0])
    (contract,), problems = rc.read(stage)
    assert contract.ok and not problems, (contract.errors, problems)

    specs, left = distribution_specs(contract)
    assert [s.name for s in specs] == ["zenith"]
    assert [p.rsplit("/", 1)[-1] for p in left] == ["temperature"]
