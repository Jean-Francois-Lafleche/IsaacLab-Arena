# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The scene's contract distributions as Arena run-time variations.

Each distribution of the contract's ``scene`` group becomes a :class:`ContractVariation` on one variation host,
``contract``, so it works like any Arena variation: enabled by default, listed by ``--list_variations``,
overridden with Hydra (``contract.<name>.sampler_cfg.low=[...]``, ``contract.<name>.choices=[...]``,
``contract.<name>.enabled=false``) and recorded per episode under ``variations``.

Arena owns the draws: at every reset the contract's reset event samples each enabled variation with its
Arena sampler and pins the values in the contract request; the contract solver then realizes a valid scene
around them (:mod:`isaaclab_arena.contract.realize`). A disabled variation, and a distribution Arena has no
sampler for, is drawn by the contract itself from the episode's seed.

The default sampler is the contract's own distribution: a uniform's bounds, a categorical's choices, an
integer uniform's range. Arena's choice sampler is uniform, so a weighted categorical or a bernoulli keeps its
probabilities by repeating choices (a bernoulli of 0.6 is three ``True`` and two ``False``).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.sampler_base import SamplerBaseCfg
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

HOST = "contract"
"""The variation host every contract variation is registered on (the Hydra prefix)."""


@configclass
class ContractVariationCfg(VariationBaseCfg):
    """A contract distribution's Arena sampler: bounds for a uniform, choices for a choice."""

    enabled: bool = True
    sampler_cfg: SamplerBaseCfg = field(default_factory=SamplerBaseCfg)
    choices: list = field(default_factory=list)
    """For a choice: the values drawn from, repeated to keep the contract's weights."""


@dataclass
class DistributionSpec:
    """What a contract distribution is, as Arena needs it. ``cfg.enabled`` is off for one the solver searches."""

    name: str
    path: str
    kind: str  # "uniform" (float or vector) or "choice"
    value_type: str
    targets: tuple[str, ...] = ()
    components: int = 1
    cfg: ContractVariationCfg | None = None


def _weighted(values: list, weights: list[float], max_denominator: int = 20) -> list:
    """``values`` repeated in proportion to ``weights`` (to the nearest fraction with a small denominator)."""
    total = float(sum(weights))
    fractions = [Fraction(w / total).limit_denominator(max_denominator) for w in weights]
    common = 1
    for f in fractions:
        common = common * f.denominator // __import__("math").gcd(common, f.denominator)
    out = []
    for value, f in zip(values, fractions):
        out += [value] * int(f * common)
    return out


def _group_of(node, by_path: dict) -> str:
    """The contract group a distribution is drawn with: its own, or for an inline one its operator's."""
    group = str(getattr(node, "group", "") or "")
    if not group and "/Operators/" in str(node.path):
        operator = by_path.get(str(node.path).rsplit("/", 1)[0])
        group = str(getattr(operator, "group", "") or "") if operator is not None else ""
    if not group:
        return "scene"  # an operator outside every group runs with the scene, once
    return group.rstrip("/").rsplit("/", 1)[-1]


def distribution_specs(contract, group: str = "scene") -> tuple[list[DistributionSpec], list[str]]:
    """The contract's ``group`` distributions Arena can sample, and the paths of those it leaves to the contract.

    Args:
        contract: A read contract (``usd_randomizer.contract.Contract``).
        group: The contract group whose distributions become variations.
    """
    specs: list[DistributionSpec] = []
    left: list[str] = []
    taken: set[str] = set()
    by_path = {str(n.path): n for n in contract.nodes}
    # Distributions a layout operator searches over (a placement's offsets, a populate's count): the
    # contract's solver redraws them until the scene fits, which a pinned value cannot. Arena may still
    # draw them (enable the variation), and an infeasible draw is then resampled.
    searched: set[str] = set()
    for node in contract.nodes:
        if node.role == "operator" and str(node.type).startswith("layout."):
            for edge in node.edges:
                if edge.name == "distribution":
                    searched.update(str(t) for t in edge.targets)
    for node in contract.nodes:
        if not node.is_distribution or node.type == "branch":
            continue
        path = str(node.path)
        if _group_of(node, by_path) != group:
            continue
        fields, family, value_type = node.fields, node.type, node.value_type
        leaf = path.rsplit("/", 1)[-1]
        name = path.split("/Operators/")[1].split("/")[0] if leaf == "Distribution" and "/Operators/" in path else leaf
        while name in taken:
            name = name + "_"
        if any(str(k).endswith(":from") for k in fields) or family not in ("uniform", "categorical", "bernoulli"):
            left.append(path)
            continue
        taken.add(name)
        targets: tuple[str, ...] = ()
        if fields.get("valuesPer") == "target":
            edge = next((e for e in node.edges if e.name == "targets"), None)
            targets = tuple(edge.targets) if edge is not None else ()
        if family == "uniform" and value_type in ("float", "float[]"):
            low = fields["minimum"] if isinstance(fields["minimum"], (tuple, list)) else (fields["minimum"],)
            high = fields["maximum"] if isinstance(fields["maximum"], (tuple, list)) else (fields["maximum"],)
            repeat = max(1, len(targets))
            cfg = ContractVariationCfg(
                sampler_cfg=UniformSamplerCfg(
                    low=[float(v) for v in low] * repeat, high=[float(v) for v in high] * repeat
                )
            )
            cfg.enabled = path not in searched
            specs.append(DistributionSpec(name, path, "uniform", value_type, targets, len(low), cfg))
        elif family == "uniform" and value_type == "int":
            choices = list(range(int(fields["minimum"]), int(fields["maximum"]) + 1))
            cfg = ContractVariationCfg(sampler_cfg=ChoiceSamplerCfg(), choices=choices, enabled=path not in searched)
            specs.append(DistributionSpec(name, path, "choice", value_type, targets, 1, cfg))
        elif family == "categorical":
            values = list(fields["choices"])
            weights = fields.get("weights")
            choices = _weighted(values, list(weights)) if weights else values
            specs.append(DistributionSpec(name, path, "choice", value_type, targets, 1,
                                          ContractVariationCfg(sampler_cfg=ChoiceSamplerCfg(), choices=choices)))  # fmt: skip
        elif family == "bernoulli":
            p = float(fields.get("probability", 0.5))
            choices = _weighted([True, False], [p, 1.0 - p])
            specs.append(DistributionSpec(name, path, "choice", value_type, targets, 1,
                                          ContractVariationCfg(sampler_cfg=ChoiceSamplerCfg(), choices=choices)))  # fmt: skip
        else:
            left.append(path)
    return specs, left


def _no_op(env, env_ids) -> None:
    """The variation's own event does nothing: the contract's reset event samples it and realizes the scene."""


class ContractVariation(RunTimeVariationBase):
    """One contract distribution, sampled by Arena at every reset and pinned in the contract's request."""

    cfg: ContractVariationCfg

    def __init__(self, spec: DistributionSpec):
        super().__init__(cfg=spec.cfg, name=spec.name)
        self.spec = spec

    def draw(self, count: int) -> list[Any]:
        """``count`` raw draws, not yet reported to listeners (the reset event may redraw some)."""
        if self.spec.kind == "choice":
            assert self.cfg.choices, f"contract variation {self.name!r} has no choices to draw from"
            return list(self.sampler._sample(count, list(self.cfg.choices)))
        return list(torch.as_tensor(self.sampler._sample(count)).reshape(count, -1))

    def record(self, raws: list[Any], env_ids: torch.Tensor) -> None:
        """Report the draws the episode actually uses to Arena's listeners (the variation recorder), once."""
        sample = list(raws) if self.spec.kind == "choice" else torch.stack(list(raws))
        self.sampler._notify(sample, env_ids)

    def pin_value(self, raw: Any) -> Any:
        """A raw draw as a contract pin: a choice as is, a uniform as a float, a vector, or a value per target."""
        if self.spec.kind == "choice":
            return raw
        row = [float(v) for v in torch.as_tensor(raw).reshape(-1)]
        if self.spec.targets:
            k = self.spec.components
            if self.spec.value_type == "float":
                return {t: row[i] for i, t in enumerate(self.spec.targets)}
            return {t: tuple(row[i * k : (i + 1) * k]) for i, t in enumerate(self.spec.targets)}
        if self.spec.value_type == "float":
            return row[0]
        return tuple(row)

    def raw_from_draws(self, draws, fallback: Any) -> Any:
        """The contract's own value for this distribution in ``draws``, shaped like :meth:`draw`'s output.

        Used when the contract drew an environment's scene itself, so the episode records what it got.
        ``fallback`` is kept for a distribution the sample left inactive (a branch it did not take).
        """
        try:
            if self.spec.targets:
                values = [draws.value(self.spec.path, t) for t in self.spec.targets]
            else:
                values = [draws.value(self.spec.path)]
        except KeyError:
            return fallback
        if any(v is None for v in values):
            return fallback
        if self.spec.kind == "choice":
            return values[0]
        flat = [float(x) for v in values for x in (v if isinstance(v, (list, tuple)) else [v])]
        return torch.as_tensor(flat, dtype=torch.as_tensor(fallback).dtype)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        # Arena builds one event per enabled variation; this one is a placeholder, because the scene a value
        # leads to depends on every draw together and is realized once by the contract's reset event.
        return f"{HOST}_{self.name}_sample", EventTermCfg(func=_no_op, mode="reset")
