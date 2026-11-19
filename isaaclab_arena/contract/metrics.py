# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena metrics for an ``mdp@1`` MDP root's ``Metrics`` terms.

Each ``Metrics/<metric>`` names an Arena metric in ``mdp:func``:

- ``success_rate`` maps to :class:`~isaaclab_arena.metrics.success_rate.SuccessRateMetric`, which reads the
  ``success`` termination the MDP root's ``Task`` stages produce;
- ``condition_rate`` is :class:`ConditionRateMetric`: the fraction of episodes in which a contract
  condition held (at any policy step, or at the last one).

A term Arena cannot compute is reported and left out, never approximated.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.success_rate import SuccessRateMetric


class ConditionRecorder(RecorderTerm):
    """Per episode, whether a contract condition held: at any policy step (``when = any``) or the last one."""

    def __init__(self, cfg: ConditionRecorderCfg, env):
        super().__init__(cfg, env)
        self.held = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.last = torch.zeros_like(self.held)
        self.first_reset = True

    def record_post_step(self):
        from isaaclab_contract.conditions import evaluator

        self.last = evaluator(self._env, self.cfg.key).holds(self.cfg.condition)
        self.held |= self.last
        return None, None

    def record_pre_reset(self, env_ids):
        # The first reset starts the run: no episode has happened yet.
        if self.first_reset:
            self.first_reset = False
            self.held[env_ids] = False
            return None, None
        flags = (self.held if self.cfg.when == "any" else self.last)[env_ids].clone()
        self.held[env_ids] = False
        return self.cfg.name, flags


@configclass
class ConditionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ConditionRecorder
    name: str = "condition"
    key: str = ""
    """The key isaaclab_contract registered the MDP under."""
    condition: str = ""
    """The condition's prim path."""
    when: str = "any"
    """``any``: held at some policy step; ``end``: held at the episode's last policy step."""


def compute_condition_rate(recorded_metric_data: list[np.ndarray]) -> float:
    """The fraction of recorded episodes whose flag is set."""
    if not recorded_metric_data:
        return 0.0
    return float(np.mean(np.concatenate(recorded_metric_data)))


class ConditionRateMetric(MetricBase):
    """The fraction of episodes in which a contract condition held (``mdp@1`` ``condition_rate``)."""

    def __init__(self, name: str, key: str, condition: str, when: str = "any"):
        assert when in ("any", "end"), f"condition_rate 'when' must be 'any' or 'end', got {when!r}"
        self.name = name
        self.recorder_term_name = name
        self.key, self.condition, self.when = key, condition, when

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return ConditionRecorderCfg(name=self.name, key=self.key, condition=self.condition, when=self.when)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_condition_rate, params={}, recorder_term_name=self.recorder_term_name
        )


@dataclass
class MappedMetrics:
    """The Arena metrics for a ``Metrics`` scope, and the terms that have none."""

    metrics: list[MetricBase]
    unmapped: list[str]


def metrics_from_mdp_root(root, key: str) -> MappedMetrics:
    """Arena metrics for the MDP root prim ``root``'s ``Metrics`` terms.

    Args:
        root: The MDP root prim (``Usd.Prim``).
        key: The key isaaclab_contract registered the MDP under (conditions are read through it).
    """
    metrics: list[MetricBase] = []
    unmapped: list[str] = []
    scope = root.GetChild("Metrics")
    for term in scope.GetChildren() if scope else []:
        func = term.GetAttribute("mdp:func").Get()
        if func == "success_rate" and not term.GetRelationship("mdp:params:stage").GetTargets():
            metrics.append(SuccessRateMetric())
        elif func == "condition_rate":
            targets = term.GetRelationship("mdp:params:condition").GetTargets()
            assert len(targets) == 1, f"{term.GetPath()}: condition_rate needs one condition, got {targets}"
            when = term.GetAttribute("mdp:params:when")
            metrics.append(ConditionRateMetric(term.GetName(), key, str(targets[0]), when.Get() if when else "any"))
        else:
            unmapped.append(f"{term.GetName()} ({func})")
    return MappedMetrics(metrics, unmapped)
