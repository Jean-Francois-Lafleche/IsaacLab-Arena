# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Move a pool scalar's value as the policy earns it.

A generated pool draws its difficulty per variation, which is the right default: a policy trained at
one difficulty has seen one problem. But the hardest end of a range can be unreachable from scratch --
nothing in a shut drawer rewards a policy that has never opened one -- so this pins the scalar at the
easy end and gives ground only once the policy is succeeding.

Driven by measured success rather than by a step schedule. A schedule advances whether or not the
policy improved, and the run that needs the curriculum most is exactly the one a schedule outpaces.
"""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.live_scene.variation_events import LIVE_VARIATION_RESET_EVENT_NAME

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_arena.live_scene.variation_events import ScalarDrivenJoint

_STATE_ATTRIBUTE = "_arena_variation_curriculum"
"""Attribute the running success rate is stashed under on the environment."""


@dataclass
class _AnnealState:
    """Running success rate, and the value the scalar is currently pinned to."""

    success_rate: float
    value: float
    active: bool = True
    """False when the scalar was already pinned before the curriculum first ran, so it stays put."""


def _joint_for_scalar(env: ManagerBasedRLEnv, scalar: str) -> ScalarDrivenJoint:
    """Return the reset target reading a named scalar.

    Args:
        env: The RL environment instance.
        scalar: Pool scalar the target reads.
    """
    term = env.event_manager.get_term_cfg(LIVE_VARIATION_RESET_EVENT_NAME)
    targets = term.params.get("scalar_driven_joints", ())
    matching = [target for target in targets if target.scalar == scalar]
    assert len(matching) == 1, (
        f"expected one reset target reading {scalar!r}, found {len(matching)}; the pool declares "
        f"{[target.scalar for target in targets]}"
    )
    return matching[0]


def anneal_scalar_on_success(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    scalar: str,
    success_func: Callable[..., torch.Tensor],
    success_params: dict,
    start: float = 1.0,
    end: float = 0.0,
    promote_above: float = 0.5,
    step: float = 0.1,
    smoothing: float = 0.02,
) -> float:
    """Step a pool scalar from ``start`` towards ``end`` whenever the policy is succeeding.

    Runs as a curriculum term, which Isaac Lab calls before the reset overwrites the environment, so
    ``succeeded`` sees how the finished episode actually ended.

    One value for the whole run, not one per environment: the vectorised environments share a policy,
    and a per-environment difficulty would have the easy ones dominate the batch as soon as the hard
    ones started failing.

    Stands aside if the scalar is already pinned when it first runs, which is how an explicit override
    on the command line beats a config's standing request for a curriculum.

    Args:
        env: The RL environment instance.
        env_ids: Environments being reset, whose outcomes are the sample.
        scalar: Pool scalar to pin, which the pool's manifest must declare.
        success_func: Reads the finished episodes' outcome, returning a ``(num_envs,)`` boolean. Give
            the task's own success condition, so the curriculum advances on the thing being asked for.
        success_params: Keyword arguments for ``success_func``, besides the environment. Kept separate
            from the function rather than bound into it, because Isaac Lab writes a run's whole term
            configuration out as YAML and a bound callable has no name to write.
        start: Value to pin at first. For an openness scalar this is the easy end.
        end: Value to stop at.
        promote_above: Success rate at which the value takes another step.
        step: How much of the range one promotion covers.
        smoothing: Weight of one batch of outcomes in the running rate. Small, so a lucky batch does
            not promote a policy that cannot hold the difficulty it has.

    Returns:
        The value now pinned, which Isaac Lab logs as the term's value.
    """
    state = getattr(env, _STATE_ATTRIBUTE, None)
    if state is None:
        joint = _joint_for_scalar(env, scalar)
        # An override already in place was asked for explicitly, on the command line, which is a more
        # specific instruction than a config's standing request for a curriculum. Stand aside and keep
        # reporting the pinned value, so the logs show which difficulty the run is actually at.
        if joint.override is not None:
            state = _AnnealState(success_rate=0.0, value=float(joint.override), active=False)
        else:
            state = _AnnealState(success_rate=0.0, value=float(start))
            joint.override = state.value
        setattr(env, _STATE_ATTRIBUTE, state)

    if not state.active:
        return state.value

    if len(env_ids):
        batch = float(success_func(env, **success_params)[env_ids].float().mean())
        state.success_rate += smoothing * (batch - state.success_rate)

    if state.success_rate > promote_above and state.value != end:
        direction = 1.0 if end > start else -1.0
        state.value = min(state.value + step, end) if direction > 0 else max(state.value - step, end)
        # Reset the estimate rather than carrying it: the rate that earned the promotion was measured
        # on the easier problem, and keeping it would promote again a few resets later on no evidence.
        state.success_rate = 0.0
        _joint_for_scalar(env, scalar).override = state.value

    return state.value
