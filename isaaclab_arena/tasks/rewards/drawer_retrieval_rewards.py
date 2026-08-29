# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dense rewards for taking an object out of a drawer and setting it down on a surface.

The task has stages that a single distance reward cannot express: the drawer has to come open before
the object is reachable at all, and the object has to be off its starting surface before moving it
towards a goal means anything. Each stage here is dense on its own and gated on the previous one, so
the policy is never rewarded for a later stage it has not earned.

A generated scene holds its alternatives as a parked superset, so which of several candidate cabinets
a given environment is using changes on every reset. Everything here resolves that per environment
from the episode-start record rather than being told, which is what lets one reward term serve a pool
that draws a different cabinet each time.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.tasks.episode_start_pose import episode_start_position

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_PARKED_DEPTH_M = 5.0
"""How far below an environment's own floor a prim has to start to count as parked.

A pool parks what a variation did not use tens of metres down, so anything within a few metres of the
floor is in the room. Well clear of both, so neither a tall cabinet nor a deep parking offset needs
this changed.
"""

_CACHE_ATTRIBUTE = "_arena_drawer_retrieval_cache"
"""Attribute the resolved joint indices are stashed under on the environment."""


@dataclass
class _ResolvedDrawers:
    """Joint indices for each candidate cabinet, resolved once rather than per step."""

    joint_ids: dict[str, list[int]] = field(default_factory=dict)


def _resolve_joint(env: ManagerBasedRLEnv, cabinet: str, joint_name: str) -> list[int]:
    """Return a cabinet's joint indices for a named joint, resolving on first use.

    Args:
        env: The RL environment instance.
        cabinet: Scene entity name of the cabinet.
        joint_name: Joint to find, which must match exactly one.
    """
    cache = getattr(env, _CACHE_ATTRIBUTE, None)
    if cache is None:
        cache = _ResolvedDrawers()
        setattr(env, _CACHE_ATTRIBUTE, cache)
    key = f"{cabinet}/{joint_name}"
    if key not in cache.joint_ids:
        found, names = env.scene[cabinet].find_joints(joint_name)
        assert len(found) == 1, f"joint {joint_name!r} matched {names} on {cabinet}; expected exactly one"
        cache.joint_ids[key] = list(found)
    return cache.joint_ids[key]


def placed_cabinet_column(env: ManagerBasedRLEnv, cabinets: tuple[str, ...]) -> torch.Tensor:
    """Return which candidate cabinet each environment placed, as an index into ``cabinets``.

    Read from where the reset put each cabinet rather than from the pool's parked flags: a parked prim
    is posed tens of metres down, so the placed one is simply the highest. That keeps this usable by
    any consumer of the episode-start record, without threading the parked channel through to every
    reward term, and it answers sensibly before the first reset, when the record still holds the base
    scene's poses and every candidate is parked.

    Args:
        env: The RL environment instance.
        cabinets: Scene entity names of every candidate, in a fixed order.

    Returns:
        ``(num_envs,)`` long tensor indexing ``cabinets``.
    """
    heights = torch.stack([episode_start_position(env, name)[:, 2] for name in cabinets], dim=1)
    count = (heights > -_PARKED_DEPTH_M).sum(dim=1)
    assert bool((count <= 1).all()), (
        f"expected at most one of {list(cabinets)} in the room per environment, got counts "
        f"{count.unique(return_counts=True)}; a variation that places two cabinets overlaps them"
    )
    return heights.argmax(dim=1)


def drawer_extension(
    env: ManagerBasedRLEnv,
    cabinets: tuple[str, ...],
    joint_name: str,
    travel: float,
) -> torch.Tensor:
    """Return how far each environment's placed drawer stands open, as a fraction of its travel.

    Args:
        env: The RL environment instance.
        cabinets: Scene entity names of every candidate cabinet.
        joint_name: The drawer joint on each of them.
        travel: Joint position at fully open, in metres.

    Returns:
        ``(num_envs,)`` tensor clamped to ``[0, 1]``.
    """
    positions = torch.stack(
        [wp.to_torch(env.scene[name].data.joint_pos)[:, _resolve_joint(env, name, joint_name)[0]] for name in cabinets],
        dim=1,
    )
    chosen = positions.gather(1, placed_cabinet_column(env, cabinets).unsqueeze(1)).squeeze(1)
    return (chosen / travel).clamp(0.0, 1.0)


def drawer_extension_observation(
    env: ManagerBasedRLEnv,
    cabinets: tuple[str, ...],
    joint_name: str,
    travel: float,
) -> torch.Tensor:
    """Return :func:`drawer_extension` shaped ``(num_envs, 1)`` for an observation group.

    A reward term wants a scalar per environment and an observation term wants a row, so the shape is
    adapted here rather than by giving the measurement itself a shape only one caller can use.

    Args:
        env: The RL environment instance.
        cabinets: Scene entity names of every candidate cabinet.
        joint_name: The drawer joint on each of them.
        travel: Joint position at fully open, in metres.
    """
    return drawer_extension(env, cabinets, joint_name, travel).unsqueeze(1)


def drawer_opened(
    env: ManagerBasedRLEnv,
    cabinets: tuple[str, ...],
    joint_name: str,
    travel: float,
) -> torch.Tensor:
    """Reward pulling the placed drawer out, in proportion to how far it has come.

    Dense in the joint position, so a policy that has moved the drawer at all is paid for it. The
    pool may have started the drawer part-way open, and that head start is paid for too rather than
    being subtracted: crediting only the policy's own contribution would make an easy variation worth
    less in total than a hard one, and the annealing curriculum needs the two comparable.

    Args:
        env: The RL environment instance.
        cabinets: Scene entity names of every candidate cabinet.
        joint_name: The drawer joint on each of them.
        travel: Joint position at fully open, in metres.
    """
    return drawer_extension(env, cabinets, joint_name, travel)


def object_reachable_in_drawer(
    env: ManagerBasedRLEnv,
    std: float,
    open_enough: float,
    cabinets: tuple[str, ...],
    joint_name: str,
    travel: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing on the object, but only once the drawer is open enough to get at it.

    Ungated, this pays the policy for pressing the gripper against a shut drawer front, which is
    where the object is but not where a hand can go.

    Args:
        env: The RL environment instance.
        std: Width of the tanh kernel on the gripper-to-object distance.
        open_enough: Drawer extension, as a fraction of travel, at which the object counts as
            accessible.
        cabinets: Scene entity names of every candidate cabinet.
        joint_name: The drawer joint on each of them.
        travel: Joint position at fully open, in metres.
        object_cfg: The object being retrieved.
        ee_frame_cfg: The gripper frame to measure from.
    """
    object_pos = wp.to_torch(env.scene[object_cfg.name].data.root_pos_w)
    ee_pos = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_pos_w)[..., 0, :]
    distance = torch.norm(object_pos - ee_pos, dim=1)
    accessible = drawer_extension(env, cabinets, joint_name, travel) >= open_enough
    return accessible * (1.0 - torch.tanh(distance / std))


def object_lifted_clear(
    env: ManagerBasedRLEnv,
    clearance: float,
    surface_z: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward getting the object up out of the drawer, to within reach of the surface it goes on.

    Measured against the destination surface rather than against where the object started, unlike a
    plain lift: the object starts down inside a drawer, so rising a few centimetres from there is
    still inside the drawer and is not progress towards putting it on the worktop.

    Args:
        env: The RL environment instance.
        clearance: How far below the surface the object may still be and count as clear (m). A small
            allowance, so the reward arrives as the object comes over the lip rather than after it.
        surface_z: Height of the destination surface, in the env-local frame (m).
        object_cfg: The object being retrieved.
    """
    height = _object_height_env_local(env, object_cfg.name)
    return (height > surface_z - clearance).float()


def object_on_surface_distance(
    env: ManagerBasedRLEnv,
    std: float,
    clearance: float,
    surface_z: float,
    goal_x: float,
    goal_z: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward carrying the object to its place on the surface, once it is clear of the drawer.

    The goal is a point on the worktop directly out from where the object started: the drawer runs on
    one axis into the room, so the cabinet's position along the wall is the object's own starting
    coordinate on that wall, and a goal derived from it lands above the drawer the object came out of
    whichever cabinet the variation drew.

    Args:
        env: The RL environment instance.
        std: Width of the tanh kernel on the remaining distance.
        clearance: Passed to the same gate as :func:`object_lifted_clear`, so the two agree.
        surface_z: Height of the destination surface, in the env-local frame (m).
        goal_x: Where on the worktop's depth the object goes, in the env-local frame (m).
        goal_z: Height of the object's centre once it rests on the surface (m).
        object_cfg: The object being retrieved.
    """
    goal = episode_start_position(env, object_cfg.name).clone()
    goal[:, 0] = goal_x
    goal[:, 2] = goal_z
    position = wp.to_torch(env.scene[object_cfg.name].data.root_pos_w) - env.scene.env_origins
    clear = _object_height_env_local(env, object_cfg.name) > surface_z - clearance
    return clear * (1.0 - torch.tanh(torch.norm(goal - position, dim=1) / std))


def object_placed_on_surface(
    env: ManagerBasedRLEnv,
    tolerance: float,
    speed_tolerance: float,
    surface_z: float,
    goal_x: float,
    goal_z: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rl_training: bool = False,
) -> torch.Tensor:
    """Return which environments have the object resting in place on the surface.

    Rest is part of the condition, not decoration: an object passing through the goal in the gripper's
    grip, or bouncing across the worktop, has not been put down.

    Args:
        env: The RL environment instance.
        tolerance: How near the goal counts as placed (m).
        speed_tolerance: How slowly the object has to be moving to count as at rest (m/s).
        surface_z: Height of the destination surface, in the env-local frame (m).
        goal_x: Where on the worktop's depth the object goes, in the env-local frame (m).
        goal_z: Height of the object's centre once it rests on the surface (m).
        object_cfg: The object being retrieved.
        rl_training: Answer False everywhere, for use as a termination term during training so that a
            policy which puts the object down goes on being paid for having it there. Anything reading
            the condition itself, a curriculum for instance, leaves this alone.
    """
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    goal = episode_start_position(env, object_cfg.name).clone()
    goal[:, 0] = goal_x
    goal[:, 2] = goal_z
    asset = env.scene[object_cfg.name]
    position = wp.to_torch(asset.data.root_pos_w) - env.scene.env_origins
    speed = torch.norm(wp.to_torch(asset.data.root_lin_vel_w), dim=1)
    return (torch.norm(goal - position, dim=1) < tolerance) & (speed < speed_tolerance)


def object_below_surface(
    env: ManagerBasedRLEnv,
    drop_below: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return which environments have dropped the object on the floor.

    An absolute floor height rather than a fall relative to where the object started, because it
    starts inside a drawer well below the surface it is going to.

    Args:
        env: The RL environment instance.
        drop_below: Env-local height under which the object counts as dropped (m).
        object_cfg: The object being retrieved.
    """
    return _object_height_env_local(env, object_cfg.name) < drop_below


def _object_height_env_local(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    """Return an asset's height in its own environment's frame, shape ``(num_envs,)``.

    Args:
        env: The RL environment instance.
        asset_name: Scene entity name of the asset.
    """
    return wp.to_torch(env.scene[asset_name].data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
