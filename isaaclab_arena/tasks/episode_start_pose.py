# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Where each environment's objects started the episode they are currently running.

A lift threshold only means something relative to the surface the object was resting on. Measuring it
from world zero works only while every environment shares one table height, which stops being true as
soon as a scene generator varies the furniture, or as soon as a variation starts an object inside a
bowl or on top of a stack. Recording the spawn pose per environment lets a task ask how far the object
has risen above its own starting point instead.
"""

from __future__ import annotations

import torch
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_RECORD_ATTRIBUTE = "_arena_episode_start_poses"
"""Attribute the record is stashed under on the environment."""


@dataclass
class EpisodeStartPoses:
    """Env-local poses the scene entities were placed at when their episode began."""

    columns: dict[str, int]
    """Scene entity name to its index along the entity axis of :attr:`poses`."""

    poses: torch.Tensor
    """Shape ``(num_envs, num_entities, 7)``: position and quaternion, in the env-local frame."""


def record_episode_start_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    entity_columns: Iterable[tuple[int, str]],
    poses: torch.Tensor,
) -> None:
    """Record the poses the resetting environments were just placed at.

    Whoever applies the reset poses is the only caller that knows them without reading them back, so
    recording happens there rather than in a follow-up event term that would have to be ordered behind
    it. The whole block lands in one scatter, so tracking an entity costs nothing per entity.

    Args:
        env: The environment being reset.
        env_ids: Environment indices being reset, indexing the first axis of the record.
        entity_columns: ``(column, scene entity name)`` pairs naming the columns worth remembering.
            Columns left unnamed stay recorded but cannot be looked up.
        poses: Shape ``(len(env_ids), num_columns, 7)`` env-local positions and quaternions.
    """
    record = getattr(env, _RECORD_ATTRIBUTE, None)
    if record is None:
        record = EpisodeStartPoses(
            columns={name: column for column, name in entity_columns},
            poses=torch.zeros((env.num_envs, poses.shape[1], 7), device=poses.device, dtype=poses.dtype),
        )
        setattr(env, _RECORD_ATTRIBUTE, record)
    record.poses[env_ids] = poses


def episode_start_position(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Return the env-local position each environment's asset started its episode at, ``(num_envs, 3)``.

    Falls back to the asset's authored spawn pose, which is the right answer for a scene whose layout
    is fixed rather than drawn per episode.

    Args:
        env: The environment to read.
        asset_name: Scene entity name of the asset.
    """
    record = getattr(env, _RECORD_ATTRIBUTE, None)
    if record is not None and asset_name in record.columns:
        return record.poses[:, record.columns[asset_name], :3]
    return wp.to_torch(env.scene[asset_name].data.default_root_pose)[:, :3]


def episode_start_height(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Return the env-local height each environment's asset started its episode at, shape ``(num_envs,)``.

    Falls back to the asset's authored spawn pose, which is the right answer for a scene whose layout
    is fixed rather than drawn per episode.

    Args:
        env: The environment to read.
        asset_name: Scene entity name of the asset.
    """
    return episode_start_position(env, asset_name)[:, 2]


def height_above_episode_start(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Return how far each environment's asset has risen above its starting height (m).

    Args:
        env: The environment to read.
        asset_name: Scene entity name of the asset.
    """
    current_height = wp.to_torch(env.scene[asset_name].data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
    return current_height - episode_start_height(env, asset_name)
