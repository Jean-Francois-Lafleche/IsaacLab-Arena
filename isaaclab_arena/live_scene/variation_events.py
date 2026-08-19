# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reset event that applies a pooled scene variation to the resetting environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.live_scene.variation_pool import LiveVariationPool, LiveVariationPoolHandle

LIVE_VARIATION_RESET_EVENT_NAME = "live_variation_reset"
"""Name of the reset event term that owns the live variation pool."""


def get_live_variation_pool(env) -> LiveVariationPool | None:
    """Return the variation pool stored on the env reset event, or ``None`` when absent.

    Lets a runtime caller reach the pool from the env alone, for example to read starvation counters.

    Args:
        env: The gym-wrapped Isaac Lab env; the base env is reached via ``env.unwrapped``.
    """
    try:
        term_cfg = env.unwrapped.event_manager.get_term_cfg(LIVE_VARIATION_RESET_EVENT_NAME)
    except ValueError:
        return None
    handle = term_cfg.params.get("variation_pool")
    assert handle is not None, f"'{LIVE_VARIATION_RESET_EVENT_NAME}' event is missing its variation_pool parameter."
    return handle.pool


def apply_scene_variation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    variation_pool: LiveVariationPoolHandle,
    write_plan: list[tuple[int, str]],
) -> None:
    """Draw one variation per resetting environment and write it to the simulator.

    The whole path is tensor writes: a gather out of the pool's device-resident ring buffer, one
    batched pose write per prim covering every resetting environment at once, and no USD authoring.
    Parked prims already carry the pool's parking offset, so they cost nothing extra here.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        variation_pool: Opaque handle to the live variation pool.
        write_plan: ``(inventory index, scene entity name)`` pairs for the prims Arena can write.
            Built once at build time so the reset path does no lookup or filtering.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    pool = variation_pool.pool
    num_scene_envs = env.scene.env_origins.shape[0]
    assert (
        pool.num_envs == num_scene_envs
    ), f"Variation pool has {pool.num_envs} envs, but scene has {num_scene_envs} env origins."

    reset_env_ids = env_ids.tolist()
    poses = pool.gather_poses(reset_env_ids).to(device=env.device)
    # Pool poses are env-local; the simulator wants world coordinates.
    poses = poses.clone()
    poses[:, :, :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    zero_velocity = torch.zeros((len(reset_env_ids), 6), device=env.device)

    for inventory_index, scene_name in write_plan:
        scene_asset = env.scene[scene_name]
        scene_asset.write_root_pose_to_sim(poses[:, inventory_index, :], env_ids=env_ids)
        scene_asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
