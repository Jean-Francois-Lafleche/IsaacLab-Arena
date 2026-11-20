# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The ``contract`` episode recorder: what each finished episode's scene was.

Arena's variation recorder already writes what Arena sampled under ``variations``, the block its sensitivity
analysis reads. This term adds, under ``contract``, the scene the contract realized from those samples (every
channel, named), the contract seed that regenerates it, and who drew it: ``arena`` (its samples, pinned),
``contract`` (Arena's samples were infeasible after retries) or ``pool`` (played back from a variation pool).
"""

from __future__ import annotations

from typing import Any

from isaaclab.utils.configclass import configclass

from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg


def record_contract_episode(env, env_id: int) -> dict[str, Any]:
    """The finished episode's contract seed, who drew it, and the scene the contract realized."""
    starts = getattr(env, "_contract_episode_starts", None)
    start = starts.current.get(int(env_id)) if starts is not None else None
    if start is None:
        return {}
    # Arena's own ``variations`` block holds what Arena sampled; the scene the contract realized from it
    # (every channel, named) goes beside the seed that regenerates it.
    return {
        "contract": {
            "seed": start["seed"],
            "level": start["level"],
            "drawn_by": start["drawn_by"],
            "realized": start["realized"],
        }
    }


@configclass
class ContractEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Record each finished episode's contract seed, pool level and scene factors."""

    func: Any = record_contract_episode
