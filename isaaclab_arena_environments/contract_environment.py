# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Any scene that carries a randomization contract and an ``mdp@1`` MDP root.

The scene file states its own task, so this one environment runs every such file; nothing in it is
written per task. See :mod:`isaaclab_arena.contract`. For example, with a contract variation pool::

    python isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_steps 900 \\
        --num_envs 4 contract --scene kitchen_franka.usda --live_variation_pool_dir kitchen_franka/pool
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class ContractEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure an environment read from a scene's contract and MDP root."""

    scene: str = ""
    """Scene file with a randomization contract and an ``mdp@1`` MDP root."""

    mdp_root: str | None = None
    """The MDP root's prim path, when the file holds several."""

    contract_seed: int = 0
    """Seed of the contract's per-episode (``reset`` group) draws."""

    scene_draws: str = "arena"
    """``arena``: the contract's scene distributions are Arena variations (``--list_variations`` lists them,
    Hydra overrides them), sampled at every reset and realized by the contract's solver. ``pool``: every reset
    plays back a row of the variation pool."""

    live_variation_pool_dir: str | None = None
    """Set by the shared Arena parser. The contract's template layout (``layout.json``), and with
    ``scene_draws = pool`` its rows. ``None`` runs every environment on the file's own sample."""

    num_envs: int = 1
    """Set by the shared Arena parser."""


@register_environment
class ContractEnvironment(ArenaEnvironmentFactory[ContractEnvironmentCfg]):
    """Run a scene file that describes its own task."""

    name: str = "contract"
    _legacy_argparse_cfg_type = ContractEnvironmentCfg

    def build(self, cfg: ContractEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.contract.mdp_environment import build_contract_environment
        from isaaclab_arena.contract.pool import is_contract_pool

        assert cfg.scene, "the contract environment needs --scene <file with a contract and an mdp@1 MDP root>"
        pool = cfg.live_variation_pool_dir
        assert pool is None or is_contract_pool(pool), (
            f"{pool} is not a contract variation pool (no layout.json); an Arena manifest pool belongs to the"
            " live-scene environments"
        )
        return build_contract_environment(
            self.name,
            cfg.scene,
            pool_dir=pool,
            mdp_root=cfg.mdp_root,
            num_envs=cfg.num_envs,
            seed=cfg.contract_seed,
            scene_draws=cfg.scene_draws if pool is not None else "pool",
        )
