# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scenes that describe their own task: a randomization contract and an ``mdp@1`` MDP root.

A scene file can carry what varies in it (a randomization contract, read with ``usd_randomizer.contract``)
and the task a robot does there (an ``mdp@1`` MDP root: observations, actions, rewards, terminations, task
stages, curriculum and metrics). Arena runs such a file through the ``contract`` environment
(:mod:`isaaclab_arena_environments.contract_environment`) instead of an environment class written per task:

- the **embodiment** is the robot the scene already holds (the MDP root's ``Scene/<name>`` term), with the
  MDP root's observations and actions;
- the **scene** is every other entity of the file, cloned per environment;
- the **task** is the MDP root's rewards, terminations (``success`` from its ``Task`` stages), curriculum,
  reset events and ``Metrics``.

A variation pool written for the contract (a directory with ``layout.json``) is passed with Arena's own
``--live_variation_pool_dir``; every reset then applies a fresh pool row, followed by the contract's
per-episode (``reset``) group.

The contract is executed by ``isaaclab_contract`` (the contract's Isaac Lab executor), which is an optional
dependency of Arena: only this package and the ``contract`` environment import it.
"""
