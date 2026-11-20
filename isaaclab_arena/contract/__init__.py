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

The contract's distributions are Arena variations (``contract.<distribution>``), defaulting to the contract's
priors. At every reset Arena samples them, the contract's solver realizes a valid scene with them pinned, and
the contract's per-episode (``reset``) group follows, drawn the same way. ``--pool`` names the contract
template's pool directory (``layout.json``, which prim and attribute each value writes); with
``--scene_draws pool`` every reset plays back one of its verified rows instead.

What the contract's placements place (a mug in a drawer, a board on a countertop) is searched by the
contract's solver by default. With ``--placer arena`` Arena's relation solver places it instead: each placement
becomes an Arena placement problem (the support an anchor, the target ``On`` it within the contract's box), its
layouts are drawn like any other variation, and the contract applies and checks them
(:mod:`isaaclab_arena.contract.arena_placement`).

The contract is executed by ``isaaclab_contract`` (the contract's Isaac Lab executor), which is an optional
dependency of Arena: only this package and the ``contract`` environment import it.
"""
