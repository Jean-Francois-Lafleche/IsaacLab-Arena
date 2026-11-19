# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The ``contract`` environment: a scene file with a randomization contract and an ``mdp@1`` MDP root.

Pool detection runs in-process. The end-to-end case runs the policy runner in a child process (it starts
Isaac Sim) on a scene named by ``ARENA_CONTRACT_SCENE`` (and its pool, ``ARENA_CONTRACT_POOL``), and is
skipped when they are unset or ``isaaclab_contract`` is not importable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess

import pytest

from isaaclab_arena.contract.pool import is_contract_pool
from isaaclab_arena.tests.utils.constants import TestConstants

POLICY_RUNNER = f"{TestConstants.repo_root}/isaaclab_arena/evaluation/policy_runner.py"


def test_a_contract_pool_is_told_apart_from_an_arena_manifest_pool(tmp_path: pathlib.Path):
    contract_pool = tmp_path / "contract"
    contract_pool.mkdir()
    (contract_pool / "layout.json").write_text(json.dumps({"channels": []}))
    arena_pool = tmp_path / "arena"
    arena_pool.mkdir()
    (arena_pool / "manifest.json").write_text("{}")
    assert is_contract_pool(contract_pool)
    assert not is_contract_pool(arena_pool)
    assert not is_contract_pool(tmp_path / "missing")
    assert not is_contract_pool(None)


@pytest.mark.skipif(
    not os.environ.get("ARENA_CONTRACT_SCENE") or importlib.util.find_spec("isaaclab_contract") is None,
    reason="set ARENA_CONTRACT_SCENE (and ARENA_CONTRACT_POOL) with isaaclab_contract on PYTHONPATH",
)
def test_a_scene_that_describes_its_task_runs_and_reports_its_metrics():
    args = [
        TestConstants.python_path, POLICY_RUNNER, "--headless", "--policy_type", "zero_action",
        "--num_steps", "450", "--num_envs", "2", "contract", "--scene", os.environ["ARENA_CONTRACT_SCENE"],
    ]  # fmt: skip
    if os.environ.get("ARENA_CONTRACT_POOL"):
        args += ["--live_variation_pool_dir", os.environ["ARENA_CONTRACT_POOL"]]
    result = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    metrics = [line for line in output.splitlines() if "Metrics:" in line]
    assert metrics, output[-4000:]
    assert "success_rate" in metrics[-1]
    assert "'num_episodes': 0" not in metrics[-1], metrics[-1]
