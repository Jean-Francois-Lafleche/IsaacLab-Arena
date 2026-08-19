# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for the live scene generation modes of the agentic environment generation runner.

Every case runs the runner as a child process. The argument-validation cases would otherwise call
``sys.exit`` inside pytest, and the ``live`` case starts Isaac Sim.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from isaaclab_arena.agentic_environment_generation.scene_prompt_inference import (
    SCENE_REQUEST_FILENAME,
    load_scene_request,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import skip_without_live_endpoint_key
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

CLI_RUNNER = f"{TestConstants.repo_root}/isaaclab_arena_examples/agentic_environment_generation/cli_runner.py"

NUM_ENVS = 2
NUM_STEPS = 4
NUM_VARIATIONS = 8


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the runner and return the completed process without asserting on its exit code.

    Args:
        *args: Arguments appended after the script path.
    """
    return subprocess.run(
        [TestConstants.python_path, CLI_RUNNER, *args],
        capture_output=True,
        text=True,
        timeout=600,
    )


def _build_fixture_pool(pool_dir: pathlib.Path) -> None:
    """Write a fixture pool, including its base scene, in a child process.

    The base scene needs ``pxr``, and importing it into the pytest process before Isaac Sim starts
    is what the repo's test guidance avoids, so the fixture is built out of process.

    Args:
        pool_dir: Directory to create the pool in.
    """
    script = (
        "from isaaclab_arena.tests.utils import variation_pool_fixtures as fixtures;"
        f"fixtures.build_fixture_pool({str(pool_dir)!r}, num_variations={NUM_VARIATIONS},"
        " with_base_scene=True)"
    )
    run_subprocess([TestConstants.python_path, "-c", script])


@skip_without_live_endpoint_key()
def test_scene_prompt_mode_writes_a_scene_request(tmp_path: pathlib.Path):
    """The handoff to the scene generator: a task prompt becomes a scene_request.json on disk.

    Needs a live endpoint, because the split is the model's judgement and stubbing it here would only
    re-test what the unit tests already cover.
    """
    result = _run_cli(
        "--mode",
        "scene_prompt",
        "--prompt",
        "teach a Franka to put away a coffee mug from a countertop to a drawer in any kitchen",
        "--out_dir",
        str(tmp_path),
    )

    assert result.returncode == 0, f"scene_prompt mode failed:\n{result.stdout}\n{result.stderr}"
    request = load_scene_request(tmp_path / SCENE_REQUEST_FILENAME)
    assert request.scene_prompt and request.task_prompt
    # The split is only useful if the halves differ; one prompt echoed twice is a failed split.
    assert request.scene_prompt != request.task_prompt


def test_the_runner_advertises_the_live_scene_modes():
    """The two new modes are the entry points to the whole live scene path."""
    result = _run_cli("--help")

    assert result.returncode == 0, result.stderr
    assert "scene_prompt" in result.stdout
    assert "live" in result.stdout
    assert "--variation_pool" in result.stdout


def test_live_mode_without_a_pool_is_rejected():
    """``live`` has no spec and no agent, so the pool is the only thing it can build from."""
    result = _run_cli("--mode", "live", "--headless")

    assert result.returncode != 0
    assert "requires --variation_pool" in result.stdout + result.stderr


def test_live_mode_with_a_missing_pool_directory_is_rejected(tmp_path: pathlib.Path):
    """A typo'd pool path should be reported before Isaac Sim spends a minute starting up."""
    result = _run_cli("--mode", "live", "--headless", "--variation_pool", str(tmp_path / "absent"))

    assert result.returncode != 0
    assert "variation pool directory not found" in result.stdout + result.stderr


def test_live_mode_rejects_a_pool_with_no_manifest(tmp_path: pathlib.Path):
    """A directory that is not a pool must fail as such, not as a missing-asset error later on."""
    pool_dir = tmp_path / "empty_pool"
    pool_dir.mkdir()

    result = _run_cli("--mode", "live", "--headless", "--variation_pool", str(pool_dir))

    assert result.returncode != 0
    assert "has no manifest at" in result.stdout + result.stderr


@pytest.mark.with_subprocess
def test_live_mode_builds_and_steps_an_env_from_a_variation_pool(tmp_path: pathlib.Path):
    """The end-to-end consumer path: a pool on disk becomes a stepping environment.

    Covers what the unit and simulation tests cannot: that the runner wires the manifest, the
    background, the derived object references, and the reset event together without a graph spec.
    """
    pool_dir = tmp_path / "pool"
    _build_fixture_pool(pool_dir)

    result = _run_cli(
        "--mode",
        "live",
        "--headless",
        "--num_envs",
        str(NUM_ENVS),
        "--num_steps",
        str(NUM_STEPS),
        "--variation_pool",
        str(pool_dir),
    )

    assert result.returncode == 0, f"live mode failed:\n{result.stdout}\n{result.stderr}"
    assert "variable prims" in result.stdout, "the runner did not report the pool it loaded"
    assert "variation pool stats" in result.stdout, "the runner did not report pool utilisation"
