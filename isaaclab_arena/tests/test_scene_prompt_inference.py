# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for splitting a task prompt into a scene-generation request.

The LLM call is stubbed: what needs guarding is that an unusable answer is rejected rather than
passed downstream, and that the artifact the scene generator reads round-trips.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import EnvironmentGenerationAgent
from isaaclab_arena.agentic_environment_generation.scene_prompt_inference import (
    SCENE_REQUEST_FILENAME,
    SceneGenerationRequest,
    ScenePromptInference,
    load_scene_request,
    write_scene_request,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import chat_response

VALID_RESPONSE = {
    "scene_prompt": "A photorealistic kitchen with a countertop and a drawer unit within reach of a robot base.",
    "task_prompt": "Put the coffee mug from the countertop into the drawer.",
    "required_objects": ["coffee mug", "countertop", "drawer unit"],
    "embodiment_hint": "Franka",
}


class _StubBackend:
    """Returns a canned structured-output response instead of calling a model.

    Args:
        response: What :meth:`run_json` returns.
    """

    def __init__(self, response: dict[str, Any]):
        self._response = response
        self.requests: list[Any] = []

    def run_json(self, request: Any) -> dict[str, Any]:
        """Record the request and return the canned response."""
        self.requests.append(request)
        return self._response


def test_a_task_prompt_splits_into_both_halves():
    """Each half must be usable on its own: one sizes the scene, the other drives spec inference."""
    stage = ScenePromptInference(_StubBackend(VALID_RESPONSE))
    traces: list[str] = []

    request = stage.infer("put away a coffee mug in any kitchen", traces)

    assert request is not None
    assert request.scene_prompt == VALID_RESPONSE["scene_prompt"]
    assert request.task_prompt == VALID_RESPONSE["task_prompt"]
    assert request.required_objects == VALID_RESPONSE["required_objects"]
    assert traces == []


def test_the_task_prompt_reaches_the_model():
    """A split made without seeing the prompt would be fabricated."""
    backend = _StubBackend(VALID_RESPONSE)
    stage = ScenePromptInference(backend)

    stage.infer("put away a coffee mug in any kitchen", [])

    assert "put away a coffee mug in any kitchen" in backend.requests[0].user


def test_an_answer_missing_a_half_is_rejected():
    """Half a split is worse than none: the missing half would be silently dropped downstream."""
    stage = ScenePromptInference(_StubBackend({"scene_prompt": "a kitchen"}))
    traces: list[str] = []

    assert stage.infer("put away a mug", traces) is None
    assert len(traces) == 1
    assert "unusable request" in traces[0]


def test_an_empty_half_is_rejected():
    """An empty prompt reaches the generator as a request to invent the whole scene."""
    stage = ScenePromptInference(_StubBackend({**VALID_RESPONSE, "scene_prompt": ""}))
    traces: list[str] = []

    assert stage.infer("put away a mug", traces) is None
    assert traces


def test_the_scene_request_round_trips_through_disk(tmp_path: Path):
    """The request is the handoff to a separate process, so it has to survive serialization."""
    request = SceneGenerationRequest(**VALID_RESPONSE)

    path = write_scene_request(tmp_path, request)
    loaded = load_scene_request(path)

    assert path.name == SCENE_REQUEST_FILENAME
    assert loaded == request


def test_loading_a_request_that_was_never_written_fails_loudly(tmp_path: Path):
    """Pointing the generator at the wrong directory should say so, not generate from nothing."""
    with pytest.raises(AssertionError, match="no scene request at"):
        load_scene_request(tmp_path / SCENE_REQUEST_FILENAME)


def test_optional_fields_default_to_empty():
    """A prompt naming no robot is normal; requiring a hint would force one to be invented."""
    request = SceneGenerationRequest(scene_prompt="a kitchen", task_prompt="pick up the mug")

    assert request.required_objects == []
    assert request.embodiment_hint == ""


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def test_the_agent_exposes_the_split_as_its_first_pipeline_stage(stub_openai):
    """``generate_scene_request`` runs before ``generate_spec``, which only sees the residual half."""
    _, client = stub_openai
    client.chat.completions.create.return_value = chat_response(content=json.dumps(VALID_RESPONSE))
    agent = EnvironmentGenerationAgent(api_key="test-key")

    request = agent.generate_scene_request("put away a coffee mug in any kitchen")

    assert request is not None
    assert request.scene_prompt == VALID_RESPONSE["scene_prompt"]
    assert not agent.traces


def test_the_agent_reports_an_unusable_split_through_its_traces(stub_openai):
    """A caller has to be able to tell a failed split from a scene request it can act on."""
    _, client = stub_openai
    client.chat.completions.create.return_value = chat_response(content=json.dumps({"scene_prompt": "a kitchen"}))
    agent = EnvironmentGenerationAgent(api_key="test-key")

    assert agent.generate_scene_request("put away a mug") is None
    assert agent.traces, "the agent swallowed the failure instead of tracing it"
