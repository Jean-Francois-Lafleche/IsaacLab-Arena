# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn a task prompt into a scene-generation prompt plus the task intent that remains.

A task prompt such as "teach a Franka to put away a coffee mug from a countertop to a drawer in any
kitchen" mixes two requests: what the scenes must contain and vary over, which a scene generator
answers, and what the robot must do in them, which Arena's spec inference answers. This stage splits
them so each half receives only what it can act on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from isaaclab_arena.agentic_environment_generation.inference_backend import (
    InferenceBackend,
    StructuredOutputRequest,
    build_strict_schema,
)

_logger = logging.getLogger(__name__)

SCENE_REQUEST_FILENAME = "scene_request.json"
"""Artifact the scene generator consumes, written alongside the generated spec."""


class SceneGenerationRequest(BaseModel):
    """The two halves of a task prompt, plus what the generator needs to size the scene."""

    scene_prompt: str = Field(
        min_length=1,
        description=(
            "Prompt for the scene generator. Describes the scenes to generate, the assets every "
            "scene must contain, and the spatial constraints that must hold in every variation."
        ),
    )

    task_prompt: str = Field(
        min_length=1,
        description=(
            "The task intent that remains once scene composition is removed. Describes what the "
            "robot does, in scenes the generator has already guaranteed."
        ),
    )

    required_objects: list[str] = Field(
        default_factory=list,
        description="Objects every generated scene must contain, so a gate can check the generator honoured them.",
    )

    embodiment_hint: str = Field(
        default="",
        description="Robot the prompt names, if any. The generator uses it to size reachable regions.",
    )

    def to_json(self) -> str:
        """Return the request as indented JSON."""
        return json.dumps(self.model_dump(mode="json"), indent=2)


def write_scene_request(out_dir: str | Path, request: SceneGenerationRequest) -> Path:
    """Write a scene request into a directory.

    Args:
        out_dir: Directory to write into. Created when missing.
        request: The request to serialize.

    Returns:
        The written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SCENE_REQUEST_FILENAME
    path.write_text(request.to_json())
    return path


def load_scene_request(path: str | Path) -> SceneGenerationRequest:
    """Read and validate a scene request.

    Args:
        path: File written by :func:`write_scene_request`.
    """
    path = Path(path)
    assert path.is_file(), f"no scene request at {path}"
    return SceneGenerationRequest.model_validate_json(path.read_text())


class ScenePromptInference:
    """Splits a task prompt into a scene-generation prompt and the residual task intent."""

    def __init__(self, inference_backend: InferenceBackend):
        self._inference_backend = inference_backend
        self._schema = build_strict_schema(SceneGenerationRequest)

    def infer(self, prompt: str, traces: list[str]) -> SceneGenerationRequest | None:
        """Return the scene request for a task prompt, or ``None`` when the model's answer is unusable.

        Args:
            prompt: End-user task description.
            traces: Diagnostic trace list, appended to on failure.

        Returns:
            The validated request, or ``None``.
        """
        data = self._inference_backend.run_json(
            StructuredOutputRequest(
                schema_name="SceneGenerationRequest",
                schema=self._schema,
                system=self._system_prompt(),
                user=f"TASK PROMPT:\n{prompt}",
                retry_label="infer_scene_prompt",
            )
        )
        try:
            request = SceneGenerationRequest.model_validate(data)
        except Exception as error:
            traces.append(f"scene prompt inference returned an unusable request: {error}")
            return None
        _logger.info("scene prompt: %s", request.scene_prompt)
        _logger.info("residual task prompt: %s", request.task_prompt)
        return request

    @staticmethod
    def _system_prompt() -> str:
        return """\
You split a robot-learning prompt into a scene-generation request and the task that remains.

The scene generator you write for produces many randomised variations of one scene. Its topology is
fixed: every variation contains the same assets, and only their transforms change. So the scene
prompt must state what is present and what spatial relationships must hold, never what the robot
does or how an episode ends.

GUIDANCE:
- Put every asset the task touches, plus the surroundings it happens in, into the scene prompt.
  Ask for realistic, randomisable scenes rather than one fixed layout.
- State the spatial constraints that must hold in every variation, with numbers where the prompt
  implies them (e.g. a drawer within 50 cm of the robot base is reachable).
- Keep the task prompt to what the robot does. Do not restate the scene contents beyond naming the
  objects the task acts on, because the generator has already guaranteed they are there.
- List every object the scene must contain in required_objects, one short noun phrase each.
- Set embodiment_hint to the robot the prompt names, or leave it empty when it names none.
"""
