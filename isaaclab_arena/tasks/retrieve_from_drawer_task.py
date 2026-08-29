# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Take an object out of a drawer and set it down on the surface above it.

Trainable rather than evaluation-only, so the reward is staged: pull the drawer open, close on the
object once it is reachable, bring it clear of the drawer, then carry it to its place on the surface.
Each stage is dense and gated on the one before it, because a policy paid for a later stage it has not
reached learns to sit in the cheapest one.

The drawer's starting openness is the task's difficulty and belongs to the scene, not here: a
generated pool draws it per variation and publishes it, and a curriculum may pin it. This task only
needs to know which joint to measure, so that its reward agrees with whatever the scene did.
"""

from __future__ import annotations

import numpy as np
from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.live_scene.variation_curriculum import anneal_scalar_on_success
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.observations import observations
from isaaclab_arena.tasks.rewards import drawer_retrieval_rewards as drawer_rewards
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@configclass
class RetrieveFromDrawerObservationsCfg:
    """Observations for the retrieve-from-drawer task."""

    task_obs: ObsGroup = MISSING

    def __init__(self, retrieved_object: Asset, robot_name: str, drawer_params: dict):
        @configclass
        class TaskObsCfg(ObsGroup):
            object_position = ObsTerm(
                func=observations.object_position_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "object_cfg": SceneEntityCfg(retrieved_object.name),
                },
            )
            drawer_extension = ObsTerm(func=drawer_rewards.drawer_extension_observation, params=dict(drawer_params))

        self.task_obs = TaskObsCfg()


@configclass
class RetrieveFromDrawerRewardCfg:
    """Reward terms for the retrieve-from-drawer task.

    The weights are ordered by stage, each worth more than the one before, so that finishing a stage
    always beats lingering in the previous one. Action-rate and joint-velocity penalties are the
    embodiment's to contribute, and Arena merges them in alongside these.
    """

    opening_drawer: RewardTermCfg = MISSING
    reaching_object: RewardTermCfg = MISSING
    clearing_drawer: RewardTermCfg = MISSING
    placing_object: RewardTermCfg = MISSING
    placing_object_fine_grained: RewardTermCfg = MISSING

    def __init__(
        self,
        retrieved_object: Asset,
        ee_frame_name: str,
        drawer_params: dict,
        surface_z: float,
        clearance: float,
        goal_x: float,
        goal_z: float,
    ):
        object_cfg = SceneEntityCfg(retrieved_object.name)
        self.opening_drawer = RewardTermCfg(
            func=drawer_rewards.drawer_opened,
            params=dict(drawer_params),
            weight=4.0,
        )
        self.reaching_object = RewardTermCfg(
            func=drawer_rewards.object_reachable_in_drawer,
            params={
                "std": 0.1,
                "open_enough": 0.5,
                **drawer_params,
                "object_cfg": object_cfg,
                "ee_frame_cfg": SceneEntityCfg(ee_frame_name),
            },
            weight=2.0,
        )
        self.clearing_drawer = RewardTermCfg(
            func=drawer_rewards.object_lifted_clear,
            params={"clearance": clearance, "surface_z": surface_z, "object_cfg": object_cfg},
            weight=15.0,
        )
        self.placing_object = RewardTermCfg(
            func=drawer_rewards.object_on_surface_distance,
            params={
                "std": 0.3,
                "clearance": clearance,
                "surface_z": surface_z,
                "goal_x": goal_x,
                "goal_z": goal_z,
                "object_cfg": object_cfg,
            },
            weight=16.0,
        )
        self.placing_object_fine_grained = RewardTermCfg(
            func=drawer_rewards.object_on_surface_distance,
            params={
                "std": 0.05,
                "clearance": clearance,
                "surface_z": surface_z,
                "goal_x": goal_x,
                "goal_z": goal_z,
                "object_cfg": object_cfg,
            },
            weight=5.0,
        )


@configclass
class RetrieveFromDrawerTerminationsCfg:
    """Termination terms for the retrieve-from-drawer task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    object_dropped: TerminationTermCfg = MISSING
    success: TerminationTermCfg = MISSING


@configclass
class RetrieveFromDrawerCurriculumCfg:
    """Anneal the scene's difficulty scalar as the policy succeeds."""

    drawer_difficulty: CurriculumTermCfg = MISSING


@register_task
class RetrieveFromDrawerTaskRL(TaskBase):
    """Retrieve an object from a drawer and place it on the surface above, with a staged dense reward."""

    def __init__(
        self,
        retrieved_object: Asset,
        embodiment: EmbodimentBase,
        drawer_articulations: tuple[str, ...],
        drawer_joint_name: str,
        drawer_travel: float,
        surface_z: float = 0.0,
        goal_x: float = 0.5,
        resting_offset: float = 0.0,
        lift_clearance: float = 0.05,
        floor_z: float = -0.9,
        position_tolerance: float = 0.08,
        episode_length_s: float = 12.0,
        difficulty_scalar: str | None = None,
        rl_training_mode: bool = True,
    ):
        """Initialise the task.

        Args:
            retrieved_object: The object that starts in the drawer.
            embodiment: The robot embodiment.
            drawer_articulations: Scene entity names of every candidate cabinet the object's drawer
                might belong to. All of them, because a generated pool holds its alternatives as a
                parked superset and places one per reset; which one is worked out per environment.
            drawer_joint_name: The drawer joint on each of those cabinets.
            drawer_travel: Joint position at fully open, in metres.
            surface_z: Height of the surface the object goes onto, in the env-local frame.
            goal_x: Where on the surface's depth the object goes, in the env-local frame. Its position
                along the other axis comes from where the object started, so the goal lands above the
                drawer it came out of.
            resting_offset: Height of the object's origin above a surface it is resting on. Zero for an
                asset whose pivot is at its base, which SimReady props generally are.
            lift_clearance: How far below the destination surface the object may still be and count as
                clear of the drawer. A small allowance, so the reward arrives as the object comes over
                the lip rather than once it is already above the worktop.
            floor_z: Env-local floor height, under which the object counts as dropped.
            position_tolerance: How near the goal counts as placed (m).
            episode_length_s: Episode length. Longer than a plain lift, because opening the drawer and
                then moving the object are consecutive rather than concurrent.
            difficulty_scalar: Scene scalar to anneal from its easy end as the policy succeeds, or
                ``None`` to take the scene's own difficulty. The scene must publish the name; for a
                generated pool that means its manifest declares it.
            rl_training_mode: When true, success does not end the episode, so a policy that puts the
                object down keeps being paid for having it there.
        """
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Take the mug out of the drawer and put it on the countertop.",
        )
        assert drawer_articulations, "the task needs at least one candidate cabinet to measure"

        self.retrieved_object = retrieved_object
        self.embodiment = embodiment
        self.rl_training_mode = rl_training_mode

        drawer_params = {
            "cabinets": tuple(drawer_articulations),
            "joint_name": drawer_joint_name,
            "travel": float(drawer_travel),
        }
        goal_z = surface_z + resting_offset
        robot_name = embodiment.get_scene_key()

        self.observation_cfg = RetrieveFromDrawerObservationsCfg(
            retrieved_object=retrieved_object, robot_name=robot_name, drawer_params=drawer_params
        )
        self.rewards_cfg = RetrieveFromDrawerRewardCfg(
            retrieved_object=retrieved_object,
            ee_frame_name=embodiment.get_ee_frame_name(embodiment.get_arm_mode()),
            drawer_params=drawer_params,
            surface_z=surface_z,
            clearance=lift_clearance,
            goal_x=goal_x,
            goal_z=goal_z,
        )
        object_cfg = SceneEntityCfg(retrieved_object.name)
        self._success_params = {
            "tolerance": position_tolerance,
            "speed_tolerance": 0.05,
            "surface_z": surface_z,
            "goal_x": goal_x,
            "goal_z": goal_z,
            "object_cfg": object_cfg,
        }
        self.termination_cfg = RetrieveFromDrawerTerminationsCfg(
            object_dropped=TerminationTermCfg(
                func=drawer_rewards.object_below_surface,
                params={"drop_below": floor_z + lift_clearance, "object_cfg": object_cfg},
            ),
            # Kept registered even while training, when it answers False throughout, because the
            # success-rate metric needs the term to exist.
            success=TerminationTermCfg(
                func=drawer_rewards.object_placed_on_surface,
                params={**self._success_params, "rl_training": rl_training_mode},
            ),
        )

        self.curriculum_cfg = None
        if difficulty_scalar is not None:
            self.curriculum_cfg = RetrieveFromDrawerCurriculumCfg(
                drawer_difficulty=CurriculumTermCfg(
                    func=anneal_scalar_on_success,
                    params={
                        "scalar": difficulty_scalar,
                        # The same predicate the success termination uses, so what the curriculum
                        # measures and what the task calls success cannot come apart.
                        "success_func": drawer_rewards.object_placed_on_surface,
                        "success_params": dict(self._success_params),
                    },
                )
            )

    def get_observation_cfg(self) -> Any:
        return self.observation_cfg

    def get_rewards_cfg(self) -> Any:
        return self.rewards_cfg

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_curriculum_cfg(self) -> Any:
        return self.curriculum_cfg

    def get_scene_cfg(self) -> Any:
        # The generated scene brings its own room; the task adds nothing to it.
        return None

    def get_events_cfg(self) -> Any:
        # The variation pool owns the reset, through the event term the builder installs for it.
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Any:
        raise NotImplementedError("retrieve_from_drawer has no Mimic config; it is trained with RL.")

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.retrieved_object,
            offset=np.array([-2.0, -2.0, 1.5]),
        )
