# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_arena.tasks.episode_start_pose import height_above_episode_start


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object clear of wherever it started the episode.

    Args:
        env: The RL environment instance.
        minimal_height: How far above its starting height the object counts as lifted (m).
        object_cfg: The configuration of the object to track.
    """
    return torch.where(height_above_episode_start(env, object_cfg.name) > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel.

    Args:
        env: The RL environment instance.
        std: Width of the tanh kernel on the goal distance.
        minimal_height: How far above its starting height the object has to be before the goal
            tracking pays out (m).
        command_name: Name of the command holding the goal pose.
        robot_cfg: The robot the goal pose is expressed relative to.
        object_cfg: The configuration of the object to track.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - wp.to_torch(object.data.root_pos_w), dim=1)
    # rewarded if the object is lifted above the threshold
    return (height_above_episode_start(env, object_cfg.name) > minimal_height) * (1 - torch.tanh(distance / std))
