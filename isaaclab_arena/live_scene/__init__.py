# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Live scene generation: consume externally generated scenes and pooled variations.

A companion generator process authors a fixed-topology base scene plus a continuously refilled
pool of variations. Arena loads the base scene as a background and draws one variation per
environment at reset. Nothing on the reset path opens USD; see
:mod:`isaaclab_arena.live_scene.pool_contract` for the on-disk interface.
"""
