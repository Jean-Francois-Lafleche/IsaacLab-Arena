# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that a fixed-base task cabinet still lands where the pool put it, and then stays there.

Fixing a cabinet's base is what stops the arm shoving it across the room, but it only works if a pose
written to the articulation root still re-anchors it: a fixed base that ignores the write would stand
every environment's cabinet at the same authored spot, and a pool that varies casework would silently
stop varying it. Run against a pool directory built by the kitchen producer.
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pool", required=True, help="Live variation pool directory to draw from.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--settle_steps", type=int, default=60, help="Steps to watch the cabinet for drift.")
parser.add_argument("--drift_tolerance_m", type=float, default=0.002)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg  # noqa: E402
from isaaclab_arena_environments.pick_place_kitchen_live_environment import (  # noqa: E402
    PickPlaceKitchenLiveEnvironment,
)


def widest_standing_cabinet(env, parked_below_z: float) -> str:
    """Return the task cabinet that the most environments have standing, so the sample is largest."""
    best, best_count = None, 0
    for name, articulation in env.unwrapped.scene.articulations.items():
        if "task_cabinet" not in name:
            continue
        count = int((articulation.data.root_pose_w[:, 2] > parked_below_z).sum())
        if count > best_count:
            best, best_count = name, count
    assert best is not None, "no environment has a task cabinet standing"
    print(f"[check] {best} stands in {best_count} of {args.num_envs} environments")
    return best


def main() -> int:
    """Build the env, then report whether cabinet poses vary per env and hold still."""
    factory = PickPlaceKitchenLiveEnvironment()
    environment_cfg = factory._legacy_argparse_cfg_type()
    environment_cfg.live_variation_pool_dir = args.pool
    cfg = ArenaEnvBuilderCfg(
        num_envs=args.num_envs,
        env_spacing=6.0,
        live_variation_pool_dir=args.pool,
        live_variation_placed_objects=("*",),
    )
    builder = ArenaEnvBuilder(factory.build(environment_cfg), cfg)
    env = builder.make_registered()
    env.reset()

    term = next(
        cfg
        for cfg in vars(env.unwrapped.cfg.events).values()
        if getattr(cfg, "params", None) and "variation_pool" in cfg.params
    )
    pool = term.params["variation_pool"].pool
    parked_below_z = float(pool.parking_offset[2]) / 2.0

    name = widest_standing_cabinet(env, parked_below_z)
    articulation = env.unwrapped.scene.articulations[name]
    print(f"[check] {name}: is_fixed_base={articulation.is_fixed_base}")

    first = articulation.data.root_pose_w.clone()
    # Only where it stands. A parked cabinet sits 25 m down in the yard, which would swamp both the
    # spread this compares and the drift it measures.
    standing = first[:, 2] > parked_below_z
    local = first[standing, :2] - env.unwrapped.scene.env_origins[standing, :2]
    per_axis = (local.max(dim=0).values - local.min(dim=0).values) if int(standing.sum()) > 1 else local[:1] * 0
    spread = float(per_axis.max())
    print(f"[check] env-local spread where it stands: x={float(per_axis[0]):.4f} m  y={float(per_axis[1]):.4f} m")
    print(f"[check] env-local positions: {[(round(float(a), 3), round(float(b), 3)) for a, b in local.tolist()]}")

    # Step by step, and only where the episode did not just restart. A reset redraws the variation,
    # which parks the cabinet that was standing 25 m down: measured across a window that contains one,
    # the "drift" is the parking depth and says nothing about whether the carcass is anchored.
    drawer_joints = [i for i, joint in enumerate(articulation.joint_names) if "Drawer" in joint]
    opening_before = articulation.data.joint_pos[standing][:, drawer_joints].clone()

    zero = torch.zeros_like(env.unwrapped.action_manager.action)
    drift, resets = 0.0, 0
    previous = articulation.data.root_pose_w.clone()
    for _ in range(args.settle_steps):
        env.step(zero)
        current = articulation.data.root_pose_w
        held = env.unwrapped.episode_length_buf > 0
        resets += int((~held).sum())
        if bool(held.any()):
            step_drift = (current[held, :3] - previous[held, :3]).norm(dim=-1)
            drift = max(drift, float(step_drift.max()))
        previous = current.clone()
    print(f"[check] worst single-step drift over {args.settle_steps} steps: {drift * 1000:.3f} mm")
    print(f"[check] env-steps skipped for a reset: {resets}")

    # The drawer closing itself was the symptom that started this. It is driven to an opening the pool
    # chose, so what matters is that it is still there after sixty steps of nobody touching it.
    opening_after = articulation.data.joint_pos[standing][:, drawer_joints]
    creep = float((opening_after - opening_before).abs().max()) if drawer_joints else 0.0
    print(f"[check] widest drawer opening: {float(opening_after.max()):.4f} m")
    print(f"[check] worst drawer creep over the window: {creep * 1000:.2f} mm")
    if creep > 0.005:
        print("[FAIL] the drawer moved on its own")
        ok_drawer = False
    else:
        ok_drawer = True

    ok = ok_drawer
    if not articulation.is_fixed_base:
        print("[FAIL] PhysX did not make the cabinet fixed-base, so the carcass is still free to move")
        ok = False
    if int(standing.sum()) > 1 and spread < 0.01:
        print("[FAIL] every environment stood its cabinet in the same place: the pose write was ignored")
        ok = False
    if drift > args.drift_tolerance_m:
        print(f"[FAIL] the cabinet moved more than {args.drift_tolerance_m * 1000:.0f} mm")
        ok = False
    print("[PASS] fixed base accepts the pool's pose and holds it" if ok else "[FAIL]")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
