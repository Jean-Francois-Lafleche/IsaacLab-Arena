# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report the joint limits Isaac Lab reads for a pool's task cabinets.

A hinge whose upper limit has been overwritten is welded shut in all but name, and neither the
authoring nor the physics complains: the door simply refuses to move. Reading the limits back off the
built articulation is the only check that says what the solver will actually allow. Run against a pool
directory built by the kitchen producer.
"""

from __future__ import annotations

import argparse
import math

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pool", required=True, help="Live variation pool directory to draw from.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--min_swing_deg", type=float, default=60.0, help="Swing a door needs to be usable.")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg  # noqa: E402
from isaaclab_arena_environments.pick_place_kitchen_live_environment import (  # noqa: E402
    PickPlaceKitchenLiveEnvironment,
)


def main() -> int:
    """Build the env and print every cabinet DOF's limits, flagging hinges that cannot open."""
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

    hinges, cramped = 0, []
    for name, articulation in sorted(env.unwrapped.scene.articulations.items()):
        if "task_cabinet" not in name:
            continue
        limits = articulation.data.joint_pos_limits[0]
        print(f"\n[{name}]")
        for index, joint_name in enumerate(articulation.joint_names):
            lower, upper = float(limits[index, 0]), float(limits[index, 1])
            if "Hinge" in joint_name:
                hinges += 1
                span = math.degrees(upper - lower)
                print(
                    f"  {joint_name:<24} [{math.degrees(lower):8.2f}, {math.degrees(upper):8.2f}] deg"
                    f"  swing={span:6.1f}"
                )
                if span < args.min_swing_deg:
                    cramped.append(f"{name}/{joint_name} ({span:.1f} deg)")
            else:
                print(f"  {joint_name:<24} [{lower:8.4f}, {upper:8.4f}] m")

    print(f"\n[check] {hinges} hinges, {len(cramped)} with less than {args.min_swing_deg:.0f} deg of swing")
    for entry in cramped:
        print(f"  [FAIL] {entry}")
    ok = hinges > 0 and not cramped
    print("[PASS] every hinge has room to open" if ok else "[FAIL] a hinge cannot open")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
