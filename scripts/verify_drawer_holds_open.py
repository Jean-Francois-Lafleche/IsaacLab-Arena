# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hold each task drawer at a mid-travel opening and report whether it stays there.

An earlier check watched a drawer that had been opened all the way, which rests against its own upper
limit -- a hard stop will hold a drawer whether or not anything else would. Commanding a partial
opening removes that stop from the picture and leaves only what the joint itself provides, which is
what a variation at low difficulty and a drawer pulled half out in Kit both rely on.
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pool", required=True, help="Live variation pool directory to draw from.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--opening", type=float, default=0.20, help="Commanded opening in metres.")
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--tolerance_m", type=float, default=0.01)
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


def main() -> int:
    """Build the env, set every task drawer part-open, and report where each one ends up."""
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

    cabinets = {
        name: articulation
        for name, articulation in sorted(env.unwrapped.scene.articulations.items())
        if "task_cabinet" in name
    }

    tracked = []
    for name, articulation in cabinets.items():
        index = next((i for i, j in enumerate(articulation.joint_names) if j == "DrawerJoint_0"), None)
        if index is None:
            continue
        state = articulation.data.joint_pos.clone()
        state[:, index] = args.opening
        articulation.write_joint_state_to_sim(state, torch.zeros_like(state))
        # Gains rather than gravity are what usually hold a joint still, so report what this one has.
        stiffness = float(articulation.data.joint_stiffness[0, index])
        damping = float(articulation.data.joint_damping[0, index])
        tracked.append((name, articulation, index, stiffness, damping))

    print(f"\n[setup] commanding {len(tracked)} drawers to {args.opening:.3f} m")
    for name, _, _, stiffness, damping in tracked[:3]:
        print(f"[setup] {name}/DrawerJoint_0 drive stiffness={stiffness:.3f} damping={damping:.3f}")

    zero = torch.zeros_like(env.unwrapped.action_manager.action)
    history = {name: [] for name, *_ in tracked}
    for _ in range(args.steps):
        env.step(zero)
        for name, articulation, index, _, _ in tracked:
            history[name].append(float(articulation.data.joint_pos[0, index]))

    print()
    closed = []
    for name, *_ in tracked:
        track = history[name]
        stride = max(1, len(track) // 10)
        print(f"[{name}] {args.opening:.3f} -> {track[-1]:.4f} m")
        print(f"  {' '.join(f'{v:.3f}' for v in track[::stride])}")
        if track[-1] < args.opening - args.tolerance_m:
            closed.append(name)

    print(f"\n[check] {len(closed)} of {len(tracked)} drawers moved shut by more than {args.tolerance_m:.3f} m")
    ok = not closed
    print("[PASS] drawers held a partial opening" if ok else "[FAIL] drawers do not hold a partial opening")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
