# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Profile the variation pool inside Isaac Sim, against a static scene of the same shape.

:mod:`isaaclab_arena.live_scene.pool_benchmark` measures the pool on its own, without Isaac Sim. That
answers whether the ring buffer is fast, but not the question a training run actually asks: what does
redrawing the layout every reset cost against simply leaving the scene where it was authored?

Two variants build the same generated scene and differ only in whether the pool owns placement:

* ``pool`` — ``live_variation_pool_dir`` is set, so the reset event writes a fresh variation.
* ``static`` — no pool, so every episode replays the poses baked into the base scene USD.

Both report environment build time, reset latency, and steady-state step time, which together
separate the cost of *having* variation from the cost of *changing* it. Build time is the number that
matters most: it is what a static-scene-per-layout approach would pay for every new layout, and what
the pool exists to avoid.

One variant per process, because the measurement is only clean on a fresh stage::

    python -m isaaclab_arena.live_scene.sim_variation_profile --headless --variant pool --num_envs 128
    python -m isaaclab_arena.live_scene.sim_variation_profile --headless --variant static --num_envs 128
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def add_profile_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the profiler's own arguments to Arena's shared parser."""
    group = parser.add_argument_group("Live variation pool profiler")
    group.add_argument(
        "--variant",
        choices=("pool", "static"),
        default="pool",
        help="Whether the pool owns placement ('pool') or the scene keeps its authored poses ('static').",
    )
    group.add_argument(
        "--pool_dir",
        type=Path,
        default=None,
        help="Profile against an existing pool instead of a synthetic one. Ignores --num_prims.",
    )
    group.add_argument("--num_prims", type=int, default=30, help="Variable props per environment.")
    group.add_argument(
        "--embodiment",
        default="franka_joint_pos",
        help="Embodiment to place in the scene, so step timing carries a realistic robot. '' for none.",
    )
    group.add_argument("--num_variations", type=int, default=4096, help="Variations packed into the profiling pool.")
    group.add_argument("--num_resets", type=int, default=50, help="Timed full-environment resets.")
    group.add_argument("--num_steps", type=int, default=200, help="Timed steps, taken without resetting.")
    group.add_argument("--warmup_steps", type=int, default=20, help="Untimed steps run before the timed ones.")
    group.add_argument("--out_json", type=Path, default=None, help="Write the measurements to this file as JSON.")


def _percentile(samples: list[float], fraction: float) -> float:
    """Return the sample at ``fraction`` through a sorted copy of ``samples``."""
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _summarise(samples: list[float]) -> dict[str, float]:
    """Return median, p99, mean, and extremes of a millisecond sample set."""
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p99_ms": _percentile(samples, 0.99),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


PROP_SPACING_M = 0.12
"""Grid pitch of the profiled props, wider than their 5 cm extent so a layout has no interpenetration."""

PROP_JITTER_M = 0.03
"""Half-width of the per-variation offset, kept under half the pitch so jitter cannot cause overlap."""

PROP_HEIGHT_M = 0.4
"""Height the props are placed at, clear of the robot's base."""


def _prop_grid(num_prims: int) -> np.ndarray:
    """Return the ``(num_prims, 3)`` nominal position of each prop.

    A compact grid in front of the robot, which is what keeps the physics broadphase tight. Scattering
    props over hundreds of metres would make PhysX, not the pool, dominate every measurement.

    Args:
        num_prims: Props to lay out.
    """
    columns = max(1, math.ceil(math.sqrt(num_prims)))
    positions = np.zeros((num_prims, 3), dtype=np.float32)
    for index in range(num_prims):
        row, column = divmod(index, columns)
        positions[index] = (
            0.4 + (row - (num_prims // columns) / 2.0) * PROP_SPACING_M,
            (column - (columns - 1) / 2.0) * PROP_SPACING_M,
            PROP_HEIGHT_M,
        )
    return positions


def _profiling_poses(num_prims: int, num_variations: int) -> np.ndarray:
    """Return ``(num_variations, num_prims, 7)`` poses that jitter the grid without overlapping.

    Every variation perturbs the same nominal layout, so the pooled and static variants present
    PhysX with the same collision workload and the measured delta is the pool's own cost.

    Args:
        num_prims: Props in the inventory.
        num_variations: Variations to generate.
    """
    from isaaclab_arena.live_scene.pool_contract import POSE_COLUMNS

    generator = np.random.default_rng(0)
    grid = _prop_grid(num_prims)
    poses = np.zeros((num_variations, num_prims, POSE_COLUMNS), dtype=np.float32)
    poses[:, :, :3] = grid[None, :, :]
    poses[:, :, :3] += generator.uniform(-PROP_JITTER_M, PROP_JITTER_M, size=(num_variations, num_prims, 3)).astype(
        np.float32
    )
    # Yaw only, so a prop never starts resting on a corner. Arena's order is (x, y, z, w), so the
    # quaternion occupies columns 3 through 6.
    yaw = generator.uniform(-np.pi, np.pi, size=(num_variations, num_prims)).astype(np.float32)
    poses[:, :, 5] = np.sin(yaw / 2.0)
    poses[:, :, 6] = np.cos(yaw / 2.0)
    return poses


def build_profiling_scene(usd_path: Path, positions: np.ndarray) -> None:
    """Author a base scene whose props sit at their nominal grid positions.

    The static variant replays exactly these poses every episode, which is what makes it the right
    baseline: same assets, same layout scale, no per-reset write.

    Args:
        usd_path: Destination USD file.
        positions: ``(num_prims, 3)`` nominal positions.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(usd_path.as_posix())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    for index, position in enumerate(positions):
        body = UsdGeom.Xform.Define(stage, f"/Root/Props/prop_{index:03d}")
        body.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in position)))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.2)
        mesh = UsdGeom.Cube.Define(stage, f"/Root/Props/prop_{index:03d}/geometry")
        mesh.CreateSizeAttr(0.05)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    assert stage.GetRootLayer().Save(), f"failed to write profiling base scene to {usd_path}"


def build_profiling_pool(pool_dir: Path, num_prims: int, num_variations: int) -> tuple[Any, float, float]:
    """Write a pool whose scene and shards are shaped like a generator's output.

    Args:
        pool_dir: Directory to create the pool in.
        num_prims: Variable props in the inventory.
        num_variations: Variations packed into the pool's single shard.

    Returns:
        A ``(manifest, scene_authoring_s, shard_write_s)`` tuple.
    """
    from isaaclab_arena.live_scene import shard_writer
    from isaaclab_arena.tests.utils import variation_pool_fixtures as fixtures

    prim_paths = tuple(f"Props/prop_{index:03d}" for index in range(num_prims))
    manifest = fixtures.build_manifest(prim_paths=prim_paths, variations_per_shard=num_variations)

    scene_start = time.perf_counter()
    build_profiling_scene(pool_dir / manifest.scene_usd, _prop_grid(num_prims))
    scene_authoring_s = time.perf_counter() - scene_start

    shard_writer.write_manifest(pool_dir, manifest)
    poses = _profiling_poses(num_prims, num_variations)
    shard_start = time.perf_counter()
    shard_writer.write_shard(pool_dir, sequence=0, poses=poses)
    shard_write_s = time.perf_counter() - shard_start

    return manifest, scene_authoring_s, shard_write_s


def build_env(pool_dir: Path, manifest: Any, args_cli: argparse.Namespace) -> tuple[ManagerBasedEnv, float]:
    """Build the profiled environment and return it with its build time in seconds.

    Both variants get the same scene, the same prim count, and the same physics load. Only the
    ``live_variation_pool_dir`` differs, so the delta between them is attributable to the pool.

    Args:
        pool_dir: Directory holding the profiling pool.
        manifest: Manifest of that pool.
        args_cli: Parsed CLI arguments.

    Returns:
        An ``(env, build_s)`` tuple. The caller owns closing the env.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.live_scene.generated_background import (
        GENERATED_SCENE_BACKGROUND_REGISTRY_NAME,
        build_object_references,
    )
    from isaaclab_arena.live_scene.pool_manifest import resolve_scene_usd
    from isaaclab_arena.scene.scene import Scene

    registry = AssetRegistry()
    background_cls = registry.get_asset_by_name(GENERATED_SCENE_BACKGROUND_REGISTRY_NAME)
    background = background_cls(usd_path=str(resolve_scene_usd(pool_dir, manifest)))
    references = build_object_references(manifest, background)
    scene = Scene(assets=[background, *references])

    embodiment = None
    if args_cli.embodiment:
        embodiment = registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=False, concatenate_observation_terms=True
        )

    arena_env = IsaacLabArenaEnvironment(
        name=f"live_variation_profile_{args_cli.variant}",
        embodiment=embodiment,
        scene=scene,
        env_cfg_callback=_configure_like_training,
    )

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    # Neither variant solves relations: the pool disables it anyway, and leaving it on for the static
    # variant would charge that variant for placement work the comparison is not about.
    builder_cfg.solve_relations = False
    builder_cfg.live_variation_pool_dir = str(pool_dir) if args_cli.variant == "pool" else None

    start = time.perf_counter()
    env = ArenaEnvBuilder(arena_env, builder_cfg).make_registered()
    return env, time.perf_counter() - start


def _configure_like_training(env_cfg: Any) -> Any:
    """Match the control rate and physics buffers the live lift task trains with.

    Profiling at Arena's default control rate would understate step cost, and the default contact
    buffers overflow once every environment holds a table's worth of props.

    Args:
        env_cfg: The environment configuration to modify in place.

    Returns:
        The same configuration, so this works directly as an ``env_cfg_callback``.
    """
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz
    from isaaclab_arena_environments.lift_object_live_environment import _size_physics_buffers

    set_control_rate_50hz(env_cfg)
    _size_physics_buffers(env_cfg)
    return env_cfg


def _zero_action(env: ManagerBasedEnv) -> Any:
    """Return a zero action sized for the environment, including the no-embodiment case."""
    import torch

    action_dim = int(getattr(env.unwrapped, "action_manager").total_action_dim)
    return torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)


def measure_reset_latency(env: ManagerBasedEnv, num_resets: int) -> list[float]:
    """Return per-reset wall time in milliseconds.

    Every environment resets on each call, so this is the worst case: the reset path's cost scales
    with how many environments are resetting, and a training run rarely resets all of them at once.

    Args:
        env: The environment to reset.
        num_resets: Timed resets to perform.
    """
    import torch

    samples: list[float] = []
    on_cuda = env.unwrapped.device.startswith("cuda")
    with torch.inference_mode():
        env.reset()
        if on_cuda:
            torch.cuda.synchronize()
        for _ in range(num_resets):
            start = time.perf_counter()
            env.reset()
            if on_cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def measure_step_time(env: ManagerBasedEnv, num_steps: int, warmup_steps: int) -> list[float]:
    """Return per-step wall time in milliseconds, measured without resetting.

    The pool contributes nothing during a step, so this is the control: the two variants should be
    indistinguishable here, and a gap would mean the pool leaked cost into the hot loop.

    Args:
        env: The environment to step.
        num_steps: Timed steps.
        warmup_steps: Untimed steps run first, covering lazy kernel loads and allocator growth.
    """
    import torch

    action = _zero_action(env)
    samples: list[float] = []
    on_cuda = env.unwrapped.device.startswith("cuda")
    with torch.inference_mode():
        env.reset()
        for _ in range(warmup_steps):
            env.step(action)
        if on_cuda:
            torch.cuda.synchronize()
        for _ in range(num_steps):
            start = time.perf_counter()
            env.step(action)
            if on_cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def measure_variation_event(env: ManagerBasedEnv, num_draws: int) -> dict[str, float] | None:
    """Return the isolated cost of the reset event's variation write, or ``None`` without a pool.

    Separates the pool's own contribution from everything else a reset does, which is what makes the
    reset-latency delta interpretable.

    Args:
        env: The environment holding the pool.
        num_draws: Timed draw-and-write iterations.
    """
    import torch

    from isaaclab_arena.live_scene.variation_events import LIVE_VARIATION_RESET_EVENT_NAME, apply_scene_variation

    try:
        term_cfg = env.unwrapped.event_manager.get_term_cfg(LIVE_VARIATION_RESET_EVENT_NAME)
    except ValueError:
        return None

    base_env = env.unwrapped
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    on_cuda = base_env.device.startswith("cuda")
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(10):
            apply_scene_variation(base_env, env_ids, **term_cfg.params)
        if on_cuda:
            torch.cuda.synchronize()
        for _ in range(num_draws):
            start = time.perf_counter()
            apply_scene_variation(base_env, env_ids, **term_cfg.params)
            if on_cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return _summarise(samples)


def measure_variation_scaling(env: ManagerBasedEnv, num_draws: int) -> dict[str, Any] | None:
    """Return the variation write cost against the number of environments resetting together.

    Decides how the cost amortises over training. The write is one batched call per inventory prim
    covering every resetting environment, so if the cost barely moves with the batch size then it is
    paid per reset *call*, and what matters is how often environments reset together rather than how
    many of them there are.

    Args:
        env: The environment holding the pool.
        num_draws: Timed iterations per batch size.
    """
    import torch

    from isaaclab_arena.live_scene.variation_events import LIVE_VARIATION_RESET_EVENT_NAME, apply_scene_variation

    base_env = env.unwrapped
    try:
        term_cfg = base_env.event_manager.get_term_cfg(LIVE_VARIATION_RESET_EVENT_NAME)
    except ValueError:
        return None

    on_cuda = base_env.device.startswith("cuda")
    batch_sizes = [size for size in (1, 2, 8, 32, 64, 128, 256) if size <= base_env.num_envs]
    if base_env.num_envs not in batch_sizes:
        batch_sizes.append(base_env.num_envs)

    results: dict[str, Any] = {}
    with torch.inference_mode():
        for size in batch_sizes:
            env_ids = torch.arange(size, device=base_env.device)
            for _ in range(5):
                apply_scene_variation(base_env, env_ids, **term_cfg.params)
            if on_cuda:
                torch.cuda.synchronize()
            samples: list[float] = []
            for _ in range(num_draws):
                start = time.perf_counter()
                apply_scene_variation(base_env, env_ids, **term_cfg.params)
                if on_cuda:
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
            results[str(size)] = _summarise(samples)
    return results


def measure_paired_reset(env: ManagerBasedEnv, num_pairs: int) -> dict[str, Any] | None:
    """Return reset latency with the variation event alternately enabled and disabled.

    Comparing two separate processes cannot separate the pool's cost from whatever else the machine
    was doing at the time; on a loaded host that noise is larger than the effect. Alternating the
    event within one process puts both arms on the same stage, the same allocator, and the same
    interleaving of other load, so their difference is attributable to the pool.

    Returns ``None`` when no variation event is installed.

    Args:
        env: The environment holding the pool.
        num_pairs: On/off reset pairs to time.
    """
    import torch

    from isaaclab_arena.live_scene.variation_events import LIVE_VARIATION_RESET_EVENT_NAME

    event_manager = env.unwrapped.event_manager
    try:
        term_cfg = event_manager.get_term_cfg(LIVE_VARIATION_RESET_EVENT_NAME)
    except ValueError:
        return None

    variation_func = term_cfg.func

    def disabled(*args: Any, **kwargs: Any) -> None:
        """Stand in for the variation write, leaving every other reset cost in place."""

    on_samples: list[float] = []
    off_samples: list[float] = []
    on_cuda = env.unwrapped.device.startswith("cuda")
    with torch.inference_mode():
        env.reset()
        if on_cuda:
            torch.cuda.synchronize()
        for index in range(num_pairs):
            # Alternate which arm goes first so a slow drift in machine load cannot land
            # systematically on one of them.
            order = (variation_func, disabled) if index % 2 == 0 else (disabled, variation_func)
            for func in order:
                term_cfg.func = func
                start = time.perf_counter()
                env.reset()
                if on_cuda:
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000.0
                (on_samples if func is variation_func else off_samples).append(elapsed)
    term_cfg.func = variation_func

    on_summary, off_summary = _summarise(on_samples), _summarise(off_samples)
    return {
        "with_variation": on_summary,
        "without_variation": off_summary,
        "delta_median_ms": on_summary["median_ms"] - off_summary["median_ms"],
    }


def _profile_pool(args_cli: argparse.Namespace, scratch_dir: Path) -> tuple[Path, Any, float, float]:
    """Return the pool to profile, building a synthetic one unless ``--pool_dir`` names a real one.

    Args:
        args_cli: Parsed CLI arguments.
        scratch_dir: Directory a synthetic pool is written into.

    Returns:
        A ``(pool_dir, manifest, scene_authoring_s, shard_write_s)`` tuple. The two timings are
        ``nan`` for a pre-existing pool, whose production cost this process did not pay.
    """
    from isaaclab_arena.live_scene.pool_manifest import load_pool_manifest

    if args_cli.pool_dir is not None:
        assert args_cli.pool_dir.is_dir(), f"pool directory not found: {args_cli.pool_dir}"
        return args_cli.pool_dir, load_pool_manifest(args_cli.pool_dir), float("nan"), float("nan")

    manifest, scene_authoring_s, shard_write_s = build_profiling_pool(
        scratch_dir, args_cli.num_prims, args_cli.num_variations
    )
    return scratch_dir, manifest, scene_authoring_s, shard_write_s


def run_profile(args_cli: argparse.Namespace) -> dict[str, Any]:
    """Build the profiled environment, take every measurement, and return them as a dict."""
    import tempfile

    from isaaclab_arena.live_scene.variation_events import get_live_variation_pool

    with tempfile.TemporaryDirectory(prefix="arena-sim-variation-profile-") as temp_dir:
        pool_dir, manifest, scene_authoring_s, shard_write_s = _profile_pool(args_cli, Path(temp_dir))
        scene_usd = pool_dir / manifest.scene_usd
        print(
            f"[profile] pool {pool_dir}: {manifest.num_prims} prims, scene {scene_usd.stat().st_size / 2**20:.1f} MiB",
            flush=True,
        )

        env, build_s = build_env(pool_dir, manifest, args_cli)
        print(f"[profile] variant={args_cli.variant} built in {build_s:.2f}s", flush=True)
        try:
            measurements: dict[str, Any] = {
                "variant": args_cli.variant,
                "pool_dir": str(pool_dir),
                "num_envs": env.unwrapped.num_envs,
                "num_prims": manifest.num_prims,
                "scene_usd_bytes": scene_usd.stat().st_size,
                "device": env.unwrapped.device,
                "step_dt_ms": float(env.unwrapped.step_dt) * 1000.0,
                "scene_authoring_s": scene_authoring_s,
                "shard_write_s": shard_write_s,
                "env_build_s": build_s,
                "step": _summarise(measure_step_time(env, args_cli.num_steps, args_cli.warmup_steps)),
                "reset": _summarise(measure_reset_latency(env, args_cli.num_resets)),
                "variation_event": measure_variation_event(env, args_cli.num_resets),
                "variation_scaling": measure_variation_scaling(env, args_cli.num_resets),
                "paired_reset": measure_paired_reset(env, args_cli.num_resets),
            }
            pool = get_live_variation_pool(env)
            measurements["pool_stats"] = pool.stats.as_dict() if pool is not None else None
        finally:
            env.close()

    return measurements


def report(measurements: dict[str, Any], out_json: Path | None) -> None:
    """Print the measurements as JSON and optionally write them to a file.

    Called before the simulation app closes, because Kit's shutdown can end the process without
    running anything queued after it.

    Args:
        measurements: The measurements to emit.
        out_json: File to write to, or ``None`` to only print.
    """
    payload = json.dumps(measurements, indent=2)
    print(f"=== PROFILE RESULT ===\n{payload}\n=== END PROFILE RESULT ===", flush=True)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(payload)
        print(f"[profile] wrote {out_json}", flush=True)


def main() -> int:
    """Parse arguments, run the profile inside Isaac Sim, and report the measurements."""
    parser = get_isaaclab_arena_cli_parser()
    add_profile_cli_args(parser)
    args_cli = parser.parse_args()

    with SimulationAppContext(args_cli):
        report(run_profile(args_cli), args_cli.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
