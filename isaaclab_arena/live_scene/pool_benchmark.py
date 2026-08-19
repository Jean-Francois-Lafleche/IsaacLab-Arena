# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure the variation pool's hot path, so the numbers gate the design rather than justify it.

Three questions, in the order they matter:

* **Draw latency** — what the reset path pays to pick one variation per environment out of the
  device ring buffer. This is the budget the regression test defends.
* **Ingest throughput** — variations per second a shard reader sustains, which has to exceed the
  consumption rate at the target environment count.
* **GIL overlap** — whether the ingest thread genuinely runs beside a busy simulation thread. The
  design rests on ingestion touching only the filesystem and numpy; this measures whether that
  claim holds by timing a synthetic compute loop with and without ingestion running.

Run it standalone, without Isaac Sim::

    python -m isaaclab_arena.live_scene.pool_benchmark --num-envs 64 --num-prims 200

Baseline on an RTX-class workstation at 64 environments and 200 variable prims: a 0.035 ms p99
draw, 770k variations per second ingested, and a GIL overlap ratio of 1.01. Ingestion outruns
consumption by orders of magnitude and the ingest thread costs the simulation thread nothing
measurable, which is what justifies the numpy-only shard format over reading USD at runtime.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import statistics
import tempfile
import threading
import time
import torch
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.live_scene import shard_reader, shard_writer
from isaaclab_arena.live_scene.pool_contract import POSE_COLUMNS, SCHEMA_VERSION
from isaaclab_arena.live_scene.pool_manifest import PoolManifest, PoolPrimSpec, ShardLayout
from isaaclab_arena.live_scene.variation_pool import LiveVariationPool

DRAW_LATENCY_BUDGET_MS = 2.0
"""Ceiling the regression test holds the draw path to at :data:`BUDGET_NUM_ENVS` environments and
:data:`BUDGET_NUM_PRIMS` prims. The measured baseline is around 0.035 ms, so this leaves room for a
loaded CI machine while still catching a change that reintroduces per-environment work."""

BUDGET_NUM_ENVS = 64
"""Environment count :data:`DRAW_LATENCY_BUDGET_MS` was set at."""

BUDGET_NUM_PRIMS = 200
"""Inventory length :data:`DRAW_LATENCY_BUDGET_MS` was set at."""

MIN_INGEST_VARIATIONS_PER_S = 5000.0
"""Floor the regression test holds ingestion to. The measured baseline is around 770k, and even a
1000-environment run resetting every step consumes far less than this."""

MIN_GIL_OVERLAP_RATIO = 0.75
"""Floor for compute throughput with ingestion running. The measured baseline is 1.01; dropping
below this would mean ingestion had started contending for the GIL, most likely because something
on the ingest path began calling into USD."""


@dataclass
class BenchmarkResult:
    """Measurements from one benchmark run."""

    num_envs: int
    num_prims: int
    variations_per_shard: int
    device: str

    draw_median_ms: float
    """Median cost of drawing and gathering one variation per environment."""

    draw_p99_ms: float
    """Worst-case draw cost, which is what a reset-latency budget has to cover."""

    ingest_variations_per_s: float
    """Sustained rate a shard reader turns files into staged host tensors."""

    gil_overlap_ratio: float
    """Compute throughput with ingestion running, divided by throughput without it. 1.0 means
    ingestion is free; well below 1.0 means the ingest thread is stealing the GIL."""

    def as_json(self) -> str:
        """Return the measurements as indented JSON."""
        return json.dumps(asdict(self), indent=2)


def build_synthetic_manifest(num_prims: int, variations_per_shard: int) -> PoolManifest:
    """Return a manifest with ``num_prims`` rigid prims.

    Args:
        num_prims: Inventory length, which is the shard's second axis.
        variations_per_shard: Rows the producer packs per shard.
    """
    return PoolManifest(
        schema_version=SCHEMA_VERSION,
        parking_offset_xyz=(0.0, 0.0, -25.0),
        shard=ShardLayout(variations_per_shard=variations_per_shard),
        prims=[
            PoolPrimSpec(path=f"Props/prop_{index:04d}", object_type=ObjectType.RIGID) for index in range(num_prims)
        ],
    )


def _write_shards(pool_dir: Path, manifest: PoolManifest, num_shards: int) -> None:
    """Write ``num_shards`` shards of random poses into a pool directory."""
    rows = manifest.shard.variations_per_shard
    generator = np.random.default_rng(0)
    for sequence in range(num_shards):
        poses = generator.random((rows, manifest.num_prims, POSE_COLUMNS), dtype=np.float32)
        shard_writer.write_shard(pool_dir, sequence=sequence, poses=poses)


def measure_draw_latency(
    pool: LiveVariationPool,
    num_envs: int,
    iterations: int = 500,
    warmup_iterations: int = 50,
) -> tuple[float, float]:
    """Return median and p99 milliseconds for one draw-and-gather across all environments.

    Args:
        pool: Pool to draw from.
        num_envs: Environments drawing each iteration.
        iterations: Draws to time.
        warmup_iterations: Untimed draws run first. The first gathers on a device pay for lazy
            kernel loading and allocator growth, which is startup cost rather than reset cost.

    Returns:
        A ``(median_ms, p99_ms)`` tuple.
    """
    env_ids = list(range(num_envs))
    on_cuda = pool.parking_offset.device.type == "cuda"

    for _ in range(warmup_iterations):
        pool.gather_poses(env_ids)
    if on_cuda:
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        pool.gather_poses(env_ids)
        if on_cuda:
            # Without this the gather's cost would be hidden behind an unsynchronised stream.
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return statistics.median(samples), samples[min(len(samples) - 1, int(0.99 * len(samples)))]


def measure_ingest_throughput(pool_dir: Path, manifest: PoolManifest) -> float:
    """Return variations per second a shard reader sustains over every shard in a pool.

    Args:
        pool_dir: Pool directory holding the shards.
        manifest: Manifest of that pool.
    """
    paths = shard_reader.discover_shard_paths(pool_dir)
    assert paths, f"no shards to read in {pool_dir}"
    start = time.perf_counter()
    variations = 0
    for path in paths:
        shard = shard_reader.load_shard(path, manifest)
        # Forcing the copy is what a real ingest does; mapping alone would measure nothing.
        variations += int(np.array(shard.poses, dtype=np.float32).shape[0])
    elapsed = time.perf_counter() - start
    return variations / elapsed if elapsed > 0 else float("inf")


def _spin_compute(duration_s: float) -> int:
    """Run a numpy-light Python loop for ``duration_s`` and return iterations completed.

    Deliberately Python bytecode rather than a numpy call, because only bytecode competes for the
    GIL. A numpy-bound loop would appear unaffected no matter what the ingest thread does.
    """
    deadline = time.perf_counter() + duration_s
    iterations = 0
    accumulator = 0
    while time.perf_counter() < deadline:
        for value in range(1000):
            accumulator += value
        iterations += 1
    return iterations


def measure_gil_overlap(pool_dir: Path, manifest: PoolManifest, duration_s: float = 1.0) -> float:
    """Return the compute throughput ratio with ingestion running versus idle.

    Args:
        pool_dir: Pool directory holding the shards.
        manifest: Manifest of that pool.
        duration_s: How long each of the two measurement windows runs.

    Returns:
        ``busy_iterations / baseline_iterations``. Values near 1.0 confirm the ingest thread
        overlaps rather than contends.
    """
    baseline = _spin_compute(duration_s)

    stop = threading.Event()

    def ingest_forever() -> None:
        paths = shard_reader.discover_shard_paths(pool_dir)
        while not stop.is_set():
            for path in paths:
                if stop.is_set():
                    return
                shard = shard_reader.load_shard(path, manifest)
                np.array(shard.poses, dtype=np.float32)

    thread = threading.Thread(target=ingest_forever, name="benchmark-ingest", daemon=True)
    thread.start()
    try:
        busy = _spin_compute(duration_s)
    finally:
        stop.set()
        thread.join(timeout=5.0)
    return busy / baseline if baseline else 0.0


def run_benchmark(
    num_envs: int = 64,
    num_prims: int = 200,
    variations_per_shard: int = 256,
    num_shards: int = 8,
    device: str | None = None,
    measure_overlap: bool = True,
) -> BenchmarkResult:
    """Build a synthetic pool and measure the hot path against it.

    Args:
        num_envs: Environments drawing per reset.
        num_prims: Variable prims in the inventory.
        variations_per_shard: Rows per shard.
        num_shards: Shards to write and read.
        device: Device for the ring buffer. Defaults to CUDA when available.
        measure_overlap: Run the GIL overlap measurement, which costs two seconds.

    Returns:
        The measurements.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    manifest = build_synthetic_manifest(num_prims, variations_per_shard)

    with tempfile.TemporaryDirectory(prefix="arena-pool-benchmark-") as temp_dir:
        pool_dir = Path(temp_dir)
        shard_writer.write_manifest(pool_dir, manifest)
        _write_shards(pool_dir, manifest, num_shards)

        ingest_rate = measure_ingest_throughput(pool_dir, manifest)
        overlap = measure_gil_overlap(pool_dir, manifest) if measure_overlap else float("nan")

        pool = LiveVariationPool(pool_dir, num_envs=num_envs, device=device, manifest=manifest, watch=False)
        try:
            pool.wait_for_initial_fill()
            median_ms, p99_ms = measure_draw_latency(pool, num_envs)
        finally:
            pool.close()

    return BenchmarkResult(
        num_envs=num_envs,
        num_prims=num_prims,
        variations_per_shard=variations_per_shard,
        device=device,
        draw_median_ms=median_ms,
        draw_p99_ms=p99_ms,
        ingest_variations_per_s=ingest_rate,
        gil_overlap_ratio=overlap,
    )


def main() -> None:
    """Parse arguments, run the benchmark, and print the measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64, help="Environments drawing per reset.")
    parser.add_argument("--num-prims", type=int, default=200, help="Variable prims in the inventory.")
    parser.add_argument("--variations-per-shard", type=int, default=256, help="Rows per shard.")
    parser.add_argument("--num-shards", type=int, default=8, help="Shards to write and read.")
    parser.add_argument("--device", default=None, help="Ring buffer device. Defaults to CUDA when available.")
    parser.add_argument("--skip-overlap", action="store_true", help="Skip the two-second GIL overlap measurement.")
    args = parser.parse_args()

    result = run_benchmark(
        num_envs=args.num_envs,
        num_prims=args.num_prims,
        variations_per_shard=args.variations_per_shard,
        num_shards=args.num_shards,
        device=args.device,
        measure_overlap=not args.skip_overlap,
    )
    print(result.as_json())


if __name__ == "__main__":
    main()
