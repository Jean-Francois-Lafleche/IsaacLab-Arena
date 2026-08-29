# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pool of pre-generated scene variations, ingested off-thread and held on device.

Performance shape, in the order it matters:

* **Ingest touches only files and numpy.** A daemon thread discovers shards, copies rows out of the
  memory-mapped files, and stages them in pinned host memory. Both the filesystem and numpy release
  the GIL, so this genuinely overlaps the simulation thread.
* **The pool lives on device.** Variations occupy a preallocated ring buffer, so drawing one at reset
  is an index into an existing tensor rather than a host-to-device copy.
* **Draws never block.** Environments consume a single ordered stream of variations, so the
  environments in one reset batch receive different variations and a later reset receives fresh ones.
  When the stream runs dry the pool cycles through resident variations and counts a starvation
  instead of waiting on the producer.

The staged pinned buffers are copied into the ring by :meth:`LiveVariationPool.drain_pending`, which
runs on the simulation thread. Only that thread touches CUDA, which keeps the design clear of
multi-threaded CUDA context questions, and the copy is small and asynchronous.
"""

from __future__ import annotations

import logging
import numpy as np
import queue
import threading
import time
import torch
from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.live_scene import shard_reader
from isaaclab_arena.live_scene.pool_manifest import PoolManifest, load_pool_manifest

_logger = logging.getLogger(__name__)

DEFAULT_MIN_VARIATIONS_PER_ENV = 8
"""Ring capacity per environment. Larger means longer before an environment repeats a variation."""

DEFAULT_POLL_INTERVAL_S = 0.25
"""How often the ingest thread looks for new shards."""


@dataclass
class PoolStats:
    """Counters describing how well the producer is keeping up."""

    ingested_variations: int = 0
    """Variations copied into the ring since construction."""

    ingested_shards: int = 0
    """Shards read since construction."""

    vanished_shards: int = 0
    """Shards discovered but pruned by a producer before they could be read."""

    draws: int = 0
    """Variations handed out to environments."""

    starved_draws: int = 0
    """Draws served by reusing a resident variation because the stream had run dry."""

    skipped_variations: int = 0
    """Variations no environment saw because ingestion outran consumption and the ring wrapped."""

    def as_dict(self) -> dict[str, int]:
        """Return the counters as a plain dict for logging."""
        return {
            "ingested_variations": self.ingested_variations,
            "ingested_shards": self.ingested_shards,
            "vanished_shards": self.vanished_shards,
            "draws": self.draws,
            "starved_draws": self.starved_draws,
            "skipped_variations": self.skipped_variations,
        }


class LiveVariationPool:
    """Ring buffer of scene variations fed from a producer's pool directory.

    Args:
        pool_dir: Directory the producer writes into.
        num_envs: Number of simulation environments drawing from the pool.
        device: Device the ring buffer lives on, e.g. ``"cuda:0"``.
        manifest: Pre-loaded manifest. Read from ``pool_dir`` when omitted.
        min_variations_per_env: Ring capacity per environment.
        poll_interval_s: How often the ingest thread looks for new shards.
        watch: When ``False``, ingest whatever exists at construction and never start a thread.
            Useful for a pre-baked pool and for tests.
    """

    def __init__(
        self,
        pool_dir: str | Path,
        num_envs: int,
        device: str | torch.device = "cpu",
        manifest: PoolManifest | None = None,
        min_variations_per_env: int = DEFAULT_MIN_VARIATIONS_PER_ENV,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        watch: bool = True,
    ) -> None:
        assert num_envs >= 1, f"num_envs must be >= 1, got {num_envs}"
        assert min_variations_per_env >= 1, f"min_variations_per_env must be >= 1, got {min_variations_per_env}"

        self._pool_dir = Path(pool_dir)
        self._manifest = manifest if manifest is not None else load_pool_manifest(self._pool_dir)
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._stats = PoolStats()

        capacity = max(num_envs * min_variations_per_env, self._manifest.shard.variations_per_shard)
        self._capacity = capacity
        self._ring = torch.zeros(
            (capacity, self._manifest.num_prims, self._manifest.shard.pose_columns),
            dtype=torch.float32,
            device=self._device,
        )
        # A second ring in lockstep with the poses, so one draw yields a variation's poses and its
        # scalars from the same slot. Width zero when the pool declares none, which keeps the reset
        # path free of a branch.
        self._scalar_names = tuple(self._manifest.shard.scalars)
        self._scalar_ring = torch.zeros(
            (capacity, len(self._scalar_names)),
            dtype=torch.float32,
            device=self._device,
        )
        self._parking_offset = torch.tensor(self._manifest.parking_offset_xyz, dtype=torch.float32, device=self._device)

        # Total variations ever written into the ring. Slot i holds global index i % capacity, so the
        # resident window is [max(0, _filled - capacity), _filled).
        self._filled = 0
        # Environments share one ordered stream, which is what makes parallel environments differ:
        # each draw in a reset batch takes the next variation rather than all reading one cursor.
        self._consume_cursor = 0
        self._reuse_rotation = 0
        self._last_variation_per_env = [-1] * num_envs

        self._pending: queue.Queue[tuple[int, torch.Tensor, torch.Tensor]] = queue.Queue()
        self._last_sequence = -1
        self._sequence_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._poll_interval_s = poll_interval_s

        # Bootstrap synchronously so the first reset has something to draw even with watching off.
        self._ingest_available_shards()
        self.drain_pending()

        if watch:
            self._thread = threading.Thread(
                target=self._watch_loop,
                name="arena-variation-ingest",
                daemon=True,
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pool_dir(self) -> Path:
        """Directory the variations are read from."""
        return self._pool_dir

    @property
    def manifest(self) -> PoolManifest:
        """Manifest of the pool being consumed."""
        return self._manifest

    @property
    def num_envs(self) -> int:
        """Number of environments this pool was sized for."""
        return self._num_envs

    @property
    def capacity(self) -> int:
        """Variations resident in the ring buffer."""
        return self._capacity

    @property
    def resident(self) -> int:
        """Variations currently resident, which is the ring capacity once it has wrapped."""
        return min(self._filled, self._capacity)

    @property
    def stats(self) -> PoolStats:
        """Producer/consumer counters."""
        return self._stats

    @property
    def parking_offset(self) -> torch.Tensor:
        """Env-local parking offset as a ``(3,)`` tensor on the pool's device."""
        return self._parking_offset

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _watch_loop(self) -> None:
        """Poll the pool directory for new shards until stopped."""
        while not self._stop.is_set():
            try:
                self._ingest_available_shards()
            except Exception:
                # A half-written or malformed shard must not take the training run down; the pool
                # keeps serving resident variations and the next poll retries.
                _logger.exception("variation ingest failed; continuing with resident variations")
            self._stop.wait(self._poll_interval_s)

    def _ingest_available_shards(self) -> None:
        """Stage every shard newer than the last ingested sequence into pinned host memory."""
        with self._sequence_lock:
            after = self._last_sequence
        paths = shard_reader.discover_shard_paths(self._pool_dir, after_sequence=after)
        for path in paths:
            try:
                shard = shard_reader.load_shard(path, self._manifest)
            except FileNotFoundError:
                # Discovered, then pruned by a producer before it could be opened. Expected of a pool
                # that keeps a bounded window of shards, and no reason to abandon the shards behind it
                # in this batch: they are the newer ones, and the ring wants them.
                self._stats.vanished_shards += 1
                continue
            # Copying out of the mmap here is the expensive part, and it is why ingestion runs off
            # the simulation thread.
            staged = torch.from_numpy(np.array(shard.poses, dtype=np.float32))
            staged_scalars = torch.from_numpy(np.array(shard.scalars, dtype=np.float32))
            if self._device.type == "cuda":
                staged = staged.pin_memory()
                staged_scalars = staged_scalars.pin_memory()
            self._pending.put((shard.sequence, staged, staged_scalars))
            with self._sequence_lock:
                self._last_sequence = max(self._last_sequence, shard.sequence)
            self._stats.ingested_shards += 1

    def poll_for_new_shards(self) -> int:
        """Ingest any shards that have appeared and copy them into the ring.

        The ingest thread does this on its own. Call it directly when the pool was constructed with
        ``watch=False``, which is how a pre-baked pool and the tests drive ingestion.

        Returns:
            Number of variations added to the ring.
        """
        self._ingest_available_shards()
        return self.drain_pending()

    def drain_pending(self) -> int:
        """Copy staged shards into the ring buffer.

        Called from the simulation thread, so it is the only place that touches the device. The copy
        is asynchronous when the staging buffer is pinned.

        Returns:
            Number of variations copied into the ring.
        """
        copied = 0
        while True:
            try:
                _, staged, staged_scalars = self._pending.get_nowait()
            except queue.Empty:
                break
            copied += self._write_rows(staged, staged_scalars)
        return copied

    def _write_rows(self, rows: torch.Tensor, scalar_rows: torch.Tensor) -> int:
        """Write staged variation rows into consecutive ring slots.

        Args:
            rows: ``(n, n_prims, 7)`` host tensor.
            scalar_rows: ``(n, n_scalars)`` host tensor, aligned with ``rows``.

        Returns:
            Number of rows written.
        """
        total = int(rows.shape[0])
        if total == 0:
            return 0
        assert (
            int(scalar_rows.shape[0]) == total
        ), f"staged scalars have {int(scalar_rows.shape[0])} rows but poses have {total}"
        # A shard larger than the ring can only leave its newest rows resident.
        if total > self._capacity:
            rows = rows[-self._capacity :]
            scalar_rows = scalar_rows[-self._capacity :]
            total = self._capacity
        written = 0
        while written < total:
            slot = (self._filled + written) % self._capacity
            chunk = min(total - written, self._capacity - slot)
            self._ring[slot : slot + chunk].copy_(rows[written : written + chunk], non_blocking=True)
            self._scalar_ring[slot : slot + chunk].copy_(scalar_rows[written : written + chunk], non_blocking=True)
            written += chunk
        self._filled += total
        self._stats.ingested_variations += total
        return total

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------

    def wait_for_initial_fill(self, timeout_s: float = 30.0) -> None:
        """Block until at least one variation is resident.

        Runs at build time, never on the reset path, so blocking here is safe.

        Args:
            timeout_s: How long to wait before giving up.
        """
        deadline = time.monotonic() + timeout_s
        while self._filled == 0:
            self.drain_pending()
            if self._filled > 0:
                break
            assert time.monotonic() < deadline, (
                f"variation pool at {self._pool_dir} produced no variations within {timeout_s}s; "
                "is the producer running?"
            )
            time.sleep(min(self._poll_interval_s, 0.05))

    @property
    def last_variation_per_env(self) -> list[int]:
        """Global variation index each environment last drew, or ``-1`` before its first draw."""
        return list(self._last_variation_per_env)

    def draw_variation_indices(self, env_ids: list[int]) -> list[int]:
        """Return the global variation index for each environment, advancing the shared stream.

        Never blocks. Consecutive environments in one call receive consecutive variations, so
        parallel environments diverge. When the producer has not kept up, the pool cycles through the
        resident window instead of stalling; environments in the batch still differ as long as more
        variations are resident than are being reset.

        Args:
            env_ids: Absolute environment indices being reset.

        Returns:
            One global variation index per entry in ``env_ids``.
        """
        assert self._filled > 0, (
            f"variation pool at {self._pool_dir} has no resident variations; "
            "call wait_for_initial_fill() during the build"
        )
        oldest = max(0, self._filled - self._capacity)
        resident = self._filled - oldest
        indices: list[int] = []
        for env_id in env_ids:
            assert 0 <= env_id < self._num_envs, f"env id {env_id} outside pool sized for {self._num_envs} envs"
            if self._consume_cursor < oldest:
                # Ingestion lapped consumption, so the variations in between are gone.
                self._stats.skipped_variations += oldest - self._consume_cursor
                self._consume_cursor = oldest
            if self._consume_cursor >= self._filled:
                index = oldest + (self._reuse_rotation % resident)
                self._reuse_rotation += 1
                self._stats.starved_draws += 1
            else:
                index = self._consume_cursor
                self._consume_cursor += 1
            indices.append(index)
            self._last_variation_per_env[env_id] = index
            self._stats.draws += 1
        return indices

    def draw_slots_for_envs(self, env_ids: list[int]) -> torch.Tensor:
        """Return the ring slot each environment should read, advancing the shared stream.

        Args:
            env_ids: Absolute environment indices being reset.

        Returns:
            ``int64`` tensor of ring slots on the pool's device, aligned with ``env_ids``.
        """
        indices = self.draw_variation_indices(env_ids)
        return torch.tensor([index % self._capacity for index in indices], dtype=torch.long, device=self._device)

    def gather_poses(self, env_ids: list[int]) -> torch.Tensor:
        """Return one variation per environment, drained and drawn in one call.

        Args:
            env_ids: Absolute environment indices being reset.

        Returns:
            ``(len(env_ids), n_prims, 7)`` tensor of env-local poses on the pool's device.
        """
        return self.gather(env_ids)[0]

    def gather(self, env_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one variation's poses and scalars per environment, from a single draw.

        Use this rather than calling :meth:`gather_poses` and a separate scalar accessor: each draw
        advances the shared stream, so two calls would hand an environment two different variations.

        Args:
            env_ids: Absolute environment indices being reset.

        Returns:
            A ``(poses, scalars)`` pair of shapes ``(len(env_ids), n_prims, 7)`` and
            ``(len(env_ids), n_scalars)``, on the pool's device.
        """
        self.drain_pending()
        slots = self.draw_slots_for_envs(env_ids)
        return self._ring.index_select(0, slots), self._scalar_ring.index_select(0, slots)

    @property
    def scalar_names(self) -> tuple[str, ...]:
        """Per-variation scalars this pool publishes, in column order."""
        return self._scalar_names

    def scalar_column(self, name: str) -> int:
        """Return the column a named scalar occupies, to be resolved once and indexed thereafter.

        Args:
            name: Scalar name, which must be one the pool declares.
        """
        assert (
            name in self._scalar_names
        ), f"pool at {self._pool_dir} publishes scalars {list(self._scalar_names)}, not {name!r}"
        return self._scalar_names.index(name)

    def close(self) -> None:
        """Stop the ingest thread and release staged buffers."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        while True:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                break


class LiveVariationPoolHandle:
    """Opaque holder for a :class:`LiveVariationPool` used as an ``EventTermCfg`` param.

    Isaac Lab deep-copies and validates event term params. Copying the pool would duplicate the ring
    buffer and orphan the ingest thread, and validation would walk into the pool's internals. This
    mirrors :class:`~isaaclab_arena.relations.placement_events.PlacementPoolHandle`: ``__slots__``
    means ``hasattr(handle, "__dict__")`` is false, so validation stops here.
    """

    __slots__ = ("pool",)

    def __init__(self, pool: LiveVariationPool) -> None:
        self.pool = pool

    def __deepcopy__(self, memo: dict[int, object]) -> LiveVariationPoolHandle:
        """Share the live pool across ``copy.deepcopy`` so the ingest thread keeps feeding it."""
        memo[id(self)] = self
        return self

    def __getstate__(self) -> dict[str, str]:
        """Serialize as a description of the pool rather than the pool itself.

        A training script dumps the environment config to YAML for its logs, which walks into every
        term parameter. The pool owns an ingest thread, a lock, and device memory, so it is
        process-local and cannot be serialized; only its identity is worth recording.
        """
        return {"pool_dir": "" if self.pool is None else str(self.pool.pool_dir)}

    def __setstate__(self, state: dict[str, str]) -> None:
        """Restore without a pool, since the live one cannot cross a process boundary."""
        self.pool = None
