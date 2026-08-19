# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the live scene variation pool contract, reader, and consumer.

Nothing here starts Isaac Sim or imports ``pxr``: the pool's whole point is that its hot path is
files and numpy, and these tests hold it to that.
"""

from __future__ import annotations

import numpy as np
import time
from pathlib import Path

import pytest

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.live_scene import pool_contract, shard_reader, shard_writer
from isaaclab_arena.live_scene.pool_manifest import PoolManifest, PoolPrimSpec, ShardLayout, load_pool_manifest
from isaaclab_arena.live_scene.scene_entities import build_variation_write_plan, object_reference_name
from isaaclab_arena.live_scene.variation_pool import LiveVariationPool
from isaaclab_arena.tests.utils import variation_pool_fixtures as fixtures

# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_manifest_round_trips_through_the_pool_directory(tmp_path: Path):
    """A written manifest reloads with its inventory order and parking offset intact."""
    written = fixtures.build_manifest()
    shard_writer.write_manifest(tmp_path, written)

    loaded = load_pool_manifest(tmp_path)

    assert [prim.path for prim in loaded.prims] == list(fixtures.FIXTURE_PRIM_PATHS)
    assert loaded.parking_offset_xyz == fixtures.FIXTURE_PARKING_OFFSET
    assert loaded.num_prims == len(fixtures.FIXTURE_PRIM_PATHS)


def test_manifest_rejects_an_unsupported_schema_version():
    """A pool written against a different contract version must fail loudly, not be guessed at."""
    with pytest.raises(Exception, match="not supported by this Arena build"):
        PoolManifest(
            schema_version=pool_contract.SCHEMA_VERSION + 1,
            parking_offset_xyz=(0.0, 0.0, -25.0),
            shard=ShardLayout(variations_per_shard=2),
            prims=[PoolPrimSpec(path="Props/mug", object_type=ObjectType.RIGID)],
        )


def test_manifest_rejects_a_stale_inventory_digest(tmp_path: Path):
    """An inventory edited after the digest was recorded invalidates every existing shard."""
    manifest = fixtures.build_manifest()
    manifest.inventory_digest = "0" * 64
    shard_writer.write_manifest(tmp_path, manifest)

    with pytest.raises(AssertionError, match="inventory digest mismatch"):
        load_pool_manifest(tmp_path)


def test_manifest_rejects_a_foreign_quaternion_order():
    """Arena writes shard rows straight to the simulator, so a mirrored convention must not load."""
    with pytest.raises(Exception, match="quaternion_order must be"):
        ShardLayout(variations_per_shard=2, quaternion_order="wxyz")


def test_manifest_rejects_duplicate_prim_paths():
    """Duplicate paths would make the shard column order ambiguous."""
    with pytest.raises(Exception, match="duplicate prim path"):
        PoolManifest(
            schema_version=pool_contract.SCHEMA_VERSION,
            parking_offset_xyz=(0.0, 0.0, -25.0),
            shard=ShardLayout(variations_per_shard=2),
            prims=[
                PoolPrimSpec(path="Props/mug", object_type=ObjectType.RIGID),
                PoolPrimSpec(path="Props/mug", object_type=ObjectType.RIGID),
            ],
        )


def test_manifest_rejects_an_absolute_prim_path():
    """Inventory paths are relative to the base scene's default prim."""
    with pytest.raises(Exception, match="must be relative to the scene"):
        PoolPrimSpec(path="/Root/Props/mug", object_type=ObjectType.RIGID)


# ---------------------------------------------------------------------------
# Shard reading
# ---------------------------------------------------------------------------


def test_shard_reader_rejects_a_prim_count_mismatch(tmp_path: Path):
    """A shard packed against a different inventory must not be read positionally."""
    manifest = fixtures.build_manifest()
    shard_writer.write_manifest(tmp_path, manifest)
    wrong_width = np.zeros((2, manifest.num_prims + 1, pool_contract.POSE_COLUMNS), dtype=np.float32)
    poses_path = shard_writer.write_shard(tmp_path, sequence=0, poses=wrong_width)

    with pytest.raises(AssertionError, match="needs a new pool directory"):
        shard_reader.load_shard(poses_path, manifest)


def test_shard_reader_requires_the_parked_mask(tmp_path: Path):
    """The poses file is the readiness marker, so it may not outlive its companion."""
    manifest = fixtures.build_fixture_pool(tmp_path, num_variations=2)
    poses_path = pool_contract.shard_poses_path(tmp_path, 0)
    pool_contract.shard_parked_path(tmp_path, 0).unlink()

    with pytest.raises(AssertionError, match="has no parked mask"):
        shard_reader.load_shard(poses_path, manifest)


def test_shard_discovery_ignores_in_progress_writes(tmp_path: Path):
    """Atomic rename is the readiness protocol: a ``.tmp`` sibling must never be discovered."""
    fixtures.build_fixture_pool(tmp_path, num_variations=2)
    in_progress = (
        pool_contract.variations_dir(tmp_path) / f"00000009{pool_contract.POSES_SUFFIX}{pool_contract.TEMP_SUFFIX}"
    )
    in_progress.write_bytes(b"partial")

    discovered = shard_reader.discover_shard_paths(tmp_path)

    assert [path.name for path in discovered] == [f"00000000{pool_contract.POSES_SUFFIX}"]


def test_shard_discovery_orders_by_sequence_not_listing_order(tmp_path: Path):
    """Consumers rely on sequence order to replay variations as the producer emitted them."""
    manifest = fixtures.build_fixture_pool(tmp_path, num_variations=2)
    for sequence in (11, 2):
        poses, parked = fixtures.make_poses(manifest, 2)
        shard_writer.write_shard(tmp_path, sequence=sequence, poses=poses, parked=parked)

    sequences = [pool_contract.shard_sequence_from_path(path) for path in shard_reader.discover_shard_paths(tmp_path)]

    assert sequences == [0, 2, 11]


def test_shard_discovery_can_skip_already_ingested_sequences(tmp_path: Path):
    """The ingest loop only reads what it has not seen."""
    manifest = fixtures.build_fixture_pool(tmp_path, num_variations=2)
    fixtures.append_fixture_shard(tmp_path, manifest, num_variations=2, first_variation_index=2)

    remaining = shard_reader.discover_shard_paths(tmp_path, after_sequence=0)

    assert [pool_contract.shard_sequence_from_path(path) for path in remaining] == [1]


# ---------------------------------------------------------------------------
# Parking
# ---------------------------------------------------------------------------


def test_parked_prims_sit_at_the_manifest_parking_offset(tmp_path: Path):
    """Parking is what hides an inactive asset, so a parked prim must be at the declared offset."""
    manifest = fixtures.build_fixture_pool(
        tmp_path,
        num_variations=2,
        parkable_paths=(fixtures.FIXTURE_PRIM_PATHS[1],),
        parked_variation_indices=(1,),
    )
    shard = shard_reader.load_shard(pool_contract.shard_poses_path(tmp_path, 0), manifest)

    shard_reader.check_parked_poses_match_manifest(shard, manifest)

    assert shard.parked[1, 1]
    assert not shard.parked[0, 1]
    assert np.allclose(shard.poses[1, 1, :3], fixtures.FIXTURE_PARKING_OFFSET)


def test_parking_a_prim_the_manifest_did_not_permit_is_reported(tmp_path: Path):
    """A prim a task depends on must not be silently parked out of the scene."""
    manifest = fixtures.build_manifest()
    shard_writer.write_manifest(tmp_path, manifest)
    poses, parked = fixtures.make_poses(manifest, 2)
    parked[0, 0] = True
    poses_path = shard_writer.write_shard(tmp_path, sequence=0, poses=poses, parked=parked)
    shard = shard_reader.load_shard(poses_path, manifest)

    violations = shard_reader.unparkable_violations(shard, manifest)

    assert len(violations) == 1
    assert "Props/mug_00" in violations[0]


# ---------------------------------------------------------------------------
# Write plan
# ---------------------------------------------------------------------------


def test_write_plan_skips_static_prims():
    """Only rigid and articulated prims accept a root pose write."""
    manifest = PoolManifest(
        schema_version=pool_contract.SCHEMA_VERSION,
        parking_offset_xyz=(0.0, 0.0, -25.0),
        shard=ShardLayout(variations_per_shard=2),
        prims=[
            PoolPrimSpec(path="Fixtures/counter", object_type=ObjectType.BASE),
            PoolPrimSpec(path="Props/mug", object_type=ObjectType.RIGID),
            PoolPrimSpec(path="Fixtures/drawer", object_type=ObjectType.ARTICULATION),
        ],
    )

    plan = build_variation_write_plan(manifest)

    # Indices stay tied to the inventory, so filtering never shifts a shard column.
    assert plan == [(1, "Props_mug"), (2, "Fixtures_drawer")]


def test_write_plan_rejects_an_inventory_with_nothing_writable():
    """A pool of only static prims could never apply a variation."""
    manifest = PoolManifest(
        schema_version=pool_contract.SCHEMA_VERSION,
        parking_offset_xyz=(0.0, 0.0, -25.0),
        shard=ShardLayout(variations_per_shard=2),
        prims=[PoolPrimSpec(path="Fixtures/counter", object_type=ObjectType.BASE)],
    )

    with pytest.raises(AssertionError, match="no rigid or articulated prims"):
        build_variation_write_plan(manifest)


def test_object_reference_name_flattens_the_prim_path():
    """Scene entity names come from the inventory path, so both halves agree without a lookup."""
    assert object_reference_name("Props/mug_00") == "Props_mug_00"
    assert object_reference_name("/Props/mug_00/") == "Props_mug_00"


# ---------------------------------------------------------------------------
# Pool consumption
# ---------------------------------------------------------------------------


def _drawn_variation_indices(pool: LiveVariationPool, env_ids: list[int]) -> list[int]:
    """Return the variation index each environment received, decoded from its poses."""
    poses = pool.gather_poses(env_ids)
    return [fixtures.variation_index_from_pose_row(poses[row, 0]) for row in range(len(env_ids))]


def test_parallel_environments_draw_different_variations(tmp_path: Path):
    """The point of the pool: environments running side by side must not look identical."""
    fixtures.build_fixture_pool(tmp_path, num_variations=4)
    pool = LiveVariationPool(tmp_path, num_envs=2, device="cpu", watch=False)
    try:
        pool.wait_for_initial_fill()

        assert _drawn_variation_indices(pool, [0, 1]) == [0, 1]
    finally:
        pool.close()


def test_a_reset_draws_a_variation_no_environment_has_seen(tmp_path: Path):
    """A reset has to advance the stream, otherwise episodes repeat their scene."""
    fixtures.build_fixture_pool(tmp_path, num_variations=4)
    pool = LiveVariationPool(tmp_path, num_envs=2, device="cpu", watch=False)
    try:
        pool.wait_for_initial_fill()
        first = _drawn_variation_indices(pool, [0, 1])

        second = _drawn_variation_indices(pool, [0, 1])

        assert second == [2, 3]
        assert not set(first) & set(second)
    finally:
        pool.close()


def test_a_partial_reset_only_consumes_for_the_resetting_environments(tmp_path: Path):
    """Resetting one environment must not burn variations the others would have received."""
    fixtures.build_fixture_pool(tmp_path, num_variations=4)
    pool = LiveVariationPool(tmp_path, num_envs=2, device="cpu", watch=False)
    try:
        pool.wait_for_initial_fill()
        _drawn_variation_indices(pool, [0, 1])

        assert _drawn_variation_indices(pool, [1]) == [2]
        assert pool.last_variation_per_env == [0, 2]
    finally:
        pool.close()


def test_a_dry_pool_reuses_resident_variations_instead_of_stalling(tmp_path: Path):
    """The simulation must keep running when the producer falls behind; starvation is a metric."""
    fixtures.build_fixture_pool(tmp_path, num_variations=2)
    pool = LiveVariationPool(tmp_path, num_envs=2, device="cpu", watch=False)
    try:
        pool.wait_for_initial_fill()
        _drawn_variation_indices(pool, [0, 1])

        reused = _drawn_variation_indices(pool, [0, 1])

        assert pool.stats.starved_draws == 2
        # Even starved, the two environments differ, because the reuse rotates through the ring.
        assert reused[0] != reused[1]
    finally:
        pool.close()


def test_the_pool_picks_up_a_shard_the_producer_adds_later(tmp_path: Path):
    """Ingestion is continuous: a shard renamed into place becomes drawable without a rebuild."""
    manifest = fixtures.build_fixture_pool(tmp_path, num_variations=2)
    pool = LiveVariationPool(tmp_path, num_envs=2, device="cpu", watch=False)
    try:
        pool.wait_for_initial_fill()
        assert _drawn_variation_indices(pool, [0, 1]) == [0, 1]

        fixtures.append_fixture_shard(tmp_path, manifest, num_variations=2, first_variation_index=2)
        pool.poll_for_new_shards()

        assert _drawn_variation_indices(pool, [0, 1]) == [2, 3]
        assert pool.stats.starved_draws == 0
    finally:
        pool.close()


def test_the_ingest_thread_picks_up_new_shards_on_its_own(tmp_path: Path):
    """With watching on, no caller has to poll; the pool refills beside the simulation thread."""
    manifest = fixtures.build_fixture_pool(tmp_path, num_variations=2)
    pool = LiveVariationPool(tmp_path, num_envs=2, device="cpu", poll_interval_s=0.01)
    try:
        pool.wait_for_initial_fill()
        fixtures.append_fixture_shard(tmp_path, manifest, num_variations=2, first_variation_index=2)

        deadline = time.monotonic() + 10.0
        while pool.stats.ingested_shards < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        pool.drain_pending()

        assert pool.stats.ingested_variations == 4
    finally:
        pool.close()


def test_the_ring_buffer_drops_variations_that_were_never_consumed(tmp_path: Path):
    """A bounded ring is what keeps device memory flat; overrun is counted, not fatal."""
    manifest = fixtures.build_fixture_pool(tmp_path, num_variations=2)
    pool = LiveVariationPool(tmp_path, num_envs=1, device="cpu", min_variations_per_env=2, watch=False)
    try:
        assert pool.capacity == 2
        for shard in range(1, 3):
            fixtures.append_fixture_shard(tmp_path, manifest, num_variations=2, first_variation_index=2 * shard)
        pool.poll_for_new_shards()

        # Six variations passed through a ring holding two, so only the newest pair is resident.
        assert pool.stats.ingested_variations == 6
        assert pool.resident == 2
        assert _drawn_variation_indices(pool, [0]) == [4]
        assert pool.stats.skipped_variations == 4
    finally:
        pool.close()


def test_a_shard_larger_than_the_ring_keeps_its_newest_variations(tmp_path: Path):
    """Wrapping within one copy must not leave older rows shadowing newer ones."""
    manifest = fixtures.build_manifest(variations_per_shard=2)
    shard_writer.write_manifest(tmp_path, manifest)
    poses, parked = fixtures.make_poses(manifest, 6)
    shard_writer.write_shard(tmp_path, sequence=0, poses=poses, parked=parked)
    pool = LiveVariationPool(tmp_path, num_envs=1, device="cpu", min_variations_per_env=2, watch=False)
    try:
        assert pool.capacity == 2

        assert _drawn_variation_indices(pool, [0]) == [4]
    finally:
        pool.close()


def test_drawing_before_anything_is_resident_fails_loudly(tmp_path: Path):
    """Silently writing a zero pose would drop every object at the world origin."""
    manifest = fixtures.build_manifest()
    shard_writer.write_manifest(tmp_path, manifest)
    pool = LiveVariationPool(tmp_path, num_envs=1, device="cpu", watch=False)
    try:
        with pytest.raises(AssertionError, match="no resident variations"):
            pool.gather_poses([0])
    finally:
        pool.close()


def test_waiting_for_a_pool_the_producer_never_filled_times_out(tmp_path: Path):
    """A missing producer should be reported at build time, not as a mysterious reset failure."""
    manifest = fixtures.build_manifest()
    shard_writer.write_manifest(tmp_path, manifest)
    pool = LiveVariationPool(tmp_path, num_envs=1, device="cpu", watch=False)
    try:
        with pytest.raises(AssertionError, match="is the producer running"):
            pool.wait_for_initial_fill(timeout_s=0.05)
    finally:
        pool.close()


def test_the_pool_handle_survives_the_event_term_deepcopy(tmp_path: Path):
    """Isaac Lab deep-copies event params; copying the pool would orphan its ingest thread."""
    import copy

    from isaaclab_arena.live_scene.variation_pool import LiveVariationPoolHandle

    fixtures.build_fixture_pool(tmp_path, num_variations=2)
    pool = LiveVariationPool(tmp_path, num_envs=1, device="cpu", watch=False)
    try:
        handle = LiveVariationPoolHandle(pool)

        copied = copy.deepcopy(handle)

        assert copied is handle
        assert copied.pool is pool
        assert not hasattr(handle, "__dict__")
    finally:
        pool.close()


def test_gathered_poses_are_indexed_by_the_manifest_inventory_order(tmp_path: Path):
    """Ordering is positional, so column i must be inventory entry i and nothing else."""
    manifest = fixtures.build_fixture_pool(tmp_path, num_variations=2)
    pool = LiveVariationPool(tmp_path, num_envs=1, device="cpu", watch=False)
    try:
        poses = pool.gather_poses([0])

        assert poses.shape == (1, manifest.num_prims, pool_contract.POSE_COLUMNS)
        # make_poses() encodes the inventory position in y, so the columns are distinguishable.
        assert pytest.approx(float(poses[0, 0, 1]), abs=1e-6) == 0.1
        assert pytest.approx(float(poses[0, 1, 1]), abs=1e-6) == 0.2
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Performance regression
# ---------------------------------------------------------------------------


def test_the_draw_path_stays_inside_its_latency_budget():
    """Guards the property the whole design rests on: a reset draw is a gather, not a loop.

    Reintroducing per-environment work, a host-to-device copy, or USD parsing would blow this.
    """
    from isaaclab_arena.live_scene import pool_benchmark

    result = pool_benchmark.run_benchmark(
        num_envs=pool_benchmark.BUDGET_NUM_ENVS,
        num_prims=pool_benchmark.BUDGET_NUM_PRIMS,
        num_shards=2,
        measure_overlap=False,
    )

    assert result.draw_p99_ms < pool_benchmark.DRAW_LATENCY_BUDGET_MS, (
        f"variation draw p99 was {result.draw_p99_ms:.3f} ms at {result.num_envs} envs and "
        f"{result.num_prims} prims, over the {pool_benchmark.DRAW_LATENCY_BUDGET_MS} ms budget"
    )
    assert result.ingest_variations_per_s > pool_benchmark.MIN_INGEST_VARIATIONS_PER_S, (
        f"shard ingest managed only {result.ingest_variations_per_s:.0f} variations/s, under the "
        f"{pool_benchmark.MIN_INGEST_VARIATIONS_PER_S:.0f} floor"
    )


def test_ingestion_does_not_steal_the_gil_from_the_simulation_thread():
    """The numpy-only shard format exists so ingestion overlaps rather than contends.

    A drop here is the signal that something on the ingest path started calling into USD.
    """
    from isaaclab_arena.live_scene import pool_benchmark

    manifest = pool_benchmark.build_synthetic_manifest(num_prims=200, variations_per_shard=256)
    import tempfile

    with tempfile.TemporaryDirectory(prefix="arena-pool-gil-") as temp_dir:
        pool_dir = Path(temp_dir)
        shard_writer.write_manifest(pool_dir, manifest)
        poses = np.zeros((256, manifest.num_prims, pool_contract.POSE_COLUMNS), dtype=np.float32)
        for sequence in range(4):
            shard_writer.write_shard(pool_dir, sequence=sequence, poses=poses)

        ratio = pool_benchmark.measure_gil_overlap(pool_dir, manifest, duration_s=0.5)

    assert ratio > pool_benchmark.MIN_GIL_OVERLAP_RATIO, (
        f"compute throughput fell to {ratio:.2f} of baseline while ingesting, under the "
        f"{pool_benchmark.MIN_GIL_OVERLAP_RATIO} floor; the ingest path may have started holding the GIL"
    )
