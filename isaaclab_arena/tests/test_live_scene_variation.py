# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simulation tests for applying pooled scene variations to a generated background.

The fixture pool stands in for the generator, so these tests cover Arena's half end to end: a
generated base scene loaded as a background, object references derived from the manifest, and the
reset event writing one variation per environment.
"""

from __future__ import annotations

import pathlib
import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
NUM_ENVS = 2

# mug_00 moves every variation and is what divergence and freshness are read from; mug_01 is parked
# in every variation, so parking can be asserted without predicting which variation an env drew.
MOVING_PRIM = "Props/mug_00"
PARKED_PRIM = "Props/mug_01"


def _build_live_scene_env(pool_dir: pathlib.Path, num_variations: int):
    """Build a two-env environment whose background and variations come from a fixture pool.

    Args:
        pool_dir: Directory to create the fixture pool in.
        num_variations: Variations to pack into the pool's single shard.

    Returns:
        An ``(env, manifest)`` tuple. The caller owns closing the env.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.live_scene.generated_background import (
        GENERATED_SCENE_BACKGROUND_REGISTRY_NAME,
        build_object_references,
    )
    from isaaclab_arena.live_scene.pool_manifest import resolve_scene_usd
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tests.utils import variation_pool_fixtures as fixtures

    manifest = fixtures.build_fixture_pool(
        pool_dir,
        num_variations=num_variations,
        parkable_paths=(PARKED_PRIM,),
        parked_variation_indices=tuple(range(num_variations)),
        with_base_scene=True,
    )

    background_cls = AssetRegistry().get_asset_by_name(GENERATED_SCENE_BACKGROUND_REGISTRY_NAME)
    background = background_cls(usd_path=str(resolve_scene_usd(pool_dir, manifest)))
    references = build_object_references(manifest, background)
    scene = Scene(assets=[background, *references])

    arena_env = IsaacLabArenaEnvironment(name="live_scene_variation_test", scene=scene)
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=NUM_ENVS, live_variation_pool_dir=str(pool_dir)),
    )
    return builder.make_registered(), manifest


def _local_positions(env, scene_name: str):
    """Return each environment's root position for a scene entity, relative to its env origin.

    Args:
        env: The gym-wrapped environment.
        scene_name: Scene entity name to read.
    """
    asset = env.unwrapped.scene[scene_name]
    return (asset.data.root_pos_w - env.unwrapped.scene.env_origins).cpu()


def _test_variation_diverges_across_envs_and_refreshes_on_reset(simulation_app, tmp_path: pathlib.Path) -> bool:
    """Two envs draw different variations, and a reset draws variations neither has seen."""
    import torch

    from isaaclab_arena.live_scene.scene_entities import object_reference_name
    from isaaclab_arena.live_scene.variation_events import get_live_variation_pool
    from isaaclab_arena.tests.utils import variation_pool_fixtures as fixtures

    env, _manifest = _build_live_scene_env(tmp_path / "pool", num_variations=8)
    try:
        moving_name = object_reference_name(MOVING_PRIM)
        with torch.inference_mode():
            env.reset()
            first = _local_positions(env, moving_name)
            first_ids = [fixtures.variation_index_from_pose_row(first[env_id]) for env_id in range(NUM_ENVS)]

            env.reset()
            second = _local_positions(env, moving_name)
            second_ids = [fixtures.variation_index_from_pose_row(second[env_id]) for env_id in range(NUM_ENVS)]

        print(f"variation ids per env: first reset {first_ids}, second reset {second_ids}")
        assert first_ids[0] != first_ids[1], f"parallel envs drew the same variation: {first_ids}"
        assert not set(first_ids) & set(second_ids), f"reset replayed a variation: {first_ids} then {second_ids}"

        pool = get_live_variation_pool(env)
        assert pool is not None, "the reset event did not expose its variation pool"
        assert pool.stats.starved_draws == 0, f"pool starved with 8 variations available: {pool.stats.as_dict()}"
    except Exception as error:
        print(f"Error: {error}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_parked_prims_land_at_the_manifest_offset(simulation_app, tmp_path: pathlib.Path) -> bool:
    """A parked prim is written to the pool's parking offset, which is what hides it."""
    import torch

    from isaaclab_arena.live_scene.scene_entities import object_reference_name

    env, manifest = _build_live_scene_env(tmp_path / "pool", num_variations=4)
    try:
        with torch.inference_mode():
            env.reset()
            parked = _local_positions(env, object_reference_name(PARKED_PRIM))

        expected = torch.tensor(manifest.parking_offset_xyz)
        print(f"parked positions per env: {parked.tolist()}, expected {expected.tolist()}")
        for env_id in range(NUM_ENVS):
            assert torch.allclose(parked[env_id], expected, atol=1e-3), (
                f"env {env_id} parked prim landed at {parked[env_id].tolist()} rather than the manifest "
                f"offset {expected.tolist()}"
            )
    except Exception as error:
        print(f"Error: {error}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def test_variation_diverges_across_envs_and_refreshes_on_reset(tmp_path: pathlib.Path):
    """Parallel envs must differ and resets must draw fresh variations."""
    result = run_function_with_persistent_simulation_app(
        _test_variation_diverges_across_envs_and_refreshes_on_reset,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, "Test failed"


def test_parked_prims_land_at_the_manifest_offset(tmp_path: pathlib.Path):
    """Parking replaces visibility toggling, so the offset has to be applied exactly."""
    result = run_function_with_persistent_simulation_app(
        _test_parked_prims_land_at_the_manifest_offset,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, "Test failed"
