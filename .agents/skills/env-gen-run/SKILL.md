---
name: env-gen-run
description: Runs the end-to-end generated-environment pipeline — select SimReady assets, author a frozen-topology base scene, produce a live variation pool with usd-randomizer, verify its gates, then train or profile an Arena policy against it. Use when the user asks to replicate or reproduce an env gen run, build a generated scene, produce or refill a variation pool, train on generated scenes, work with the lift_object_live task, or pass --live_variation_pool_dir.
allowed-tools: Bash(git submodule *) Bash(git rev-parse *) Bash(docker exec *) Bash(docker ps *) Bash(uv *) Bash(id -un) Read Glob Grep Write Edit
---

# Generated Environment Run

Two processes, one on-disk contract. A **producer** (`usd-randomizer`) authors a fixed-topology base
scene plus a continuously refilled pool of pose-only variations; **Arena** loads the base scene as a
background and draws one variation per environment at reset. Nothing on Arena's reset path opens USD.

Read `isaaclab_arena/live_scene/pool_contract.py` first — its module docstring is the authoritative
pool layout. The producer's own guidance is far more detailed than this file and lives in the
submodule at `submodules/usd-randomizer-internal/.agents/skills/scene-gen-live/SKILL.md`; read it
before authoring a base scene or producer. This skill covers what Arena adds: where the two sides
meet, and the constraints that cost a debug cycle each when discovered by hand.

## Setup

The producer is vendored as a submodule, pinned to the branch carrying the live-variation feature:

```bash
git submodule update --init submodules/usd-randomizer-internal
cd submodules/usd-randomizer-internal && uv sync && cd -
```

Resolve this clone's container per the `dev-container` skill (`$ARENA_CONTAINER`).

**Where artifacts go.** `docker/run_docker.sh` bind-mounts `-v /tmp:/tmp`, so `/tmp` is the same
directory at the same path on host and in the container. Put build artifacts in
`/tmp/arena_live_build` and pools in `/tmp/arena_live_pools/<slug>_vN`. This keeps generated USD and
multi-GB scenes out of the repo while remaining visible to both sides. It also means a reboot loses
them, so treat the scripts below as disposable and this skill as the record.

**Which interpreter.** The split is not arbitrary:

| Stage | Runs | Why |
|---|---|---|
| Asset selection, base scene | Container, `/isaac-sim/python.sh` | SimReady assets are https URLs only Isaac Sim's asset resolver opens |
| Producer, gates | Host, `uv run` in the submodule | Needs `usd-core` and `newton`, not Isaac Sim |

To import the producer library inside the container, point at the submodule's source rather than
copying it:

```bash
PYTHONPATH=/workspaces/isaaclab_arena/submodules/usd-randomizer-internal/src
```

## Pipeline

Write each stage's script into `/tmp/arena_live_build/`, then run in order.

### 1. Select assets

Query SimReady for candidates per role and write `assets.json`. Runs in the container because it
imports Arena:

```python
from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SimReadySearchConfig,
    search_simready_objects,
)

catalogue = search_simready_objects(["plate", "dish", "saucer"], SimReadySearchConfig(max_results_per_object=24))
```

Keep every distinct candidate and let the base scene do the final rejection, which needs the asset
resolver. Two filters are mandatory, because SimReady matches on path substrings:

- **Collection**: require `/Residential/` for tableware, or it answers "plate" with an automotive
  camper plate. Furniture worth standing a robot at is spread across `/Residential/`, `/Hospital/`,
  and `/Industrial/`, and drawing from all three is what makes the table vary between environments.
- **Filename**: for furniture, require one of `table`, `desk`, `island`, `counter`, `workbench`.
  Furniture phrases otherwise answer with storage that has no usable top surface.

### 2. Author the base scene

A **superset**: every asset any variation could need, with topology frozen forever after. Export a
flattened `base_inlined.usdc` plus a `layout.json` of measurements the producer places from.

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /tmp/arena_live_build && \
   PYTHONPATH=/workspaces/isaaclab_arena/submodules/usd-randomizer-internal/src \
   /isaac-sim/python.sh build_base_scene.py"
```

Non-obvious requirements, each learned the hard way:

- **Multiplicity is the superset size.** A role with one instance looks identical in every
  environment, because Arena clones one environment's worth of scene. The instance count per role
  also caps how cluttered a table can get.
- **Author size variation up front.** A shard carries pose only, so a variation cannot rescale a
  prim. Give each instance its own scale factor. Do not drive heterogeneous tableware to one
  absolute size — it turns a wine flute into a straw.
- **Reject what a single root pose cannot move**: internal `UsdPhysics.Joint`s (PhysX snaps the
  pieces back together), no mesh, or no collider. Traverse with `Usd.TraverseInstanceProxies()`, or
  SimReady's geometry and physics hide inside a USD instance.
- **Exactly one rigid body per instance**, at the prim the shard poses. Nested bodies are rejected
  by the simulator and would simulate a prim other than the posed one.
- **Tables are kinematic**, so they stay pose-writable for parking without gravity toppling them.
  Replace the shipped SDF collider with an invisible box level with the top surface: SDF is a
  dynamic-body feature, and the kinematic-compatible approximations left props hovering above some
  tables and sinking through others. Make the box a child of the instance so parking takes it along.
  De-instance the subtree first (`SetInstanceable(False)`) — prototype prims are read-only.
- **Props keep their SimReady SDF colliders.** Convex decompositions destabilise stacks, because the
  hulls of a plate or bowl rim overlap whatever rests against them.
- **Fit the floor to the tables, not the reverse.** Real furniture comes in the heights it comes in.
  Drop the floor to the median table height, then assert every table presents its surface inside the
  arm's reachable band and covers the arm's working area — a variation must never place a prop over
  thin air.
- **Butt the table against the robot's pedestal.** A gap there collects props pushed off the near edge.
- **Put a slab under the parking offset.** A parked body in free fall never sleeps and arrives at the
  next reset carrying whatever speed it reached. Size the parking pitch from the widest thing that
  parks (`2 * max_radius + 0.1`); tables park too, and at the library default of 0.6 m two of them
  overlap by most of a metre, which is a contact the solver resolves every step.
- **Measure into `layout.json`**, per instance: offsets from origin to bottom and top faces,
  footprint radius, height, and for receptacles the *inner* surface height and clear inner radius.
  This is what lets the producer place anything on anything without opening geometry. A bowl's inner
  surface is not its bounding-box top: resting an object at the box top balances it on the rim.

### 3. Produce the pool

Transforms only, never topology. The inventory order **is** the shard column order.

```bash
cd submodules/usd-randomizer-internal
uv run python /tmp/arena_live_build/tabletop_variations.py \
  --pool /tmp/arena_live_pools/tabletop_v3 \
  --num-envs 128 --seed 0 --variations-per-shard 256 --max-variations 512
```

Producer flags: `--pool`, `--num-envs`, `--seed`, `--variations-per-shard` (256),
`--max-variations` (omit to run as a daemon), `--keep-shards` (8), `--interval` (0),
`--settle`/`--settle-substeps` (0.0 s / 2), `--no-archival-layers`.

Authoring rules, beyond the producer skill's:

- **The task target is never parkable.** A variation that parks the lift object is unsolvable.
  Place it first, onto an empty table, so it is both reachable and clear; placing it last leaves it
  squeezed in among the clutter.
- **Inset every placement by the prop's own footprint radius** plus a margin, against the chosen
  table's measured surface intersected with the arm's reach. Sampling centres inside a fixed box lets
  a wide plate overhang the edge by its whole radius; the arm then tips it onto the floor.
- **A piece set on a support needs room to spare** — roughly 80% of the support's clear inner radius
  — and must be no more than about three times as tall as it is wide. Otherwise it pivots about the
  rim when the arm brushes it, or goes over as soon as the solver settles.
- **Stack on the rim, put objects in the well.** Two different measured heights. Stacking a plate at
  the well height drives it through the rim below, and the solver answers an overlap by throwing one
  of the pieces off the table.

Run bounded while iterating; run unbounded beside training to keep refilling. Arena reuses variations
it has already seen when the pool runs dry, so a slow producer costs variety, not correctness.

### 4. Verify the gates

Check every gate from the producer skill (`GATE_POOL_OK`, `GATE_TOPOLOGY_STABLE_OK`,
`GATE_DELTA_ONLY_OK`, `GATE_PARKING_OK`, `GATE_SHARD_CONSISTENT_OK`, `GATE_ENV_MULTIPLICITY_OK`,
`GATE_RIGID_BODY_POLICY_OK`, `GATE_VARIATION_DISTINCT_OK`, `GATE_REPRODUCIBLE_OK`) using the
`check_*` helpers exported from `usd_randomizer`. All must pass with printed evidence.

Then confirm the pool has at least `num_envs` distinct rows, or parallel environments will look
identical:

```bash
cd submodules/usd-randomizer-internal && uv run python -c "
import glob, numpy as np
shards = sorted(glob.glob('/tmp/arena_live_pools/tabletop_v3/variations/*.poses.npy'))
rows = np.concatenate([np.load(s).reshape(np.load(s).shape[0], -1) for s in shards])
print(f'{rows.shape[0]} variations, {len({r.tobytes() for r in rows})} distinct')"
```

Render a few archival layers from `layers/`, never the base scene — the base scene has nothing parked
and every asset at its authored pose, so it tells you nothing about what Arena will see.

### 5. Train, or profile

The pool supplies both the scene and its per-reset variations, so `--live_variation_pool_dir` is the
only scene argument. Arena disables relation solving automatically when a pool is set.

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
   submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
   --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
   --task lift_object_live --rl_training_mode \
   --live_variation_pool_dir /tmp/arena_live_pools/tabletop_v3 \
   --num_envs 128 --env_spacing 3.0 --max_iterations 300 --framed_env_count 9 \
   --headless --enable_cameras --video --video_interval 500 --video_length 200"
```

To measure the pool's cost rather than train on it, `isaaclab_arena.live_scene.sim_variation_profile`
takes `--pool_dir` for a real pool, or builds a synthetic one and compares `--variant pool` against
`--variant static`.

## Reference run

The `tabletop_v3` pool, for comparison when something looks off:

| | |
|---|---|
| Inventory | 46 prims — 8 table, 8 plate, 8 bowl, 8 tray, 14 object |
| Lift target | `object_00`, the only non-parkable prim |
| Sized for | 128 environments, `env_spacing` 3.0 |
| Shards | 256 variations each, 512 produced at `--seed 0` |
| Parking offset | `(0, 0, -25)` |
| `scene.usdc` | 58.3 MB (SimReady geometry, inlined) |
| Producer rate | ~9.5 ms per variation (~105/s), no settle |

## Gotchas

- **`--video_env_count` does not exist**; the flag is `--framed_env_count`. Isaac Lab's launcher owns
  the `--video*` namespace and eats any `--video_`-prefixed flag before Arena's parser sees it.
- **Never edit an existing pool's inventory or its order.** Shards are packed positionally, so an
  edit silently invalidates every shard on disk. Produce into a new pool directory.
- **A rebuilt base scene invalidates the pool.** Arena checks the manifest's digests and refuses a
  pool rebuilt underneath a run. Bump to `<slug>_vN+1` rather than overwriting.
- **`SCHEMA_VERSION` must match** between `pool_contract.py` and the producer's
  `live_variation_contract.py`. Arena refuses a pool whose version it does not recognise; that is the
  intended failure, not a bug to work around.
- **Props sinking through the table** means PhysX dropped contacts on buffer overflow, not a collider
  problem. `lift_object_live` already enlarges them via `_size_physics_buffers`; a new generated-scene
  environment needs the same.
- **The pool must be readable at the same path inside the container.** `/tmp` is bind-mounted; a pool
  under `$HOME` is not.
- **The producer holds the base scene open for its lifetime.** Editing the base scene mid-run has no
  effect until the producer restarts.
