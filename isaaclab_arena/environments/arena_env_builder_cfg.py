# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for compiling an Arena environment."""

from dataclasses import dataclass, field


# TODO(cvolk, 2026-07-06): [typed-config-migration] Replace this flat legacy-CLI-shaped configuration with
# nested scene, placement, and physics configs once the typed run configuration
# owns configuration composition.
@dataclass
class ArenaEnvBuilderCfg:
    """Configure how Arena builds an Isaac Lab environment."""

    num_envs: int = 1
    env_spacing: float = 30.0
    seed: int = 42
    solve_relations: bool = True
    placement_seed: int | None = None
    resolve_on_reset: bool | None = None
    disable_fabric: bool = False
    mimic: bool = False
    presets: str | None = None
    device: str = "cuda:0"
    language_instruction: str | None = None
    live_variation_pool_dir: str | None = None
    """Pool directory a generator writes scene variations into. When set, a reset event draws one
    variation per resetting environment. The generated scene owns object placement, so relations are
    not solved for these environments."""

    live_variation_scalar_overrides: dict[str, float] = field(default_factory=dict)
    """Pin a pool scalar to one value instead of drawing it, keyed by scalar name.

    The joints a reset drives come from the pool's manifest, which is what knows a joint's name and
    travel; this only takes control of the value, for a curriculum or to isolate one difficulty. A
    prim the pool placed resting in a driven joint is carried along, so the scene stays consistent.
    """

    live_variation_placed_objects: tuple[str, ...] = ()
    """Pool prims Arena poses itself instead of taking the pool's pose for, by inventory path.

    They are laid out together by Arena's relation solver on a surface the pool's env-graph sidecar
    names, and a fresh layout is drawn every reset. Use this for the objects a task is about:
    leaving them to the pool bakes a task decision into the scene generator, where changing it
    means regenerating every variation. Pass ``("*",)`` to place everything the sidecar gives a
    support for.
    """

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be greater than zero"
        for name, value in self.live_variation_scalar_overrides.items():
            assert 0.0 <= value <= 1.0, f"scalar override {name}={value} must be in [0, 1]"
