# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""An Arena environment read from a scene's randomization contract and ``mdp@1`` MDP root.

``isaaclab_contract.mdp_root.env_cfg_from_usd`` reads the file into a manager-based environment
configuration. This module hands its parts to Arena through Arena's own three slots, so the rest of Arena
(the builder, the policy runner, metrics, recorders, evaluation reports) works unchanged:

========================  ==============================================================================
Arena slot                From the file
========================  ==============================================================================
embodiment                The MDP root's ``Scene/<name>`` articulation, its observations and actions
scene                     Every other entity: the file once per environment, its rigid bodies and
                          articulations (option clones of an unrolled asset included)
task                      Rewards, terminations (``success`` from the ``Task`` stages), curriculum, reset
                          events (scene defaults, the pool row, then the contract's ``reset`` group),
                          ``Metrics`` and the instruction
========================  ==============================================================================

The MDP root's physics step and decimation replace Arena's defaults; the physics backend stays Arena's.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import make_configclass

SCENE_SETTINGS = {
    "num_envs", "env_spacing", "lazy_sensor_update", "replicate_physics", "filter_collisions",
    "clone_in_fabric", "physics_clone", "visualize_clone",
}  # fmt: skip
"""``InteractiveSceneCfg`` fields that are settings, not entities."""


def _entries(cfg: Any) -> dict[str, Any]:
    """A config's entries by name, in declaration order (empty for ``None``).

    isaaclab_contract sets a manager's terms, and the scene's entities, as attributes rather than
    dataclass fields. Arena combines configs by their fields, so both are read here, fields first.
    """
    if cfg is None:
        return {}
    out = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)} if dataclasses.is_dataclass(cfg) else {}
    out.update({k: v for k, v in vars(cfg).items() if not k.startswith("_") and k not in out})
    return out


def _configclass(name: str, entries: dict[str, Any]) -> Any | None:
    """A configclass instance whose fields are ``entries`` (``None`` when there are none)."""
    entries = {k: v for k, v in entries.items() if v is not None}
    if not entries:
        return None
    return make_configclass(name, [(k, type(v), v) for k, v in entries.items()])()


def _manager(name: str, cfg: Any) -> Any | None:
    """A manager's terms as a configclass Arena can combine."""
    return _configclass(name, _entries(cfg))


def _contract_key(lab_cfg: Any) -> str:
    """The key isaaclab_contract registered this MDP under, read from a term that carries it."""
    for manager in (lab_cfg.rewards, lab_cfg.terminations, lab_cfg.curriculum, lab_cfg.events):
        for term in _entries(manager).values():
            params = getattr(term, "params", None) or {}
            if "key" in params:
                return params["key"]
    raise ValueError("no term of the MDP root carries the contract key; is it an isaaclab_contract configuration?")


class ContractEmbodiment(EmbodimentBase):
    """The robot the scene already holds, with the MDP root's observations and actions.

    Unlike Arena's registered embodiments it spawns nothing: its articulation is a prim of the scene file.
    """

    registry_name = "contract_embodiment"

    def __init__(self, robot: str, robot_cfg: Any, actions: Any, observations: Any):
        super().__init__()
        self.name = robot  # the scene entity name the MDP's terms address it by
        self.scene_config = _configclass("ContractEmbodimentSceneCfg", {robot: robot_cfg})
        self.action_config = actions
        self.observation_config = observations

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_action_cfg(self) -> Any:
        return self.action_config

    def get_observation_cfg(self) -> Any:
        return self.observation_config

    def get_events_cfg(self) -> Any:
        return None  # the task's reset_scene restores the robot with the rest of the scene


class ContractScene(Scene):
    """Every entity of the scene file but the robot: the file itself (``scene_template``) and its bodies."""

    def __init__(self, entities: dict[str, Any], variations: list | None = None):
        super().__init__(assets=[])
        self.entities = entities
        self.variations = variations or []

    def get_scene_cfg(self) -> Any:
        return _configclass("ContractSceneCfg", self.entities)

    def get_asset_variations(self) -> dict[str, list]:
        """The contract's distributions, as Arena variations on the ``contract`` host (Hydra: ``contract.<name>``)."""
        from isaaclab_arena.contract.variations import HOST

        return {HOST: self.variations} if self.variations else {}


class ContractTask(TaskBase):
    """The MDP root's rewards, terminations, curriculum, reset events and metrics."""

    def __init__(
        self,
        lab_cfg: Any,
        metrics: list,
        instruction: str | None,
        view: tuple | None = None,
        variations: list | None = None,
        seed: int = 0,
        reset_variations: list | None = None,
    ):
        super().__init__(episode_length_s=lab_cfg.episode_length_s, task_description=instruction)
        self.lab_cfg = lab_cfg
        self.metrics = metrics
        self.view = view
        self.variations = variations  # None: play a pool back; a list: Arena draws, the contract realizes
        self.reset_variations = reset_variations or []
        self.seed = seed

    def get_viewer_cfg(self):
        """Environment 0 seen from the scene's own camera, when it has one; Arena's default otherwise."""
        from isaaclab.envs import ViewerCfg

        if self.view is None:
            return ViewerCfg()
        eye, lookat = self.view
        return ViewerCfg(eye=eye, lookat=lookat, origin_type="env", env_index=0)

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> Any:
        """The MDP root's terminations as Arena's: ``success`` is the completion criterion (Arena's progress tracker
        ends the episode on it), a time-out term is Arena's timeout, every other term is a failure."""
        from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
        from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

        terms = dict(_entries(self.lab_cfg.terminations))
        success = terms.pop("success", None)
        timeout = [name for name, term in terms.items() if getattr(term, "time_out", False)]
        for name in timeout:
            terms.pop(name)
        return TaskTerminationCfg(
            timeout_s=self.lab_cfg.episode_length_s if timeout else None,
            success=[CompletionCriteria(name="success", predicate_sequence=[success])] if success is not None else [],
            failures=terms,
        )

    def get_events_cfg(self) -> Any:
        from isaaclab_arena.contract.pool import contract_pool_event_type
        from isaaclab_arena.contract.realize import contract_scene_event_type

        # In order: reset_scene (defaults), the scene group (Arena's draws realized by the contract, or a
        # pool row), then the contract's reset group.
        events = _manager("ContractEventsCfg", self.lab_cfg.events)
        if getattr(events, "pool", None) is not None:
            if self.variations is None:
                events.pool.func = contract_pool_event_type()
            else:
                events.pool.func = contract_scene_event_type()
                events.pool.params = {**events.pool.params, "variations": self.variations, "seed": self.seed}
        if self.reset_variations and getattr(events, "contract", None) is not None:
            from isaaclab_arena.contract.realize import contract_reset_event_type

            events.contract.func = contract_reset_event_type()
            events.contract.params = {**events.contract.params, "variations": self.reset_variations}
        return events

    def get_rewards_cfg(self) -> Any:
        return _manager("ContractRewardsCfg", self.lab_cfg.rewards)

    def get_curriculum_cfg(self) -> Any:
        return _manager("ContractCurriculumCfg", self.lab_cfg.curriculum)

    def get_commands_cfg(self) -> Any:
        return _manager("ContractCommandsCfg", self.lab_cfg.commands)

    def get_metrics(self) -> list:
        return self.metrics

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        raise NotImplementedError("an MDP root has no Mimic configuration")


def scene_camera_view(stage) -> tuple[tuple, tuple] | None:
    """(eye, look-at) of the first camera below the template's root, in the template's (environment) frame."""
    from pxr import Gf, Usd, UsdGeom

    template = stage.GetDefaultPrim()
    for prim in Usd.PrimRange(template):
        if prim.IsA(UsdGeom.Camera):
            xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            eye = xform.Transform(Gf.Vec3d(0, 0, 0))
            lookat = eye + xform.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized() * 2.0
            return tuple(float(v) for v in eye), tuple(float(v) for v in lookat)
    return None


def _viewer_in_world(env_cfg) -> None:
    """Move an environment-relative viewer to world coordinates.

    The video recorder copies the viewer's ``eye`` and ``lookat`` as world positions and ignores
    ``origin_type``, so a view of environment ``env_index`` is offset here by that environment's grid origin.
    """
    viewer = env_cfg.viewer
    if viewer.origin_type != "env":
        return
    from isaaclab.cloner import grid_transforms

    origins, _ = grid_transforms(env_cfg.scene.num_envs, env_cfg.scene.env_spacing)
    origin = [float(v) for v in origins[viewer.env_index]]
    viewer.eye = tuple(e + o for e, o in zip(viewer.eye, origin))
    viewer.lookat = tuple(e + o for e, o in zip(viewer.lookat, origin))
    viewer.origin_type = "world"


def contract_variations(nominal: str) -> dict[str, list]:
    """The contract's ``scene`` and ``reset`` distributions as Arena variations, by group (from the nominal scene)."""
    from pxr import Usd
    from usd_randomizer import contract as rc

    from isaaclab_arena.contract.variations import ContractVariation, distribution_specs

    (contract,), problems = rc.read(Usd.Stage.Open(str(nominal)))
    assert contract.ok and not problems, f"the contract does not read: {contract.errors} {problems}"
    out = {}
    for group in ("scene", "reset"):
        specs, left = distribution_specs(contract, group)
        if left:
            print(
                f"[contract] Drawn by the contract, not Arena ({group}):"
                f" {', '.join(p.rsplit('/', 1)[-1] for p in left)}"
            )
        searched = [s.name for s in specs if not s.cfg.enabled]
        if searched:
            print(f"[contract] Searched by the contract's solver unless enabled: {', '.join(searched)}")
        out[group] = [ContractVariation(spec) for spec in specs]
    return out


def _import_isaaclab_contract():
    try:
        from isaaclab_contract import mdp_root
    except ImportError as error:
        raise ImportError(
            "The contract environment needs isaaclab_contract (the randomization contract's Isaac Lab executor)"
            " and usd_randomizer on PYTHONPATH."
        ) from error
    return mdp_root


def build_contract_environment(
    name: str,
    scene_usd: str,
    *,
    pool_dir: str | None = None,
    mdp_root: str | None = None,
    num_envs: int = 1,
    seed: int = 0,
    device: str = "cuda:0",
    scene_draws: str = "arena",
):
    """An :class:`IsaacLabArenaEnvironment` for a scene file with a contract and an ``mdp@1`` MDP root.

    Args:
        name: Environment name.
        scene_usd: The scene file. With ``pool_dir`` the pool's own template is used and this only names it.
        pool_dir: A contract variation pool (a directory with ``layout.json``), or ``None`` for every
            environment on the file's own sample.
        mdp_root: The MDP root's path, when the file holds several.
        num_envs: Number of environments.
        seed: Seed of the contract's draws (the ``reset`` group, and what Arena leaves to the contract).
        device: Simulation device.
        scene_draws: ``arena``: the scene group's distributions are Arena variations, sampled at every reset
            and realized by the contract's solver (``pool_dir`` then only supplies the template's layout);
            ``pool``: every reset plays back a row of the variation pool in ``pool_dir``.
    """
    from pxr import Usd

    from isaaclab_arena.contract.metrics import metrics_from_mdp_root
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    mdp = _import_isaaclab_contract()
    # The physics configuration isaaclab_contract writes is not used: Arena's --presets choose the backend.
    lab_cfg = mdp.env_cfg_from_usd(
        scene_usd, num_envs=num_envs, mdp_root=mdp_root, physics="newton", device=device, seed=seed, pool=pool_dir
    )

    from isaaclab_contract.pool_template import Layout

    template = scene_usd
    if pool_dir is not None:
        template = Layout.load(Path(pool_dir) / "layout.json").template
    stage = Usd.Stage.Open(str(template))
    root = stage.GetPrimAtPath(mdp_root or mdp.find_mdp_root(stage))
    scene_terms = root.GetChild("Scene")
    assert (
        scene_terms and scene_terms.GetChildren()
    ), f"{root.GetPath()} has no Scene/<robot> term naming its embodiment"
    robot = scene_terms.GetChildren()[0].GetName()

    from isaaclab.scene import InteractiveSceneCfg

    # every InteractiveSceneCfg field is a setting (num_envs, class_type, ...); the rest are the scene's entities
    settings = SCENE_SETTINGS | {f.name for f in dataclasses.fields(InteractiveSceneCfg)}
    entities = {k: v for k, v in _entries(lab_cfg.scene).items() if k not in settings and v is not None}
    assert robot in entities, f"the MDP root's embodiment {robot!r} is not a scene entity: {sorted(entities)}"

    mapped = metrics_from_mdp_root(root, _contract_key(lab_cfg))
    if mapped.unmapped:
        print(f"[contract] Metrics with no Arena metric, not computed: {', '.join(mapped.unmapped)}")
    task_prim = root.GetChild("Task")
    instruction_attr = task_prim.GetAttribute("mdp:instruction") if task_prim else None
    instruction = instruction_attr.Get() if instruction_attr else None

    embodiment = ContractEmbodiment(
        robot,
        entities[robot],
        _manager("ContractActionsCfg", lab_cfg.actions),
        _manager("ContractObservationsCfg", lab_cfg.observations),
    )
    by_group: dict[str, list] = {"scene": None, "reset": []}
    if scene_draws == "arena":
        assert pool_dir is not None, "Arena draws need the template's layout: pass a pool directory with layout.json"
        by_group = contract_variations(Layout.load(Path(pool_dir) / "layout.json").nominal)
    else:
        assert scene_draws == "pool", f"scene_draws must be 'arena' or 'pool', got {scene_draws!r}"
    hosted = (by_group["scene"] or []) + by_group["reset"]
    scene = ContractScene({k: v for k, v in entities.items() if k != robot}, hosted)
    task = ContractTask(
        lab_cfg,
        mapped.metrics,
        instruction,
        view=scene_camera_view(stage),
        variations=by_group["scene"],
        seed=seed,
        reset_variations=by_group["reset"],
    )

    def adopt_mdp_timing(env_cfg):
        # The MDP root's step and the template's spacing; Arena's physics backend is left alone.
        env_cfg.sim.dt = lab_cfg.sim.dt
        env_cfg.decimation = lab_cfg.decimation
        env_cfg.scene.env_spacing = lab_cfg.scene.env_spacing
        env_cfg.is_finite_horizon = lab_cfg.is_finite_horizon
        _viewer_in_world(env_cfg)
        return env_cfg

    from isaaclab_arena.contract.episodes import ContractEpisodeRecorderTermCfg

    return IsaacLabArenaEnvironment(
        name=name,
        embodiment=embodiment,
        scene=scene,
        task=task,
        env_cfg_callback=adopt_mdp_timing,
        # each finished episode's scene factors, for Arena's sensitivity analysis, and its contract seed
        episode_recorder_terms={"contract": ContractEpisodeRecorderTermCfg()},
    )
