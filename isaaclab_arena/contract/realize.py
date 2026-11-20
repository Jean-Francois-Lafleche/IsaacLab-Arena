# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Realize, at every reset, the scene Arena's variation samples ask for.

For each resetting environment the event samples every enabled :class:`ContractVariation` (so Arena's
variation recorder sees each value), pins the values in a contract request, and has the contract's reference
solver build a valid scene around them: it draws what Arena left to it (disabled variations, the
distributions a placement searches), searches placements and checks every required condition. The scene's
changes are read as one row of the template's channel layout and written through the same writers as a pool
row: physics through the simulation, rendering through Kit's Fabric.

A request the contract finds infeasible (a pinned value breaks a required condition, or no placement
search finds a valid pose around the pinned values) is sampled again by Arena, up to ``RETRIES`` times; past that the
contract draws the environment's scene itself, and the episode records the values it drew.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import torch
from pathlib import Path

RETRIES = 4
"""Arena resamples an environment whose pinned values the contract cannot satisfy this many times."""


class _RowQueue:
    """Stands in for a pool level: hands the pool event the rows the solver just realized."""

    def __init__(self):
        self.rows: list[np.ndarray] = []
        self.directory = Path("arena")
        self.draws = 0
        self.starved = 0

    def draw(self, count: int) -> np.ndarray:
        out, self.rows = self.rows[:count], self.rows[count:]
        assert len(out) == count, f"realized {len(out)} rows for {count} resetting environments"
        self.draws += count
        return np.stack(out)


def _warp_log_level_shim() -> None:
    """The solver quiets Warp through ``warp.config.log_level`` (Warp 1.14+); Isaac Sim's Warp 1.13 has none.

    Inert stand-ins make the solver's quieting a no-op there; a Warp that has the setting is left alone.
    """
    import warp

    if not hasattr(warp.config, "log_level"):
        warp.LOG_INFO = getattr(warp, "LOG_INFO", 20)
        warp.LOG_WARNING = getattr(warp, "LOG_WARNING", 30)
        warp.config.log_level = warp.LOG_INFO


class Realizer:
    """The contract and its solver on the template's nominal scene, turning requests into layout rows."""

    def __init__(self, layout, engine: str = "newton"):
        _warp_log_level_shim()
        from isaaclab_contract.pool_producer import RowReader
        from pxr import Usd
        from usd_randomizer import contract as rc
        from usd_randomizer.solver import Solver

        self.rc = rc
        self.stage = Usd.Stage.Open(layout.nominal)
        (self.contract,), problems = rc.read(self.stage)
        assert self.contract.ok and not problems, f"the contract does not read: {self.contract.errors} {problems}"
        self.solver = Solver(self.contract, self.stage, engine=engine)
        self.reader = RowReader(layout, self.stage)

    def realize(self, seed: int, pins: dict):
        """``(row, sample)``: the layout row of the verified sample for ``seed`` with ``pins``, or ``None`` and
        the unverified sample (its ``evidence`` says why)."""
        from pxr import Sdf
        from usd_randomizer.contract._changes import author

        sample = self.solver.sample(self.rc.Request(seed=seed, pin=pins, attempts=16))
        if sample.status != "verified":
            return None, sample
        session = self.stage.GetSessionLayer()
        layer = Sdf.Layer.CreateAnonymous("arena_sample.usda")
        author(layer, sample.changes)
        session.subLayerPaths.insert(0, layer.identifier)
        try:
            return self.reader.row(sample.changes), sample
        finally:
            session.subLayerPaths.remove(layer.identifier)


def contract_scene_event_type() -> type:
    """The reset event that realizes Arena's variation samples as a scene (see the module docstring)."""
    from isaaclab.managers import ManagerTermBase
    from isaaclab_contract.mdp_root import registered
    from isaaclab_contract.pool_template import Layout

    from isaaclab_arena.contract.pool import EpisodeStarts, contract_pool_event_type

    PoolWriter = contract_pool_event_type()

    class ContractSceneEvent(PoolWriter):
        def __init__(self, cfg, env):
            ManagerTermBase.__init__(self, cfg, env)  # not PoolWriter's: there is no pool to ingest
            params = cfg.params
            self.info = registered(params["key"])
            self.layout = Layout.load(Path(params["pool"]) / "layout.json")
            self.offsets = self.layout.offsets()
            self.entity_of = self.info.entities
            self.template = self.info.template_path
            self.renderer = None
            self.selection = {}
            env._contract_static = {}
            env._contract_choice = self.selection
            self.pools = [_RowQueue()]
            self.pool = self.pools[0]
            self.variations = params["variations"]
            self.base_seed = int(params.get("seed", 0))
            self.realizer = Realizer(self.layout)
            self.episodes = [0] * env.num_envs
            env._contract_episode_starts = EpisodeStarts(params["pool"], self.layout, self.offsets)

        def __call__(self, env, env_ids, key: str, pool: str, variations=None, seed: int = 0) -> None:
            if env_ids is None or isinstance(env_ids, slice):
                env_ids = torch.arange(env.num_envs, device=env.device)
            ids = [int(i) for i in env_ids]
            enabled = [v for v in self.variations if v.enabled]
            # Arena's draws for every resetting environment, reported to its recorder once they are final.
            raws = {v.name: v.draw(len(ids)) for v in enabled}
            rows, seeds, drawn_by = [], [], []
            for k, env_id in enumerate(ids):
                row = None
                for attempt in range(RETRIES + 1):
                    seed_k = self.base_seed + 1_000_003 * env_id + 7919 * self.episodes[env_id] + attempt
                    pins = {v.spec.path: v.pin_value(raws[v.name][k]) for v in enabled}
                    row, _ = self.realizer.realize(seed_k, pins)
                    if row is not None:
                        drawn_by.append("arena")
                        break
                    if attempt < RETRIES:  # Arena draws this environment again
                        for v in enabled:
                            raws[v.name][k] = v.draw(1)[0]
                if row is None:  # the contract draws this environment's scene itself
                    seed_k = self.base_seed + 1_000_003 * env_id + 7919 * self.episodes[env_id] + 999
                    row, sample = self.realizer.realize(seed_k, {})
                    assert row is not None, f"no valid scene for environment {env_id}: {sample.evidence}"
                    drawn_by.append("contract")
                    # record the values the scene was built from, not Arena's rejected draws
                    for v in enabled:
                        raws[v.name][k] = v.raw_from_draws(sample.draws, raws[v.name][k])
                rows.append(row)
                seeds.append(seed_k)
                self.episodes[env_id] += 1
            id_tensor = torch.as_tensor(ids)
            for v in enabled:
                v.record(raws[v.name], id_tensor)
            self.pool.rows = rows
            self._pending = {"seeds": seeds, "drawn_by": drawn_by}
            super().__call__(env, env_ids, key, pool)

    return ContractSceneEvent


def _rekeyed(pin, keys: dict[str, str]):
    """A per-target pin with its targets named as ``keys`` maps them; any other value as it is."""
    return {keys.get(t, t): value for t, value in pin.items()} if isinstance(pin, dict) else pin


def contract_reset_event_type() -> type:
    """isaaclab_contract's reset-group event, firing each environment with Arena's draws pinned.

    The contract's per-episode group (e.g. the arm's start offsets) is drawn by Arena's variations like the
    scene group: each resetting environment's values are pinned in the executor's request, and the executor
    fires that environment alone, so its writes (joint states, root poses) are the contract executor's own.
    """
    from isaaclab_contract.events import ContractEvent, _env_ids

    class ContractResetEvent(ContractEvent):
        _keys: dict | None = None
        _checked: bool = False

        def _check_applied(self, variations, pins: dict, rows: list) -> None:
            """Arena's pinned values are the ones the executor drew: a pin keyed wrongly is silently unused."""
            by_id = {r["node"]: r for r in rows}
            for v in variations:
                node_id = v.spec.path.rsplit("/", 1)[-1]
                pin = pins[v.spec.path]
                for target, value in pin.items() if isinstance(pin, dict) else [("", pin)]:
                    got = [r for r in rows if r["node"] == node_id and r.get("target", "") == target]
                    assert got, f"the executor drew no {node_id} for {target!r}: {sorted(by_id)}"
                    want = np.atleast_1d(np.asarray(value, dtype=float))
                    have = np.atleast_1d(np.asarray(got[0]["value"], dtype=float))
                    assert np.allclose(want, have, atol=1e-5), f"{node_id} {target}: pinned {want}, drawn {have}"
            self._checked = True

        def _executor_keys(self, variations) -> dict[str, str]:
            """Target identities as the variations name them (the nominal scene's) -> as the executor does.

            The executor's identities are relative to the live environment template (``/World/envs/env_0``),
            the variations' to the nominal scene: the same target, matched by the longest common path tail.
            """
            if self._keys is None:
                self._keys = {}
                structure = self.executor._structure
                for v in variations:
                    live = [ident for ident, _path in structure.slots(v.spec.path)]
                    for ident in v.spec.keys:
                        tail = ident.split("/", 1)[-1]
                        match = [x for x in live if x in (ident, tail) or x.endswith("/" + tail)]
                        assert len(match) == 1, f"no single executor target for {ident!r} among {live}"
                        self._keys[ident] = match[0]
            return self._keys

        def __call__(self, env, env_ids, contract, variations=None, seed: int = 0) -> None:
            ids = [int(i) for i in _env_ids(env_ids, env.num_envs)]
            enabled = [v for v in (variations or []) if v.enabled]
            if not enabled:
                super().__call__(env, env_ids, contract)
                return
            base = self.executor.request
            raws = {v.name: v.draw(len(ids)) for v in enabled}
            rekey = self._executor_keys(enabled)
            for k, env_id in enumerate(ids):
                pins = {v.spec.path: _rekeyed(v.pin_value(raws[v.name][k]), rekey) for v in enabled}
                self.executor.request = dataclasses.replace(base, pin=pins)
                start = len(self.executor._rows)
                try:
                    self.executor.fire(self.trigger, [env_id])
                finally:
                    self.executor.request = base
                if not self._checked:  # once: the executor drew exactly what Arena sampled
                    self._check_applied(enabled, pins, self.executor._rows[start:])
            id_tensor = torch.as_tensor(ids)
            for v in enabled:
                v.record(raws[v.name], id_tensor)

    return ContractResetEvent
