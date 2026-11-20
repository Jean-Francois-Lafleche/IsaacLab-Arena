# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation pools written for a randomization contract."""

from __future__ import annotations

from pathlib import Path

CONTRACT_POOL_LAYOUT = "layout.json"
"""A contract pool's channel layout: what each column of a pool row writes."""

ARENA_POOL_MANIFEST = "manifest.json"
"""An Arena live-scene pool's manifest (:mod:`isaaclab_arena.live_scene.pool_contract`)."""


def is_contract_pool(pool_dir: str | Path | None) -> bool:
    """Whether ``pool_dir`` holds a pool written for a randomization contract rather than an Arena manifest.

    A contract pool carries every quantity its contract's scene group varies (poses, joint state and
    dynamics, masses, friction, lights, materials, unrolled options); an Arena manifest pool carries poses,
    parked flags and scalars. The two are told apart by their index file.

    Args:
        pool_dir: Pool directory, or ``None``.
    """
    if pool_dir is None:
        return False
    path = Path(pool_dir)
    return (path / CONTRACT_POOL_LAYOUT).is_file() and not (path / ARENA_POOL_MANIFEST).is_file()


def _write_fabric(paths: list[str], attribute: str, values, matrix: bool) -> None:
    """Write one value per prim of ``paths`` into Fabric, the scene Kit renders from.

    Fabric rather than USD: a USD edit below a simulated body (an option on a drawer front) makes PhysX
    re-parse the body and invalidates the simulation's views, while Fabric writes reach only rendering.

    Args:
        paths: One prim path per environment.
        attribute: ``omni:xform`` (a local transform, ``omni:fabric:localMatrix``) or an attribute name.
        values: One row per path: 16 row-major numbers for a transform, else a scalar or a vector.
        matrix: Whether the values are transforms.
    """
    import numpy as np

    import usdrt
    from isaaclab.sim.utils.stage import get_current_stage_id

    stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
    for path, row in zip(paths, np.asarray(values)):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        value = [float(v) for v in np.ravel(row)]
        if matrix:
            name, type_name, data = (
                "omni:fabric:localMatrix",
                usdrt.Sdf.ValueTypeNames.Matrix4d,
                usdrt.Gf.Matrix4d(*value),
            )
        elif len(value) == 1:
            name, type_name, data = attribute, usdrt.Sdf.ValueTypeNames.Float, value[0]
        else:
            name, type_name, data = attribute, usdrt.Sdf.ValueTypeNames.Float3, usdrt.Gf.Vec3f(*value[:3])
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, type_name, True)
        attr.Set(data)


class EpisodeStarts:
    """What each environment's current episode started from: the pool row it drew, decoded, and its origin.

    The pool event records it at reset; the ``contract`` episode recorder reports it when the episode
    ends (:mod:`isaaclab_arena.contract.episodes`). A row's contract seed is found by matching the row
    against the pool's shards (``*.rows.npy`` beside ``*.seeds.npy``).
    """

    def __init__(self, pool_dir: str, layout, offsets: dict[str, int]):
        self.pool_dir = Path(pool_dir)
        self.layout = layout
        self.offsets = offsets
        self.current: dict[int, dict] = {}
        self.index: dict[bytes, int] = {}
        self._index_shards()

    def _index_shards(self) -> None:
        import numpy as np

        for rows_path in self.pool_dir.rglob("*.rows.npy"):
            seeds_path = rows_path.with_name(rows_path.name.replace(".rows.npy", ".seeds.npy"))
            if seeds_path.is_file():
                for row, seed in zip(np.load(rows_path), np.load(seeds_path)):
                    self.index[np.asarray(row).tobytes()] = int(seed)

    def seed_of(self, row) -> int | None:
        import numpy as np

        key = np.asarray(row).tobytes()
        if key not in self.index:
            self._index_shards()  # a producer may have written new shards since
        return self.index.get(key)

    def record(self, env_ids: list[int], level: str, rows, seeds=None, drawn_by=None) -> None:
        """Note the rows ``env_ids`` start from; ``seeds`` when they were realized live rather than pooled."""
        import numpy as np

        for k, (env_id, row) in enumerate(zip(env_ids, np.asarray(rows))):
            self.current[env_id] = {
                "seed": seeds[k] if seeds is not None else self.seed_of(row),
                "level": level,
                "drawn_by": drawn_by[k] if drawn_by is not None else "pool",
                "realized": decode_row(self.layout, self.offsets, row),
            }


def _leaf(path: str) -> str:
    """The object a channel's prim belongs to: the asset above its geometry, material or shader."""
    parts = [p for p in path.split("/") if p]
    if parts and parts[-1] == "Shader" and len(parts) > 1:
        return parts[-2]
    for marker in ("Geometry", "PhysicsMaterials"):
        if marker in parts and parts.index(marker) > 0:
            return parts[parts.index(marker) - 1]
    return parts[-1] if parts else path


def _label(value: str) -> str:
    """A readable option label: an asset's folder name, else a path's or file's last part."""
    text = str(value)
    parts = [p for p in text.replace("\\", "/").split("/") if p]
    if text.endswith((".usd", ".usda", ".usdc")) and len(parts) > 1:
        return parts[-2]
    return parts[-1].rsplit(".", 1)[0] if parts else text


def decode_row(layout, offsets: dict[str, int], row) -> dict:
    """A pool row as named factors: what the episode's scene actually was, channel by channel.

    A choice becomes its option's label (a variant, an asset, a material, shown or hidden), a joint
    state one number per joint, a transform its translation, a body pose its position, and any other
    value its number (or vector). Names follow the prim the channel writes.
    """
    import numpy as np

    values = np.asarray(row, dtype=float)
    out: dict = {}
    for channel in layout.channels:
        where = offsets[channel.name]  # a slice of the row (or its start column)
        cols = values[where] if isinstance(where, slice) else values[where : where + channel.width]
        prim, source = _leaf(channel.path), (channel.source or channel.kind).split(":")[-1]
        if channel.kind == "choice":
            k = int(round(cols[0]))
            labels = channel.values or channel.options
            label = _label(labels[k]) if 0 <= k < len(labels) else str(k)
            if source == "presence":
                label = "hidden" if k == 0 else "shown"
            out[f"{prim}.{source}"] = label
        elif channel.kind == "joint_pos" and channel.names:
            for name, value in zip(channel.names, cols):
                out[f"{prim}.{name}"] = round(float(value), 5)
        elif channel.kind == "xform":
            out[f"{prim}.offset"] = [round(float(v), 5) for v in cols[12:15]]
        elif channel.kind == "root_pose":
            out[f"{prim}.position"] = [round(float(v), 5) for v in cols[:3]]
        elif channel.width == 1:
            out[f"{prim}.{source}"] = round(float(cols[0]), 5)
        else:
            out[f"{prim}.{source}"] = [round(float(v), 5) for v in cols]
    return out


def contract_pool_event_type() -> type:
    """``isaaclab_contract``'s pool reset event, with its friction write tolerant of this Isaac Lab.

    ``isaaclab_contract`` reads material properties as a torch tensor; the PhysX tensor views of Isaac Lab
    3.0 betas return a Warp array. Every other channel is written by ``isaaclab_contract`` unchanged.
    """
    import torch

    from isaaclab_contract.pool_runtime import PoolEvent

    class ContractPoolEvent(PoolEvent):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            # What each environment's episode starts from, for the ``contract`` episode recorder.
            env._contract_episode_starts = EpisodeStarts(cfg.params["pool"], self.layout, self.offsets)

        def __call__(self, env, env_ids, key: str, pool: str) -> None:
            drawn = []
            for level, level_pool in enumerate(self.pools):
                original = level_pool.draw

                def draw(count, _original=original, _level=level):
                    rows = _original(count)
                    drawn.append((_level, rows))
                    return rows

                level_pool.draw = draw
            try:
                super().__call__(env, env_ids, key, pool)
            finally:
                for level_pool in self.pools:
                    level_pool.__dict__.pop("draw", None)
            if drawn:
                ids = env_ids if env_ids is not None and not isinstance(env_ids, slice) else range(env.num_envs)
                level, rows = drawn[-1]
                pending = self.__dict__.pop("_pending", {})
                env._contract_episode_starts.record(
                    [int(i) for i in ids],
                    self.pools[level].directory.name,
                    rows,
                    seeds=pending.get("seeds"),
                    drawn_by=pending.get("drawn_by"),
                )

        def _write(self, prim: str, env_ids, attribute: str, values, matrix: bool = False) -> None:
            """A visual write: through OVRTX's stage when it renders, else onto the USD stage Kit renders.

            isaaclab_contract writes what only rendering sees (option parking, light and wall transforms,
            light and shader inputs) through an OVRTX stage, and skips them without one. Arena renders with
            Kit, so the same values go to Kit's Fabric scene, one prim per resetting environment.
            """
            if self._render_scene() is not None:
                super()._write(prim, env_ids, attribute, values, matrix)
                return
            _write_fabric(self._paths(prim, env_ids), attribute, values, matrix)

        def _friction(self, channel, cols: torch.Tensor, ids: torch.Tensor) -> None:
            entity = self._entity(channel.path.split("/PhysicsMaterials")[0])
            view = getattr(entity, "root_view", None)
            if view is None or not hasattr(view, "get_material_properties"):
                return
            raw = view.get_material_properties()  # (instances, shapes, 3): static, dynamic, restitution
            if isinstance(raw, torch.Tensor):
                super()._friction(channel, cols, ids)
                return
            import warp as wp

            props = wp.to_torch(raw).clone().to(cols.device)
            props[ids, :, 0] = cols[:, 0:1]
            props[ids, :, 1] = cols[:, 1:2]
            view.set_material_properties(
                wp.from_torch(props.cpu().contiguous()), wp.from_torch(ids.cpu().to(torch.int32).contiguous())
            )

    return ContractPoolEvent
