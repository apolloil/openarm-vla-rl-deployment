# Copyright 2025 Enactic, Inc.
#
# Visual-only presets for ``play_lift.py``: diffuse albedo + roughness for
# cube (object), table, and procedural floor. Does not affect RL state / physics.

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils


@dataclass(frozen=True)
class LiftPlayScenePreset:
    """RGB in linear 0–1; roughness for ``PreviewSurface``."""

    id: int
    name: str
    object_rgb: tuple[float, float, float]
    object_roughness: float
    table_rgb: tuple[float, float, float]
    table_roughness: float
    floor_rgb: tuple[float, float, float]
    floor_roughness: float


# ── 10 presets (index 0 … 9) ───────────────────────────────────────────────

PLAY_SCENE_PRESETS: tuple[LiftPlayScenePreset, ...] = (
    LiftPlayScenePreset(
        id=0,
        name="default",
        object_rgb=(0.80, 0.20, 0.10),
        object_roughness=0.50,
        table_rgb=(0.52, 0.34, 0.18),
        table_roughness=0.72,
        floor_rgb=(0.22, 0.22, 0.24),
        floor_roughness=0.92,
    ),
    LiftPlayScenePreset(
        id=1,
        name="ocean",
        object_rgb=(0.15, 0.55, 0.75),
        object_roughness=0.45,
        table_rgb=(0.18, 0.28, 0.42),
        table_roughness=0.68,
        floor_rgb=(0.12, 0.18, 0.26),
        floor_roughness=0.90,
    ),
    LiftPlayScenePreset(
        id=2,
        name="forest",
        object_rgb=(0.20, 0.62, 0.28),
        object_roughness=0.55,
        table_rgb=(0.38, 0.24, 0.12),
        table_roughness=0.78,
        floor_rgb=(0.16, 0.22, 0.14),
        floor_roughness=0.88,
    ),
    LiftPlayScenePreset(
        id=3,
        name="sunburst",
        object_rgb=(0.92, 0.72, 0.12),
        object_roughness=0.40,
        table_rgb=(0.58, 0.40, 0.22),
        table_roughness=0.70,
        floor_rgb=(0.35, 0.30, 0.22),
        floor_roughness=0.85,
    ),
    LiftPlayScenePreset(
        id=4,
        name="noir",
        object_rgb=(0.55, 0.08, 0.10),
        object_roughness=0.35,
        table_rgb=(0.12, 0.12, 0.13),
        table_roughness=0.55,
        floor_rgb=(0.06, 0.06, 0.07),
        floor_roughness=0.95,
    ),
    LiftPlayScenePreset(
        id=5,
        name="candy",
        object_rgb=(0.85, 0.25, 0.55),
        object_roughness=0.42,
        table_rgb=(0.72, 0.55, 0.48),
        table_roughness=0.65,
        floor_rgb=(0.40, 0.35, 0.48),
        floor_roughness=0.88,
    ),
    LiftPlayScenePreset(
        id=6,
        name="industrial",
        object_rgb=(0.90, 0.45, 0.08),
        object_roughness=0.38,
        table_rgb=(0.42, 0.44, 0.46),
        table_roughness=0.48,
        floor_rgb=(0.28, 0.28, 0.30),
        floor_roughness=0.82,
    ),
    LiftPlayScenePreset(
        id=7,
        name="arctic",
        object_rgb=(0.82, 0.88, 0.92),
        object_roughness=0.30,
        table_rgb=(0.62, 0.52, 0.40),
        table_roughness=0.60,
        floor_rgb=(0.45, 0.55, 0.62),
        floor_roughness=0.75,
    ),
    LiftPlayScenePreset(
        id=8,
        name="violet_wood",
        object_rgb=(0.45, 0.22, 0.70),
        object_roughness=0.48,
        table_rgb=(0.48, 0.32, 0.18),
        table_roughness=0.74,
        floor_rgb=(0.24, 0.18, 0.20),
        floor_roughness=0.90,
    ),
    LiftPlayScenePreset(
        id=9,
        name="neon_lime",
        object_rgb=(0.65, 0.92, 0.20),
        object_roughness=0.28,
        table_rgb=(0.32, 0.20, 0.12),
        table_roughness=0.76,
        floor_rgb=(0.14, 0.12, 0.22),
        floor_roughness=0.93,
    ),
)

NUM_PLAY_SCENE_PRESETS = len(PLAY_SCENE_PRESETS)


def get_play_scene_preset(index: int) -> LiftPlayScenePreset:
    if index < 0 or index >= NUM_PLAY_SCENE_PRESETS:
        raise ValueError(
            f"PLAY_SCENE_PRESET_ID must be in [0, {NUM_PLAY_SCENE_PRESETS - 1}], got {index}."
        )
    p = PLAY_SCENE_PRESETS[index]
    if p.id != index:
        raise RuntimeError(f"Preset id mismatch: index={index}, preset.id={p.id}")
    return p


def apply_play_scene_materials(env_cfg: object, preset_index: int) -> LiftPlayScenePreset:
    """Swap PreviewSurface on VLA procedural cube / table / floor before ``gym.make``."""
    preset = get_play_scene_preset(preset_index)
    scene = env_cfg.scene

    obj = scene.object
    scene.object = obj.replace(
        spawn=obj.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=preset.object_rgb,
                roughness=preset.object_roughness,
            )
        )
    )

    tbl = scene.table
    scene.table = tbl.replace(
        spawn=tbl.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=preset.table_rgb,
                roughness=preset.table_roughness,
            )
        )
    )

    pl = scene.plane
    scene.plane = pl.replace(
        spawn=pl.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=preset.floor_rgb,
                roughness=preset.floor_roughness,
            )
        )
    )

    return preset
