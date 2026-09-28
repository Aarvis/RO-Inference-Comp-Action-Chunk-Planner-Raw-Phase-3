"""Process-level Phase-3 speed and canonical-prompt contract."""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Mapping

from .config import OrigamiCompActionChunkRuntimeConfig


logger = logging.getLogger(__name__)


def configure_phase3_speed(
    config: OrigamiCompActionChunkRuntimeConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> OrigamiCompActionChunkRuntimeConfig:
    """Require ORIGAMI_SPEED and derive the only prompt the policy can use."""

    source = os.environ if environ is None else environ
    raw_speed = source.get("ORIGAMI_SPEED")
    if raw_speed is None or raw_speed == "":
        raise ValueError("ORIGAMI_SPEED is required and must be exactly 1 or 2.")
    if raw_speed not in {"1", "2"}:
        raise ValueError(f"ORIGAMI_SPEED must be exactly 1 or 2, got {raw_speed!r}.")

    speed = int(raw_speed)
    if speed not in config.phase3.allowed_speeds:
        raise ValueError(
            f"ORIGAMI_SPEED={speed} is not allowed by this bundle; "
            f"allowed values are {config.phase3.allowed_speeds}."
        )
    prompt = config.phase3.prompt_template.format(speed=speed)
    phase3 = dataclasses.replace(config.phase3, selected_speed=speed)
    replay = dataclasses.replace(config.dataset_replay, prompt=prompt)
    configured = dataclasses.replace(config, phase3=phase3, dataset_replay=replay)
    logger.info(
        "Phase-3 speed configured speed=%dx canonical_prompt=%r planner_enabled=%s",
        speed,
        prompt,
        configured.runtime.planner_enabled,
    )
    return configured


def require_canonical_prompt(config: OrigamiCompActionChunkRuntimeConfig) -> str:
    """Return the startup-selected prompt, rejecting unconfigured use."""

    speed = config.phase3.selected_speed
    if speed not in config.phase3.allowed_speeds:
        raise RuntimeError(
            "Phase-3 speed is not configured. Call configure_phase3_speed before "
            "constructing the pipeline."
        )
    return config.phase3.prompt_template.format(speed=int(speed))
