"""Fast, model-free verification for the immutable Phase-3 runtime contract.

This deliberately does not load JAX, the checkpoint, or videos. Run it after
copying the package or building the image to catch configuration drift before a
GPU rollout. Bundle file existence and checkpoint assets are checked separately
by ``verify_comp_action_chunk_bundle.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OPENPI_MODULE = ROOT / "OpenPI_Module"
if str(OPENPI_MODULE) not in sys.path:
    sys.path.insert(0, str(OPENPI_MODULE))

from OpenPI_Module.openpi_runtime.tactile import (  # noqa: E402
    TACTILE_DEFORM_GRID_SHAPE,
    TACTILE_RAW_GRID_SHAPE,
    prepare_tactile_image_inputs,
)
from origami_comp_action_chunk_runtime import load_origami_comp_action_chunk_bundle  # noqa: E402
from origami_comp_action_chunk_runtime.phase3 import configure_phase3_speed  # noqa: E402


PROMPTS = {
    1: "Fold paper into airplane. Speed: 1x.",
    2: "Fold paper into airplane. Speed: 2x.",
}


def _expect_error(label: str, callback) -> None:
    try:
        callback()
    except (TypeError, ValueError, RuntimeError):
        return
    raise AssertionError(f"{label}: expected an error")


def _check_speed_and_prompt(bundle_root: Path) -> None:
    config = load_origami_comp_action_chunk_bundle(bundle_root)
    assert config.runtime.planner_enabled is False
    assert config.server.action_horizon == 25
    assert config.server.action_dim == 65
    assert config.server.require_prompt is False
    for speed, expected_prompt in PROMPTS.items():
        configured = configure_phase3_speed(config, environ={"ORIGAMI_SPEED": str(speed)})
        assert configured.phase3.selected_speed == speed
        assert configured.dataset_replay.prompt == expected_prompt
    _expect_error("missing ORIGAMI_SPEED", lambda: configure_phase3_speed(config, environ={}))
    _expect_error("invalid ORIGAMI_SPEED=0", lambda: configure_phase3_speed(config, environ={"ORIGAMI_SPEED": "0"}))
    _expect_error("invalid ORIGAMI_SPEED=3", lambda: configure_phase3_speed(config, environ={"ORIGAMI_SPEED": "3"}))
    _expect_error("invalid ORIGAMI_SPEED=2x", lambda: configure_phase3_speed(config, environ={"ORIGAMI_SPEED": "2x"}))


def _check_raw_tactile_policy() -> None:
    deform = np.zeros(TACTILE_DEFORM_GRID_SHAPE, dtype=np.uint8)

    missing = prepare_tactile_image_inputs(tactile_deform_grid=deform)
    assert missing.raw_available is False and missing.raw_status == "missing"

    all_zero = prepare_tactile_image_inputs(
        tactile_deform_grid=deform,
        tactile_raw_grid=np.zeros(TACTILE_RAW_GRID_SHAPE, dtype=np.uint8),
    )
    assert all_zero.raw_available is False and all_zero.raw_status == "all_zero"

    # Only one pixel is nonzero: this proves that a valid *complete* raw grid
    # is retained even when many individual tactile tiles are all zero.
    partial_nonzero = np.zeros(TACTILE_RAW_GRID_SHAPE, dtype=np.uint8)
    partial_nonzero[0, 0, 0] = 1
    provided = prepare_tactile_image_inputs(
        tactile_deform_grid=deform,
        tactile_raw_grid=partial_nonzero,
    )
    assert provided.raw_available is True and provided.raw_status == "provided_grid"
    assert provided.raw_images.shape == (10, 3, 224, 224)

    invalid = prepare_tactile_image_inputs(
        tactile_deform_grid=deform,
        tactile_raw_grid=np.zeros((1, 1, 3), dtype=np.uint8),
        tolerate_invalid_optional_raw=True,
    )
    assert invalid.raw_available is False and invalid.raw_status == "invalid"

    _expect_error(
        "missing required deform tactile",
        lambda: prepare_tactile_image_inputs(
            tactile_raw_grid=np.ones(TACTILE_RAW_GRID_SHAPE, dtype=np.uint8)
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=ROOT / "model_bundle")
    args = parser.parse_args()
    _check_speed_and_prompt(args.bundle_root.resolve())
    _check_raw_tactile_policy()
    print(
        "OK: Phase-3 contract accepts only ORIGAMI_SPEED=1/2, fixes the matching "
        "training prompt, keeps planner masked, and applies complete-grid raw tactile fallback."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
