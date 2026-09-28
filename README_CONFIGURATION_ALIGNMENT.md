# Phase-3 configuration alignment

| Layer | Required value |
| --- | --- |
| Registered OpenPI config | `pi05_origami_comp_action_chunk_phase3` |
| Action horizon | `25` |
| Action target stride | `1` |
| State/action dimension | `65` |
| Tactile prompt dimension | `60` |
| Camera/tactile crop size | `224`, bilinear resize |
| Planner | hard-disabled, zero masked prefix |
| Prompt | `Fold paper into airplane. Speed: {speed}x.` |
| Valid runtime speed | `ORIGAMI_SPEED=1` or `ORIGAMI_SPEED=2` |
| Raw tactile | all-grid optional; all-zero means unavailable |
| OpenPI assets | final Phase-3 `norm_stats.json` in checkpoint assets |

The bundle, its OpenPI snapshot, and its vendored OpenPI source must move
together with the Phase-3 checkpoint. Do not use a Phase-2 source snapshot or
normalization asset with this endpoint.
