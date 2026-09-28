# Phase-3 bundle setup

Populate only the policy assets required by the planner-free endpoint:

```bash
mkdir -p model_bundle/assets
mkdir -p model_bundle/weights/openpi

# Copy/download the tokenizer used by the Phase-3 OpenPI checkpoint.
cp /path/to/paligemma_tokenizer.model model_bundle/assets/paligemma_tokenizer.model

# Copy the final checkpoint directory, including params/ and assets/.
cp -a /path/to/final_phase3_checkpoint \
  model_bundle/weights/openpi/pi05_origami_comp_action_chunk_phase3
```

The copied checkpoint must contain:

```text
params/
assets/competition_paper_reprocessed_origami_comp_action_chunk_phase3/norm_stats.json
```

No DINO, OOI, or checkpoint-planner model files are needed for Phase-3.
