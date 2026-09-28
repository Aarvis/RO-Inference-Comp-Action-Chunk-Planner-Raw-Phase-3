# Populate this Phase-3 model bundle

Before image build, place the selected final checkpoint at:

```text
weights/openpi/pi05_origami_comp_action_chunk_phase3/
```

It must contain both `params/` and:

```text
assets/competition_paper_reprocessed_origami_comp_action_chunk_phase3/norm_stats.json
```

Also provide `assets/paligemma_tokenizer.model`. Run the two verification
scripts from the project README before building the Docker image.
