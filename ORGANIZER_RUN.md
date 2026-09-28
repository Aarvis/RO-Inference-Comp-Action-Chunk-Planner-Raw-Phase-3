# Phase-3 Docker run guide

The image implements the existing `origami-zenoh-v1` contract from
`inferencekit_09_02_2026`. It requires an explicit `ORIGAMI_SPEED` for every
policy/replay process.

## Build and verify

```bash
cd RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-3

python scripts/verify_phase3_runtime_contract.py
python scripts/verify_comp_action_chunk_bundle.py --bundle-root model_bundle

docker build \
  -f docker/submission-zenoh-bundled.Dockerfile \
  -t ro-inference-comp-action-chunk-planner-raw-phase3:latest .

IMAGE='ro-inference-comp-action-chunk-planner-raw-phase3:latest'

docker run --rm \
  -e ORIGAMI_SPEED=1 \
  "$IMAGE" verify-bundle
```

`ORIGAMI_SPEED` must be exactly `1` or `2`. Missing, `0`, `3`, `1x`, and all
other values fail before the model is loaded. `ORIGAMI_PLANNER_ENABLED=true`
also fails: this endpoint is deliberately planner-free.

## Run the policy server

```bash
IMAGE='ro-inference-comp-action-chunk-planner-raw-phase3:latest'
SESSION='phase3-contract-test'
ROUTER_IMAGE='eclipse/zenoh@sha256:157965d71e0bfd0a044d76a985ff0e5c306ad3968929168fb9678cd2a7fec23f'

docker network create origami-phase3-test 2>/dev/null || true

docker run -d --name origami-phase3-router \
  --network origami-phase3-test \
  -p 127.0.0.1:17447:7447 \
  "$ROUTER_IMAGE" \
  -l tcp/0.0.0.0:7447 \
  --no-multicast-scouting \
  --cfg 'transport/shared_memory/enabled:false'

docker run -d --name origami-phase3-policy \
  --network origami-phase3-test \
  --gpus all \
  --shm-size 8g \
  -e ORIGAMI_ZENOH_ENDPOINT='tcp/origami-phase3-router:7447' \
  -e ORIGAMI_SESSION_ID="$SESSION" \
  -e ORIGAMI_SPEED=1 \
  -e EXECUTION_MODE=sync \
  -e ORIGAMI_WARMUP_INFERENCES=1 \
  -e ORIGAMI_OPENPI_PARAM_DTYPE=checkpoint \
  -e ORIGAMI_JAX_MEM_FRACTION=0.70 \
  "$IMAGE" serve

docker logs -f origami-phase3-policy
```

The startup log must state `speed=1`, the exact canonical prompt, and
`planner_enabled=false`, followed by `READY transport=origami-zenoh-v1`.

To run the same fixed policy with temporal aggregation, change only:

```bash
-e EXECUTION_MODE=async
```

## Validate with the inference kit

```bash
cd /path/to/inferencekit_09_02_2026/sharpa_north_ces_lite_sdk-main

uv run --no-sync python ./examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id "$SESSION" \
  --timeout 300 \
  --requests 300 \
  --expected-horizon 25
```

## Dataset replay

The same `ORIGAMI_SPEED` controls replay prompts. Client/replay YAML prompts do
not override it.

```bash
mkdir -p "$PWD/dataset_replay_phase3_speed1"

docker run --rm --gpus all \
  --shm-size 8g \
  --user "$(id -u):$(id -g)" \
  -e ORIGAMI_SPEED=1 \
  -e ORIGAMI_OPENPI_PARAM_DTYPE=checkpoint \
  -e ORIGAMI_WARMUP_INFERENCES=1 \
  -v "$PWD/dataset_replay_phase3_speed1:/output" \
  "$IMAGE" dataset-replay \
  --max-episodes 1 \
  --max-total-samples 8000 \
  --max-loss-samples 800 \
  --loss-every-n-samples 1 \
  --tactile-modes deform_only deform_plus_raw \
  --output-dir /output
```

The per-request/replay logs report `raw_present`, `raw_available`, and
`raw_status`. Expect `raw_status=provided_grid` when at least one pixel in the
complete raw tactile grid is non-zero; `all_zero`, `missing`, and `invalid`
run deform-only.

## Cleanup

```bash
docker rm -f origami-phase3-policy origami-phase3-router
docker network rm origami-phase3-test
```
