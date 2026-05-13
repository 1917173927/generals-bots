# Larger Map PPO Support

Date: 2026-05-13

## Summary

The experimental PPO tools now support training and visualization on larger square maps. The raw PPO trainer remains backward-compatible with the previous 4x4 simple-map default, but it can now opt into generated terrain with mountains, cities, configurable general spacing, city armies, truncation, and reset-pool sizing.

## Problem

The training path was still shaped around the original 4x4 smoke-test setup:

- The policy network could technically be constructed for a larger `grid_size`, but the raw trainer always initialized 4x4 states.
- Auto-reset regenerated hardcoded 4x4 empty maps inside the rollout step.
- The visualization tool loaded every model as a 4x4 model and rendered only hardcoded 4x4 maps.
- The map generator had an old fixed extra-city cap and a mountain-placement shape bound that prevented larger generated maps from honoring higher terrain settings.

That made 4x4 useful as a smoke test, but too small for meaningful Generals.io training experiments.

## Changes

### Raw PPO Trainer

File: `examples/_experimental/ppo/train.py`

Added command-line controls:

- `--grid-size`
- `--truncation`
- `--pool-size`
- `--map-generator simple|generated`
- `--mountain-density-min`
- `--mountain-density-max`
- `--num-cities-min`
- `--num-cities-max`
- `--min-generals-distance`
- `--max-generals-distance`
- `--city-army-min`
- `--city-army-max`

The raw trainer now pre-generates a reusable pool of initial states. For `simple`, the pool contains empty maps with two random generals. For `generated`, the pool comes from `generals.core.grid.generate_grid` and includes the configured terrain. Rollout auto-reset indexes into this pool instead of creating 4x4 maps inside the JIT step.

### GeneralsEnv PPO Trainer

File: `examples/_experimental/ppo/train2.py`

Added matching larger-map and terrain CLI controls for the wrapper-based PPO path. This keeps the secondary trainer compatible with the same generated-map settings while still using `GeneralsEnv` for reset-pool behavior.

### Visualization

File: `examples/_experimental/visualize_policy.py`

The visualizer now accepts the same map-size and generated-terrain arguments. It constructs `PolicyValueNetwork` with the requested `--grid-size`, so larger saved models can be deserialized and rendered.

### Map Generation

File: `generals/core/grid.py`

The generator now derives static mountain and city placement capacities from the configured ranges rather than from the old fixed limits. This lets larger maps use more cities and denser mountains when requested. Fallback castle placement also avoids overwriting generals when local reachable cells are exhausted.

### Documentation and Tests

Files:

- `examples/_experimental/README.md`
- `tests/test_grid_generation_performance.py`

The experimental README now includes larger-map training and visualization commands. The grid generation tests include a 12x12 generated-map case with custom mountain and city settings.

## Example Commands

Simple larger-map smoke:

```bash
uv run python examples/_experimental/ppo/train.py 64 --grid-size 8 --num-steps 64 --num-iterations 10 --model-path /tmp/generals-ppo-8x8-simple.eqx
```

Generated terrain:

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --num-steps 64 \
  --num-iterations 10 \
  --pool-size 512 \
  --model-path /tmp/generals-ppo-8x8-generated.eqx
```

Visualize the generated-map model:

```bash
uv run python examples/_experimental/visualize_policy.py /tmp/generals-ppo-8x8-generated.eqx 10 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5
```

## Notes

4x4 remains the default because it is a cheap smoke-test target. It should not be treated as a meaningful strategic benchmark. For training experiments, use at least 8x8 generated maps, then scale up after checking throughput and memory.

## Verification Results

Validated on 2026-05-13:

- `uv run python -m compileall examples/_experimental/ppo examples/_experimental/visualize_policy.py generals/core/grid.py tests/test_grid_generation_performance.py`
- `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python - <<'PY' ... print(jax.default_backend(), jax.devices()) ... PY`
- `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/_experimental/ppo/train.py 2 --num-steps 2 --num-iterations 1 --pool-size 8 --model-path /tmp/generals-ppo-simple-smoke.eqx`
- `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/_experimental/ppo/train.py 4 --grid-size 8 --map-generator generated --pool-size 16 --num-steps 2 --num-iterations 1 --truncation 100 --num-cities-min 4 --num-cities-max 8 --mountain-density-min 0.12 --mountain-density-max 0.22 --min-generals-distance 5 --model-path /tmp/generals-ppo-generated-8x8-smoke.eqx`
- `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/_experimental/ppo/train2.py 4 --grid-size 8 --pool-size 16 --num-steps 2 --num-iterations 1 --truncation 100 --num-cities-min 4 --num-cities-max 8 --mountain-density-min 0.12 --mountain-density-max 0.22 --min-generals-distance 5 --model-path /tmp/generals-ppo-env-generated-8x8-smoke.eqx`
- `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python - <<'PY' ... load /tmp/generals-ppo-generated-8x8-smoke.eqx and run one forward pass ... PY`
- `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest tests/test_grid_generation_performance.py`
- `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest`
- `git diff --check`

Results:

- JAX reported `gpu` and `[CudaDevice(id=0)]`.
- The raw trainer completed both 4x4/simple and 8x8/generated smoke runs on `cuda:0`.
- The GeneralsEnv trainer completed the 8x8/generated smoke run on `cuda:0`.
- The saved 8x8 generated-map model deserialized successfully and produced finite value, log-probability, and entropy outputs on a generated 8x8 map.
- `tests/test_grid_generation_performance.py` passed with 6 tests.
- The full suite passed with 14 tests.

Known environment note: the CUDA runtime still prints a non-fatal kernel driver version parse warning, but JAX selects the GPU backend and the training smoke tests execute on `cuda:0`.
