# PPO Training Repair

Date: 2026-05-13

## Summary

The experimental PPO implementation can now be used for short training runs again. The primary supported entry point is `examples/_experimental/ppo/train.py`, which uses the raw JAX game API. The secondary `examples/_experimental/ppo/train2.py` wrapper path was also repaired so it matches the current `GeneralsEnv` API.

This devlog is intentionally stored as a standalone Markdown file. The training scripts do not write devlogs or modify documentation at runtime.

## Problem Analysis

The PPO code had three classes of issues:

1. Missing project dependencies
   - The experimental training code imports `equinox` and `optax`.
   - Those packages were not declared in `pyproject.toml` or `requirements.txt`, so a fresh project environment could not run the PPO trainer reliably.

2. Incorrect PPO target construction
   - The trainer computed GAE advantages and then normalized them.
   - It then built value returns from the normalized advantages with `returns = advantages + values`.
   - That is mathematically wrong for PPO value learning because the critic target should be based on raw GAE returns, while normalization should affect only the policy advantage term.

3. Broken secondary and visualization entry points
   - `train2.py` called `GeneralsEnv.step` without the required `pool` argument.
   - `train2.py` used post-auto-reset observations for shaped rewards, which can mix the terminal transition with the reset state.
   - `visualize_policy.py` had a stale fallback import path: `examples.ppo.network` does not exist in this repository layout.
   - Random action sampling used a fixed `size=100` in several places, which is incorrect for arbitrary board sizes and unsafe when there are no legal moves.

4. Core game timing regression exposed by training validation
   - The targeted JAX game tests showed that `global_update` was adding army growth too early.
   - Passing on the first move changed army counts because the step advanced from time 0 to time 1 and then applied structure growth.
   - Directly calling `global_update` at time 2 did not grow generals/cities even though the tests expect that timing.
   - Full-board growth also needed to avoid firing at time 0.

## Changes

### Dependency Declarations

Added PPO training dependencies to both project dependency files:

- `equinox>=0.13.8`
- `optax>=0.2.8`

The Equinox and Optax training patterns were checked against upstream documentation before changing optimizer and serialization calls. The repaired path uses filtered floating-point array leaves for optimizer initialization and updates, then applies updates with `eqx.apply_updates`.

### Raw PPO Trainer

File: `examples/_experimental/ppo/train.py`

The raw-game trainer is now the main usable training script.

Fixes:

- Added command-line arguments for:
  - `num_envs`
  - `--num-steps`
  - `--num-iterations`
  - `--lr`
  - `--model-path`
- Added local path bootstrapping so the script can be run directly from the repository root.
- Replaced the fixed random-action candidate buffer with `size=mask.size`.
- Added a safe fallback pass action when no legal move exists.
- Reworked `compute_gae` to return both raw advantages and raw returns.
- Normalized only the policy advantages.
- Kept raw returns for the value loss target.
- Switched optimizer initialization and update parameters to `eqx.filter(network, eqx.is_inexact_array)`.
- Saved the trained model to the configurable `--model-path`.

Recommended smoke command:

```bash
uv run python examples/_experimental/ppo/train.py 2 --num-steps 2 --num-iterations 1 --model-path /tmp/generals-ppo-smoke.eqx
```

The default training command remains intentionally small enough for experimental iteration:

```bash
uv run python examples/_experimental/ppo/train.py
```

For a longer run:

```bash
uv run python examples/_experimental/ppo/train.py 128 --num-steps 128 --num-iterations 50 --model-path jax_ppo_model.eqx
```

### GeneralsEnv PPO Trainer

File: `examples/_experimental/ppo/train2.py`

This script is still secondary to the raw-game trainer, but it no longer fails on the current environment API.

Fixes:

- Added direct-run path bootstrapping.
- Added the same training CLI controls as the raw trainer, plus:
  - `--num-epochs`
  - `--minibatch-size`
  - `--pool-size`
- Generated and retained the environment reset pool with `pool, _ = env.reset(pool_key)`.
- Passed the pool explicitly through every `env.step` call.
- Computed shaped reward observations from `timesteps.last_state`, which represents the transition result before auto-reset.
- Repaired GAE return handling so value targets use raw returns.
- Normalized only policy advantages.
- Used filtered floating-point Equinox leaves for Optax initialization and updates.

Recommended smoke command:

```bash
uv run python examples/_experimental/ppo/train2.py 2 --num-steps 2 --num-iterations 1 --pool-size 16 --model-path /tmp/generals-ppo-env-smoke.eqx
```

### Policy Visualization

File: `examples/_experimental/visualize_policy.py`

Fixes:

- Added path bootstrapping for direct execution.
- Replaced the stale `examples.ppo.network` fallback with a direct import from the local PPO directory.
- Repaired random-action sampling with dynamic mask sizing and a safe no-move fallback.

Usage:

```bash
uv run python examples/_experimental/visualize_policy.py jax_ppo_model.eqx 10
```

### Core Game Timing

File: `generals/core/game.py`

Fixes:

- Changed full-board army growth to require `time > 0` and `time % 50 == 0`.
- Changed generals/cities growth to require `time > 0` and `time % 2 == 0`.
- This matches the existing JAX game tests and avoids mutating armies on a pure first-turn pass.

## Can It Be Used For Training?

Yes, the repaired PPO path can now be used for experimental training. The raw trainer can execute a rollout, compute PPO losses, update the Equinox model with Optax, and serialize the resulting model.

However, this is still an experimental baseline rather than a finished strong Generals.io agent. The current setup has important limitations:

- The network and trainer are oriented around a 4x4 grid.
- The opponent is a random policy, not self-play or a curriculum.
- The raw trainer uses a simple random reset over general positions.
- PPO minibatching, multi-epoch updates, checkpointing, evaluation, and curriculum scheduling are still minimal.
- The pass action is represented as one policy channel spread over board cells, which works mechanically but is not an ideal action-space design.
- No long-run convergence claim has been established.

The practical conclusion is:

- Use `train.py` for short experimental PPO training and smoke validation.
- Treat produced models as research artifacts, not production-ready bots.
- Add evaluation curves and longer repeated runs before claiming learning quality.

## Verification Plan

The repair should be validated with:

```bash
uv run python -m compileall examples/_experimental/ppo examples/_experimental/visualize_policy.py
uv run python examples/_experimental/ppo/train.py 2 --num-steps 2 --num-iterations 1 --model-path /tmp/generals-ppo-smoke.eqx
uv run python examples/_experimental/ppo/train2.py 2 --num-steps 2 --num-iterations 1 --pool-size 16 --model-path /tmp/generals-ppo-env-smoke.eqx
uv run pytest tests/test_game_jax.py
```

Known environment note: on this machine JAX may print CUDA plugin or driver warnings and still fall back to CPU execution. Those warnings do not by themselves invalidate the CPU smoke tests.

## Verification Results

Validated on 2026-05-13:

- `uv run python -m compileall examples/_experimental/ppo examples/_experimental/visualize_policy.py`
- `uv run python examples/_experimental/ppo/train.py 2 --num-steps 2 --num-iterations 1 --model-path /tmp/generals-ppo-smoke.eqx`
- `uv run python examples/_experimental/ppo/train2.py 2 --num-steps 2 --num-iterations 1 --pool-size 16 --model-path /tmp/generals-ppo-env-smoke.eqx`
- `uv run pytest tests/test_game_jax.py`

The first target-test run exposed the core army-growth timing issue described above. After the `generals/core/game.py` fix, the targeted JAX game suite passed.

## Follow-Up Work

Recommended next steps:

1. Add a small automated PPO smoke test that avoids saving models inside the repository.
2. Replace the pass-action encoding with a cleaner single pass logit.
3. Add evaluation episodes against RandomAgent and ExpanderAgent after each checkpoint.
4. Add seed-controlled benchmark runs before making claims about policy quality.
5. Decide whether the experimental benchmark script should be updated to the current `GeneralsEnv.step(state, actions, pool)` API in a separate cleanup.
