# Large-Scale Policy Training

Date: 2026-05-13

## Summary

This run trained and validated an 8x8 generated-map policy with a clear advantage over the random opponent. The final selected checkpoint is:

```text
/tmp/generals-bc-8x8-soft-v3.eqx
```

The checkpoint is intentionally stored under `/tmp` because `.eqx` files are ignored by the repository and should not be committed.

## Goal

The goal was not just to run a smoke test. The target was a policy with a significant measured advantage on larger generated maps with mountains and cities.

The evaluation distribution was:

- 8x8 generated maps
- mountain density: 0.12-0.22
- cities: 4-8
- minimum general distance: 5
- opponent: random valid-action policy
- execution mode: sampled policy

## Baselines

Before training, I measured the environment baselines on the same map distribution:

```text
Random vs Random, 512 games, 250 steps:
  wins/losses/draws = 8/8/496
  decisive win rate = 50.0%
  draw rate = 96.9%

Randomized Expander vs Random, 512 games, 250 steps:
  wins/losses/draws = 315/1/196
  decisive win rate = 99.7%
  draw rate = 38.3%

Random vs Randomized Expander, 512 games, 250 steps:
  wins/losses/draws = 1/305/206
  decisive win rate = 0.3%
  draw rate = 40.2%
```

The baseline showed that random play produces too many draws and that the randomized Expander heuristic is a strong teacher against Random.

## Implementation Changes

### `examples/_experimental/ppo/network.py`

- Added `PolicyValueNetwork.logits_value(obs, mask)`.
- This exposes masked flattened action logits and value predictions without changing module fields.
- Existing `.eqx` serialization compatibility is preserved because only methods changed, not Equinox module fields.

### `examples/_experimental/ppo/common.py`

Added shared experimental helpers:

- simple/generated grid batching
- reset-pool creation
- action index encoding/decoding
- action normalization
- greedy policy action selection
- sampled policy action selection
- Expander soft-target distribution construction
- standalone `obs_to_array`

### `examples/_experimental/ppo/behavior_clone.py`

Added a behavior-cloning trainer that:

- rolls out teacher-vs-random games on generated maps
- uses the randomized Expander scoring distribution as a soft target
- trains the policy with cross entropy against that target distribution
- saves Equinox checkpoints to a configurable path

### `examples/_experimental/ppo/evaluate_policy.py`

Added a batch evaluator that:

- loads a saved `.eqx` policy checkpoint
- evaluates on independent generated maps
- supports Random or Expander opponents
- supports greedy or sampled policy execution
- reports wins, losses, draws, win rate, decisive win rate, draw rate, mean final time, and runtime

### `generals/agents/_expander_logic.py`

Added `expander_greedy_action` during experimentation. It was useful diagnostically but was not the final teacher because it produced much lower overall win rate than the existing randomized Expander.

## Training Runs

### Hard-Label Randomized Expander BC

Command shape:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 512 \
  --grid-size 8 \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 200 \
  --lr 0.001 \
  --model-path /tmp/generals-bc-8x8-v1.eqx
```

Result:

- training accuracy reached about 33%
- 1024-game evaluation at 500 steps reached 39.8% total win rate
- decisive win rate was high, but draw rate remained too high

This was not accepted as the final model.

### Soft-Target Expander BC, v2

Command shape:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --pool-size 2048 \
  --num-steps 32 \
  --num-iterations 800 \
  --lr 0.001 \
  --model-path /tmp/generals-bc-8x8-soft-v2.eqx \
  --seed 45
```

Result:

```text
1024 games, 500 steps, sampled policy vs Random:
  wins/losses/draws = 881/16/127
  win rate = 86.0%
  decisive win rate = 98.2%
  draw rate = 12.4%
```

This was strong but still below the later v3 run.

### Soft-Target Expander BC, v3

Final selected command:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 2000 \
  --lr 0.0007 \
  --model-path /tmp/generals-bc-8x8-soft-v3.eqx \
  --seed 46
```

Result:

- final checkpoint: `/tmp/generals-bc-8x8-soft-v3.eqx`
- final training loss fluctuated around 1.5-1.7
- sampled execution was much stronger than greedy execution

## Final Validation

Independent evaluations on unseen generated maps:

```text
2048 games, 500 steps, seed 4001, sampled policy vs Random:
  wins/losses/draws = 1860/28/160
  win rate = 90.8%
  decisive win rate = 98.5%
  draw rate = 7.8%

2048 games, 500 steps, seed 5001, sampled policy vs Random:
  wins/losses/draws = 1887/21/140
  win rate = 92.1%
  decisive win rate = 98.9%
  draw rate = 6.8%

2048 games, 500 steps, seed 5002, sampled policy vs Random:
  wins/losses/draws = 1905/16/127
  win rate = 93.0%
  decisive win rate = 99.2%
  draw rate = 6.2%

2048 games, 250 steps, seed 5003, sampled policy vs Random:
  wins/losses/draws = 1219/18/811
  win rate = 59.5%
  decisive win rate = 98.5%
  draw rate = 39.6%
```

The 500-step evaluations show a stable and significant advantage. The 250-step evaluation still has many draws, but among decisive games the policy wins overwhelmingly.

A diagnostic evaluation against Expander showed the trained network is not stronger than the heuristic teacher:

```text
1024 games, 500 steps, sampled policy vs Randomized Expander:
  wins/losses/draws = 430/501/93
  win rate = 42.0%
  decisive win rate = 46.2%
```

## Conclusion

The final v3 policy has a significant advantage over Random on larger generated maps. It is not yet stronger than the Expander heuristic, so the result should be described as a strong Random-opponent policy, not a generally strong Generals.io agent.

Recommended next step: use `/tmp/generals-bc-8x8-soft-v3.eqx` as the warm-start for PPO or self-play, and use `evaluate_policy.py` as the acceptance gate.
