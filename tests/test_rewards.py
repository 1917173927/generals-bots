import jax.numpy as jnp

from generals.core import game
from generals.core.rewards import general_target_reward_fn


def make_general_target_state(player_cell):
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2)
    state = game.create_initial_state(grid)
    row, col = player_cell
    state = state._replace(
        armies=state.armies.at[0, 0].set(1).at[row, col].set(5),
        ownership=state.ownership.at[0, row, col].set(True),
        ownership_neutral=state.ownership_neutral.at[row, col].set(False),
    )
    return state


def test_general_target_reward_is_positive_when_strong_cell_gets_closer_to_enemy_general():
    prior = make_general_target_state((0, 0))
    current = make_general_target_state((1, 0))

    reward = general_target_reward_fn(
        prior,
        current,
        player=0,
        scale=1.0,
        max_distance=8,
        min_army=2,
    )

    assert reward > 0.0


def test_general_target_reward_is_negative_when_strong_cell_gets_farther_from_enemy_general():
    prior = make_general_target_state((1, 0))
    current = make_general_target_state((0, 0))

    reward = general_target_reward_fn(
        prior,
        current,
        player=0,
        scale=1.0,
        max_distance=8,
        min_army=2,
    )

    assert reward < 0.0


def test_general_target_reward_scale_zero_disables_shaping():
    prior = make_general_target_state((0, 0))
    current = make_general_target_state((1, 0))

    reward = general_target_reward_fn(
        prior,
        current,
        player=0,
        scale=0.0,
        max_distance=8,
        min_army=2,
    )

    assert reward == 0.0
