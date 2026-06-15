import jax.numpy as jnp

from examples._experimental.ppo.search_policy import score_observation
from generals.core.game import GameInfo
from generals.core.observation import Observation


def _observation(owned_army, opponent_army, owned_land, opponent_land):
    board = jnp.zeros((4, 4), dtype=jnp.int32)
    mask = jnp.zeros((4, 4), dtype=bool)
    return Observation(
        armies=board,
        generals=mask,
        cities=mask,
        mountains=mask,
        neutral_cells=mask,
        owned_cells=mask,
        opponent_cells=mask,
        fog_cells=mask,
        structures_in_fog=mask,
        owned_land_count=jnp.int32(owned_land),
        owned_army_count=jnp.int32(owned_army),
        opponent_land_count=jnp.int32(opponent_land),
        opponent_army_count=jnp.int32(opponent_army),
        timestep=jnp.int32(0),
    )


def _info(winner):
    return GameInfo(
        army=jnp.zeros((2,), dtype=jnp.int32),
        land=jnp.zeros((2,), dtype=jnp.int32),
        is_done=winner >= 0,
        winner=jnp.int32(winner),
        time=jnp.int32(10),
    )


def test_score_observation_prefers_wins_and_material_advantage():
    neutral = score_observation(_info(-1), _observation(20, 20, 8, 8), player=0)
    material = score_observation(_info(-1), _observation(40, 20, 12, 6), player=0)
    win = score_observation(_info(0), _observation(5, 40, 4, 20), player=0)
    loss = score_observation(_info(1), _observation(40, 5, 20, 4), player=0)

    assert material > neutral
    assert win > material
    assert loss < neutral
