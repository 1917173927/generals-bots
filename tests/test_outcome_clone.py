import jax.numpy as jnp

from examples._experimental.ppo.outcome_clone import choose_loser_trajectories, choose_winner_trajectories


def test_choose_winner_trajectories_uses_winner_perspective_and_skips_draws():
    obs_p0 = jnp.full((2, 3, 1, 1, 1), 10.0)
    obs_p1 = jnp.full((2, 3, 1, 1, 1), 20.0)
    masks_p0 = jnp.full((2, 3, 1, 1, 1), True)
    masks_p1 = jnp.full((2, 3, 1, 1, 1), False)
    actions_p0 = jnp.full((2, 3, 5), 1, dtype=jnp.int32)
    actions_p1 = jnp.full((2, 3, 5), 2, dtype=jnp.int32)
    active = jnp.array([[True, True, True], [False, True, True]])
    winners = jnp.array([0, 1, -1], dtype=jnp.int32)

    obs, masks, actions, weights = choose_winner_trajectories(
        obs_p0,
        obs_p1,
        masks_p0,
        masks_p1,
        actions_p0,
        actions_p1,
        active,
        winners,
    )

    assert jnp.array_equal(obs[:, 0], obs_p0[:, 0])
    assert jnp.array_equal(obs[:, 1], obs_p1[:, 1])
    assert jnp.array_equal(actions[:, 0], actions_p0[:, 0])
    assert jnp.array_equal(actions[:, 1], actions_p1[:, 1])
    assert jnp.array_equal(masks[:, 0], masks_p0[:, 0])
    assert jnp.array_equal(masks[:, 1], masks_p1[:, 1])
    assert jnp.array_equal(weights, jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32))


def test_choose_winner_trajectories_can_keep_only_learner_wins():
    obs_p0 = jnp.zeros((2, 3, 1, 1, 1))
    obs_p1 = jnp.ones((2, 3, 1, 1, 1))
    masks_p0 = jnp.full((2, 3, 1, 1, 1), True)
    masks_p1 = jnp.full((2, 3, 1, 1, 1), True)
    actions_p0 = jnp.full((2, 3, 5), 1, dtype=jnp.int32)
    actions_p1 = jnp.full((2, 3, 5), 2, dtype=jnp.int32)
    active = jnp.array([[True, True, True], [True, True, True]])
    winners = jnp.array([0, 1, -1], dtype=jnp.int32)

    _, _, _, weights = choose_winner_trajectories(
        obs_p0,
        obs_p1,
        masks_p0,
        masks_p1,
        actions_p0,
        actions_p1,
        active,
        winners,
        learner_player=0,
        sample_source=1,
    )

    assert jnp.array_equal(weights, jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32))


def test_choose_loser_trajectories_uses_opposite_decisive_perspective():
    obs_p0 = jnp.full((2, 3, 1, 1, 1), 10.0)
    obs_p1 = jnp.full((2, 3, 1, 1, 1), 20.0)
    masks_p0 = jnp.full((2, 3, 1, 1, 1), True)
    masks_p1 = jnp.full((2, 3, 1, 1, 1), False)
    actions_p0 = jnp.full((2, 3, 5), 1, dtype=jnp.int32)
    actions_p1 = jnp.full((2, 3, 5), 2, dtype=jnp.int32)
    active = jnp.array([[True, True, True], [True, False, True]])
    winners = jnp.array([0, 1, -1], dtype=jnp.int32)

    obs, masks, actions, weights = choose_loser_trajectories(
        obs_p0,
        obs_p1,
        masks_p0,
        masks_p1,
        actions_p0,
        actions_p1,
        active,
        winners,
    )

    assert jnp.array_equal(obs[:, 0], obs_p1[:, 0])
    assert jnp.array_equal(obs[:, 1], obs_p0[:, 1])
    assert jnp.array_equal(actions[:, 0], actions_p1[:, 0])
    assert jnp.array_equal(actions[:, 1], actions_p0[:, 1])
    assert jnp.array_equal(masks[:, 0], masks_p1[:, 0])
    assert jnp.array_equal(masks[:, 1], masks_p0[:, 1])
    assert jnp.array_equal(weights, jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32))
