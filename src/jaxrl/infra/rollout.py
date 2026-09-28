"""Collecting trajectories.

Episodes are fixed length. That is what lets a rollout be a lax.scan and a
batch of rollouts be a vmap -- early termination would make them ragged, and
ragged arrays force a Python loop.
"""

from typing import NamedTuple

import jax


class Trajectory(NamedTuple):
    """One episode, or a batch of them with a leading axis.

    `obs[t]` is the state the policy acted on and `action[t]` is what it did
    there, so the pair is aligned for supervised learning.
    """

    obs: jax.Array  # (horizon, 4)
    action: jax.Array  # (horizon, 1)

    def flatten(self):
        """Supervised pairs: (n, horizon, d) -> (n*horizon, d)."""
        return (self.obs.reshape(-1, self.obs.shape[-1]),
                self.action.reshape(-1, self.action.shape[-1]))


def sample_trajectory(key, policy_fn, env, horizon) -> Trajectory:
    """Roll out one episode of exactly `horizon` steps.

    `policy_fn(key, state) -> action` is the only interface a policy needs, so
    the expert and a learned policy go through this unchanged -- which is what
    makes the DAgger loop a two-line change.

    The key is split into `horizon + 1`: one for the reset, one per step. The
    per-step keys are the `xs` of the scan, so a stochastic policy gets fresh
    randomness without the carry having to hold a key.
    """
    keys = jax.random.split(key, horizon + 1)
    state0 = env.reset(keys[0])

    def body(state, key):  # carry, one x
        action = policy_fn(key, state)
        return env.step(state, action), (state, action)
        #      ^ next carry              ^ recorded this step

    final_state, (obs, act) = jax.lax.scan(body, state0, keys[1:])
    return Trajectory(obs=obs, action=act)


def sample_trajectories(key, policy_fn, env, n, horizon):
    """`n` independent episodes, returned with a leading batch axis.

    vmap over the keys rather than a loop over episodes: `obs` comes back as
    (n, horizon, 4). Only the key varies, hence in_axes=(0, None, None, None).
    """
    keys = jax.random.split(key, n)
    return jax.vmap(sample_trajectory,
                    in_axes=(0, None, None, None))(keys, policy_fn, env,
                                                   horizon)
