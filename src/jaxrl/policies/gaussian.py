"""A diagonal Gaussian policy over continuous actions.

One network emits both the mean and the log standard deviation: the last layer
is 2 * ac_dim wide and gets split. Sharing the trunk means the features that
predict what to do also predict how confident to be, which is usually the same
information.
"""

from typing import cast

import jax
import jax.numpy as jnp

from ..nn import MLP


class GaussianPolicy:
    """Holds the architecture only; params are passed to every method."""

    def __init__(self, ob_dim, ac_dim, hidden, obs_mean=None, obs_std=None):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.obs_mean = jnp.zeros(ob_dim) if obs_mean is None else obs_mean
        self.obs_std = jnp.ones(ob_dim) if obs_std is None else obs_std
        self.model = MLP(features=tuple(hidden) + (2 * ac_dim, ))

    def init(self, key):
        """Random starting weights. The zeros array is only there for its
        shape -- Flax infers each layer's input width by tracing with it."""
        return self.model.init(key, jnp.zeros((self.ob_dim, )))

    def forward(self, params, obs):
        """(mu, log_std) for one obs (ob_dim,) or a batch (N, ob_dim).

        log_std is clipped because the target here is a deterministic expert:
        a perfect fit would send sigma to zero and the log-likelihood to
        infinity, so the loss has no floor to converge to.
        """
        obs = (obs - self.obs_mean) / self.obs_std
        out = cast(jax.Array, self.model.apply(params, obs))
        mu, log_std = jnp.split(out, 2, axis=-1)
        log_std = jnp.clip(log_std, -5.0, 2.0)
        return mu, log_std

    def log_prob(self, params, obs, acts):
        """Log-density of `acts`, summed over action dims -> (batch,).

        Summed, not averaged: the covariance is diagonal, so the joint
        log-density of the action vector is the sum over its components.
        """
        mu, log_std = self.forward(params, obs)
        sigma = jnp.exp(log_std)
        log_p = -0.5 * (
            (acts - mu) / sigma)**2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
        return log_p.sum(axis=-1)

    def loss(self, params, obs, acts):
        """Mean negative log-likelihood -- the BC objective.

        A scalar, because jax.grad requires one. Expect it to go negative:
        this is a continuous density, so once sigma is small the density
        exceeds 1 and its log is positive. Watch sigma, not the loss value.
        """
        return -self.log_prob(params, obs, acts).mean()

    def mean_action(self, params, obs):
        return self.forward(params, obs)[0]
