"""A small MLP, built with Flax linen.

Flax keeps the weights outside the module: `init` returns them as a pytree and
`apply` takes them back in. The module itself is just the architecture, so it
is safe to hold on `self`; the params travel as arguments, which is what lets
optax update them and jax.grad differentiate them.

    model  = MLP(features=(64, 64, 1))
    params = model.init(key, jnp.zeros((4,)))
    out    = model.apply(params, x)
"""

from typing import Sequence

import flax.linen as nn


class MLP(nn.Module):
    """Dense layers with tanh between them and a linear output.

    `features` is the widths AFTER the input, so (64, 64, 1) on a 4-vector
    input gives 4 -> 64 -> 64 -> 1. The input width is inferred at init time
    from the example array passed to `init`.
    """

    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for f in self.features[:-1]:
            x = nn.tanh(nn.Dense(f)(x))
        return nn.Dense(self.features[-1])(x)   # linear: a force, not a probability
