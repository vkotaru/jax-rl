"""Behaviour cloning: supervised learning on (state, expert action) pairs.

No environment interaction, no reward, no bootstrapping. The only thing that
makes it an RL method at all is where the data came from, and the only thing
that makes it fail is that the policy is evaluated on states it chose rather
than on the states the expert demonstrated.
"""

import jax
import optax


def make_update(policy, optimizer):
    """Build one jitted gradient step for `policy` under `optimizer`.

    Returns a function rather than doing the step directly so that `policy`
    and `optimizer` are captured once, as constants, instead of being passed
    in on every call. jit would otherwise have to treat them as static
    arguments and retrace whenever they changed identity.

    The returned `update` takes and returns both `params` and `opt_state`,
    because neither can be stored anywhere: params change every step, and
    optimizer state is Adam's two running averages, shaped like the params.
    Threading them through is the same pattern as `state` in env.step and
    `key` in reset.
    """

    @jax.jit
    def update(params, opt_state, x, y):
        # value_and_grad gives the loss and the gradients in one pass; the
        # loss is reported anyway, so computing it separately would be waste.
        # policy.loss is the mean negative log-likelihood of the expert's
        # actions -- a scalar, which is what grad requires.
        loss, grads = jax.value_and_grad(policy.loss)(params, x, y)

        # optax splits "decide the step" from "apply it": update() turns
        # gradients into parameter deltas using its own state, and
        # apply_updates adds them. Both walk the params pytree leaf by leaf,
        # so this works whatever shape the network is.
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return update
