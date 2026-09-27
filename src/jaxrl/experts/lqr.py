"""Infinite-horizon LQR, used here as the expert to imitate.

Unlike a frozen neural expert, this one can be queried at any state at all,
which is what DAgger needs: it relabels states the learner visited, not states
anyone demonstrated.
"""

import jax.numpy as jnp
import control


class LQRPolicy:
    """Linear state feedback, u = -K (x - setpoint).

    K is constant once solved, so it lives on `self` and `act` stays a pure
    function of the state.
    """

    def __init__(self, a_mat, b_mat, q_mat, r_mat, setpoint):
        """A and B must be CONTINUOUS-time.

        `control.lqr` solves the continuous Riccati equation. Handing it a
        discrete pair returns gains around 1e7 that destabilise the plant, with
        no error raised -- prefer `from_env`, which picks the right pair.

        Q and R set the relative price of state error against control effort;
        only their ratio matters, not their scale.
        """
        self.A = a_mat
        self.B = b_mat
        self.Q = q_mat
        self.R = r_mat
        K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)
        self.K = jnp.asarray(K)
        self.setpoint = setpoint

    @classmethod
    def from_env(cls, env, Q, R, setpoint=None):
        """Build from an env, so A, B and the setpoint cannot disagree.

        Defaults to regulating about `env.eq`; pass a setpoint to hold the
        pole upright somewhere other than x = 0.
        """
        A, B = env.linearize()
        return cls(A, B, Q, R, env.eq if setpoint is None else setpoint)

    def act(self, key, state):
        """Action for one state, shape (1,).

        The key is unused -- this policy is deterministic. It is in the
        signature so a rollout can call the expert and a stochastic learned
        policy through the same interface.
        """
        return -self.K @ (state - self.setpoint)
