"""Cart-pole dynamics.

State is ``[x, theta, x_dot, theta_dot]``. Theta is measured from the downward
vertical and is positive counter-clockwise, so theta = 0 is the pole hanging
(stable) and theta = pi is upright (unstable, and the one worth controlling).

Nothing here holds simulation state: `step` takes a state and returns the next
one. That keeps every method a pure function of its arguments, which is what
lets them pass through jit, vmap and grad. Only constants live on `self`, and
they must never be reassigned -- jit bakes them in at trace time, so a later
change is silently ignored by anything already compiled.
"""

import jax
import jax.numpy as jnp


class CartPoleEnv:
    """A cart on a rail with a point mass on a rigid massless rod."""

    def __init__(self,
                 cart_mass=1.0,
                 pole_mass=0.1,
                 pole_length=1.0,
                 gravity=9.81,
                 cart_friction=0.0,
                 pole_friction=0.0,
                 dt=0.01):
        """`pole_length` is the pivot-to-bob distance; the rod carries no mass.

        Friction coefficients default to zero, which makes the system
        conservative -- useful, because then total energy is a check on the
        integrator.
        """
        self.M = cart_mass
        self.m = pole_mass
        self.l = pole_length
        self.g = gravity
        self.mu_c = cart_friction
        self.mu_p = pole_friction
        self.dt = dt
        # upright: the equilibrium the expert stabilises and resets perturb
        self.eq = jnp.array([0.0, jnp.pi, 0.0, 0.0])
        # per-component half-widths: x (m), theta (rad), x_dot, theta_dot
        self.reset_scale = jnp.array([4.0, jnp.pi / 18, 0.5, 0.1 * jnp.pi])

    def reset(self, key: jax.Array) -> jax.Array:
        """Draw an initial state in a box around `eq`.

        The key is consumed: calling this twice with the same key gives the
        same state. Callers split a key per episode.
        """
        return self.eq + jax.random.uniform(
            key, (4, ), minval=-self.reset_scale, maxval=self.reset_scale)

    def linearize(self):
        """Continuous-time A, B about the UPRIGHT equilibrium, derived by hand.

        The entries assume sin(theta)=0 and cos(theta)=-1, i.e. theta = pi.
        They are not recomputed from `self.eq`, so changing `eq` would leave
        these silently wrong; use `linearize_discrete` if you move it.

        Continuous, so this is the pair to hand to a CARE solver such as
        `control.lqr`. Passing the discrete pair there instead produces a
        plausible-looking gain that diverges.
        """
        m, M, l, g = self.m, self.M, self.l, self.g
        mu_c, mu_p = self.mu_c, self.mu_p
        A = jnp.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, m * g / M, -mu_c / M, -mu_p / (M * l)],
            [
                0.0, (M + m) * g / (M * l), -mu_c / (M * l),
                -(M + m) * mu_p / (M * m * l**2)
            ],
        ])
        B = jnp.array([[0.0], [0.0], [1.0 / M], [1.0 / (M * l)]])
        return A, B

    def linearize_discrete(self):
        """Discrete A, B by differentiating `step`, exact for this integrator.

        Differs from `I + A*dt` at order dt^2 because of the semi-implicit
        ordering. Use with a DARE solver, or as a check on `linearize`.
        """
        u0 = jnp.zeros(())
        A = jax.jacobian(self.step, argnums=0)(self.eq, u0)
        B = jax.jacobian(self.step, argnums=1)(self.eq, u0).reshape(4, 1)
        return A, B

    def step(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """One timestep. `action` is a horizontal force on the cart, in newtons.

        Accepts a scalar or shape-(1,) action and returns shape (4,). Batch by
        wrapping in vmap rather than by passing stacked inputs here.
        """
        m, M, l, g = self.m, self.M, self.l, self.g
        mu_c, mu_p = self.mu_c, self.mu_p

        x, th, dx, dth = state
        u = jnp.squeeze(action)
        # accelerations at the current state
        d2x = ((u - mu_c * dx + m * l * dth**2 * jnp.sin(th) + m * g *
                jnp.sin(th) * jnp.cos(th) + mu_p / l * dth * jnp.cos(th)) /
               (M + m * jnp.sin(th)**2))
        d2th = ((-(M + m) * g * jnp.sin(th) - m * l * dth * dth * jnp.sin(th) *
                 jnp.cos(th) + mu_c * dx * jnp.cos(th) - u * jnp.cos(th) -
                 (M + m) / (m * l) * mu_p * dth) / (l *
                                                    (M + m * jnp.sin(th)**2)))

        # Semi-implicit (symplectic) Euler: advance the velocities, then use
        # the NEW velocities to advance the positions.
        #
        # Feeding the old velocities into the position update instead expands
        # the phase-space map by (1 + dt^2) per step rather than exactly 1, so
        # energy grows without bound -- around 20 J from nothing over 100k
        # zero-force steps. This ordering holds the error near 1e-5 forever,
        # at identical cost.
        #
        # Velocity Verlet is more accurate but requires acceleration to depend
        # on position alone; d2th depends on dth through the centrifugal and
        # friction terms.
        dx = dx + self.dt * d2x
        dth = dth + self.dt * d2th
        x = x + self.dt * dx
        th = th + self.dt * dth
        th = jnp.mod(th, 2 * jnp.pi)

        return jnp.array([x, th, dx, dth])
