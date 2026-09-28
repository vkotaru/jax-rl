"""Backward reachable sets of the upright equilibrium under the LQR policy.

The backward reachable set at horizon T is the set of states from which the
closed loop arrives at the equilibrium within T:

    BRS(T) = { s0 : phi(t, s0) in B_eps(s*) for some t <= T }

and the region of attraction is the set that gets there eventually:

    ROA    = { s0 : phi(t, s0) -> s* as t -> infinity }

Under a FIXED policy these are the same set in the limit. Every point of the
ROA eventually enters any ball around the equilibrium, and any point that
enters a small enough ball converges, because the ball lies in the ROA. So
BRS(T) grows monotonically into the ROA, and ROA = union over T of BRS(T).

They separate in two places. At finite T the containment is strict, which is
what the contours below show. And if the control is a *choice* rather than a
fixed policy, the backward reachable set is every state some admissible input
can drive to the target -- a property of the system, not of a controller, and
larger than any single policy's ROA.

One forward simulation per start gives every horizon at once: record the first
time the trajectory enters the ball, and BRS(T) is the sublevel set of that
time. Never entering means outside the ROA.

    uv run python scripts/reachability.py --save roa.png
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jaxrl.envs.cart_pole import CartPoleEnv
from jaxrl.experts.lqr import LQRPolicy
from jaxrl.utils.cart_pole_render import BG, GRID, TEXT

NEVER = "#f0ece6"        # outside the ROA: never reaches the equilibrium
RESET = "#ff7043"
# Contour levels are quantiles of the reach times rather than fixed seconds:
# with exponential convergence the times cluster, and a fixed ladder puts every
# contour in the same place.
QUANTILES = (0.1, 0.5, 0.9)


def parse_args():
    p = argparse.ArgumentParser(
        description="Backward reachable sets and the region of attraction.")
    p.add_argument("--grid", type=int, default=301,
                   help="samples per axis on each slice")
    p.add_argument("--horizon", type=int, default=3000,
                   help="steps simulated; the largest horizon resolvable, so "
                        "the ROA is whatever is reached inside it")
    p.add_argument("--episode-steps", type=int, default=500,
                   help="the episode length the other experiments use. What "
                        "they can see is BRS at this horizon, not the ROA")
    p.add_argument("--tol", type=float, default=0.5,
                   help="radius of the target ball. Small values measure "
                        "exponential settling rather than travel time, and "
                        "every start then takes about equally long")
    p.add_argument("--tilt-lim", type=float, default=180.0)
    p.add_argument("--thd-lim", type=float, default=12.0)
    p.add_argument("--x-lim", type=float, default=30.0)
    p.add_argument("--xd-lim", type=float, default=20.0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--save", type=str, default=None)
    return p.parse_args()


def make_time_to_reach(env, expert, horizon, tol):
    """Seconds until the closed loop enters the ball, or inf if it never does."""
    eq = env.eq

    def ttr(state):
        def body(carry, i):
            s, first = carry
            s = env.step(s, expert.act(None, s))
            err = s - eq
            # the angle is periodic, so compare it on the circle
            err = err.at[1].set(jnp.mod(err[1] + jnp.pi, 2 * jnp.pi) - jnp.pi)
            hit = jnp.linalg.norm(err) < tol
            first = jnp.where((first < 0) & hit, i, first)
            return (s, first), None

        (_, first), _ = jax.lax.scan(body, (state, -1), jnp.arange(horizon))
        return jnp.where(first < 0, jnp.inf, first * env.dt)

    return jax.jit(jax.vmap(ttr))


def slice_grid(env, ttr, axis_a, axis_b, lim_a, lim_b, n):
    """Time-to-reach on a 2D slice through the equilibrium.

    A slice, not a projection: a point dark here may still lie in the ROA at
    some other value of the two frozen coordinates.
    """
    a = np.linspace(-lim_a, lim_a, n)
    b = np.linspace(-lim_b, lim_b, n)
    A, B = np.meshgrid(a, b, indexing="xy")
    states = np.tile(np.asarray(env.eq), (A.size, 1))
    states[:, axis_a] += A.ravel()
    states[:, axis_b] += B.ravel()
    return a, b, np.asarray(ttr(jnp.asarray(states))).reshape(A.shape)


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9.5, length=4, color=GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def draw(ax, xs, ys, T, reset_box, vmax, levels):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(NEVER)
    m = ax.pcolormesh(xs, ys, np.ma.masked_invalid(T), cmap=cmap,
                      vmin=0, vmax=vmax, shading="auto", rasterized=True)
    finite = np.isfinite(T)
    cs = ax.contour(xs, ys, np.where(finite, T, 1e9), levels=list(levels),
                    colors="white", linewidths=1.1, alpha=0.85)
    ax.clabel(cs, fmt=lambda v: f"{v:g}s", fontsize=8.5, colors="white",
              inline_spacing=6)
    ax.contour(xs, ys, finite.astype(float), levels=[0.5],
               colors=["#1f3a37"], linewidths=2.0)
    w, h = reset_box
    ax.add_patch(plt.Rectangle((-w, -h), 2 * w, 2 * h, fill=False,
                               ec=RESET, lw=2.0, zorder=6))
    return m


def main(args):
    env = CartPoleEnv()
    expert = LQRPolicy.from_env(env,
                                Q=jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1])),
                                R=jnp.diag(jnp.array([1.0])))
    ttr = make_time_to_reach(env, expert, args.horizon, args.tol)
    rs = np.asarray(env.reset_scale)

    print(f"{args.grid}x{args.grid} starts per slice, {args.horizon} steps "
          f"({args.horizon * env.dt:.0f}s), target ball r={args.tol}")
    th, thd, T_ang = slice_grid(env, ttr, 1, 3,
                                np.radians(args.tilt_lim), args.thd_lim,
                                args.grid)
    x, xd, T_pos = slice_grid(env, ttr, 0, 2,
                              args.x_lim, args.xd_lim, args.grid)

    t_ep_s = args.episode_steps * env.dt
    for nm, T in (("angle (theta, thetadot)", T_ang), ("cart  (x, xdot)", T_pos)):
        fin = np.isfinite(T)
        print(f"\n  {nm} slice")
        print(f"    in the ROA          : {fin.mean():6.1%} of the window")
        if fin.any():
            print(f"    time to reach       : median {np.median(T[fin]):.2f}s, "
                  f"max {T[fin].max():.2f}s")
            print(f"    BRS({t_ep_s:g}s)/ROA at the episode horizon"
                  f"      : {(T <= t_ep_s).sum() / fin.sum():6.1%}")
            for q in QUANTILES:
                L = np.quantile(T[fin], q)
                frac = (T <= L).sum() / max(fin.sum(), 1)
                print(f"    BRS({L:>5.2f}s) / ROA : {frac:6.1%}")

    vmax = max(np.nanpercentile(T_ang[np.isfinite(T_ang)], 99),
               np.nanpercentile(T_pos[np.isfinite(T_pos)], 99))

    # A dedicated colourbar column, or its label lands on the third panel's.
    fig = plt.figure(figsize=(16.4, 5.0), facecolor=BG)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.055, 0.16, 0.95],
                          wspace=0.30, bottom=0.17, top=0.90)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 4])]
    cax = fig.add_subplot(gs[0, 2])
    allT = np.concatenate([T_ang[np.isfinite(T_ang)].ravel(),
                           T_pos[np.isfinite(T_pos)].ravel()])
    levels = sorted(set(np.round(np.quantile(allT, QUANTILES), 2)))

    t_ep = args.episode_steps * env.dt
    m = draw(axes[0], np.degrees(th), thd, T_ang,
             (np.degrees(rs[1]), rs[3]), vmax, levels)
    for a, T, xs_, ys_ in ((axes[0], T_ang, np.degrees(th), thd),
                           (axes[1], T_pos, x, xd)):
        a.contour(xs_, ys_, np.where(np.isfinite(T), T, 1e9), levels=[t_ep],
                  colors=[RESET], linewidths=2.4, zorder=7)
    axes[0].set_title("$(\\theta,\\dot\\theta)$ slice, $x=\\dot x=0$",
                      fontsize=11.5)
    axes[0].set_xlabel("tilt from upright (deg)", fontsize=11)
    axes[0].set_ylabel("$\\dot\\theta$ (rad/s)", fontsize=11)

    draw(axes[1], x, xd, T_pos, (rs[0], rs[2]), vmax, levels)
    axes[1].set_title("$(x,\\dot x)$ slice, upright", fontsize=11.5)
    axes[1].set_xlabel("$x$ (m)", fontsize=11)
    axes[1].set_ylabel("$\\dot x$ (m/s)", fontsize=11)

    # How the finite-horizon sets fill out the ROA.
    ax = axes[2]
    grid_T = np.linspace(0.05, min(vmax * 1.6, args.horizon * env.dt), 260)
    for T, nm, c in ((T_ang, "$(\\theta,\\dot\\theta)$", "#2f6f6a"),
                     (T_pos, "$(x,\\dot x)$", "#c9a227")):
        fin = np.isfinite(T)
        frac = [(T <= t).sum() / fin.sum() for t in grid_T]
        ax.plot(grid_T, frac, lw=2.2, color=c, label=nm)
    ax.axhline(1.0, lw=1.2, ls=":", color=TEXT)
    ax.axvline(t_ep, lw=2.0, color=RESET, zorder=1)
    ax.annotate(f"episode horizon\n{args.episode_steps} steps = {t_ep:g}s",
                xy=(t_ep, 0.06), xytext=(6, 0), textcoords="offset points",
                color=RESET, fontsize=9, va="bottom")
    ax.set_ylim(0, 1.06)
    ax.set_xlim(0, grid_T[-1])
    ax.set_title("BRS$(T)$ fills the ROA", fontsize=11.5)
    ax.set_xlabel("horizon $T$ (s)", fontsize=11)
    ax.set_ylabel("area BRS$(T)$ / area ROA", fontsize=11)
    leg = ax.legend(frameon=False, fontsize=10, loc="lower right")
    for t in leg.get_texts():
        t.set_color(TEXT)

    for a in axes:
        style(a)
    cb = fig.colorbar(m, cax=cax)
    cb.ax.yaxis.set_label_position("left")
    cb.ax.yaxis.set_ticks_position("left")
    cb.set_label("time to reach the equilibrium (s)", color=TEXT, fontsize=10)
    cb.ax.tick_params(colors=TEXT, labelsize=8.5)
    cb.outline.set_edgecolor(GRID)
    fig.text(0.5, 0.035,
             "dark outline: ROA boundary    orange: BRS at the episode "
             "horizon, and the reset box    white: BRS(T) contours    "
             "pale: never reaches",
             color=TEXT, fontsize=9.5, ha="center")
    if args.save:
        fig.savefig(args.save, dpi=args.dpi, facecolor=BG, bbox_inches="tight")
        print(f"\nwrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main(parse_args())
