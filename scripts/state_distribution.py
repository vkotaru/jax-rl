"""Compare the state distribution p(s) that each policy actually induces.

The premise behind behaviour cloning is that a policy fit to the expert's
states will be run on those same states. It will not be: it is run on its
own. This script measures the gap directly -- roll every policy from one
shared set of initial states, pool every state visited along the way, and
plot what comes out.

    uv run python scripts/state_distribution.py --save dist.png

Each policy gets the same initial states, so a difference between two
distributions is the policy and not the draw.
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from jaxrl.envs.cart_pole import CartPoleEnv
from jaxrl.experts.lqr import LQRPolicy
from jaxrl.policies.gaussian import GaussianPolicy
from jaxrl.algos import behavior_cloning as bc
from jaxrl.algos import dagger
from jaxrl.infra.rollout import sample_trajectories
from jaxrl.utils.cart_pole_render import BG, GRID, TEXT

# One colour per policy, kept in the order they are plotted.
COLOURS = {
    "expert": "#9bb5b0",
    "BC, 3 demos": "#ff7043",
    "BC, 13 demos": "#c9a227",
    "DAgger, 3+10": "#2f6f6a",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="State distributions induced by expert, BC and DAgger.")
    p.add_argument("--n-states", type=int, default=500,
                   help="initial states to roll each policy from")
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=2500,
                   help="optimisation steps per training round")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--n-demos-few", type=int, default=3,
                   help="demonstrations for the small BC run, and for DAgger")
    p.add_argument("--n-demos-many", type=int, default=13,
                   help="demonstrations for the BC run matched to DAgger's data")
    p.add_argument("--n-iters", type=int, default=10,
                   help="DAgger rounds")
    p.add_argument("--bins", type=int, default=120,
                   help="bins per axis for the plotted density")
    p.add_argument("--js-bins", type=int, default=40,
                   help="bins per axis for the divergence. Coarser than the "
                        "plot on purpose: a fine grid leaves most bins nearly "
                        "empty and the sampling floor then hides real gaps")
    p.add_argument("--x-lim", type=float, default=6.0,
                   help="cart position axis limit, metres")
    p.add_argument("--th-lim", type=float, default=60.0,
                   help="tilt axis limit, degrees from upright")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train_policy(args, env, expert, n_demos, n_iters):
    """Behaviour cloning, then n_iters rounds of DAgger on top of it."""
    data_key, init_key, train_key = jax.random.split(jax.random.key(args.seed), 3)
    demos = sample_trajectories(data_key, expert.act, env,
                                n=n_demos, horizon=args.horizon)
    obs, act = demos.obs.reshape(-1, 4), demos.action.reshape(-1, 1)

    # Frozen on the initial demonstrations: DAgger moves the distribution on
    # purpose, and rescaling as it moves would shift the net's inputs mid-run.
    policy = GaussianPolicy(4, 1, args.hidden,
                            obs_mean=obs.mean(0), obs_std=obs.std(0) + 1e-6)
    params = policy.init(init_key)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)
    update = bc.make_update(policy, opt)
    params, opt_state, _ = bc.train(update, params, opt_state, obs, act,
                                    train_key, args.n_steps, args.batch_size,
                                    verbose=False)

    roll_key = jax.random.key(args.seed + 2000)
    for _ in range(n_iters):
        roll_key, sub = jax.random.split(roll_key)
        traj = sample_trajectories(sub,
                                   lambda k, s: policy.mean_action(params, s),
                                   env, n=1, horizon=args.horizon)
        add_obs, add_act = dagger.relabel(expert, traj).flatten()
        obs = jnp.concatenate([obs, add_obs])
        act = jnp.concatenate([act, add_act])
        train_key, iter_key = jax.random.split(train_key)
        params, opt_state, _ = bc.train(update, params, opt_state, obs, act,
                                        iter_key, args.n_steps,
                                        args.batch_size, verbose=False)
    return policy, params


def tilt_degrees(obs):
    """Signed degrees from upright. The env keeps theta in [0, 2pi)."""
    return np.degrees(np.asarray(obs[..., 1])) - 180.0


def style(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9.5, length=4, color=GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def main(args):
    env = CartPoleEnv()
    expert = LQRPolicy.from_env(env,
                                Q=jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1])),
                                R=jnp.diag(jnp.array([1.0])))

    print(f"training on {args.n_demos_few} and {args.n_demos_many} demos, "
          f"plus {args.n_iters} DAgger rounds")
    bc_few = train_policy(args, env, expert, args.n_demos_few, 0)
    bc_many = train_policy(args, env, expert, args.n_demos_many, 0)
    dag = train_policy(args, env, expert, args.n_demos_few, args.n_iters)

    # One key for every policy, so every rollout set starts from the same
    # states and a difference downstream is the policy alone.
    roll_key = jax.random.key(args.seed + 1000)
    runs = {"expert": expert.act}
    for name, (pol, prm) in (("BC, 3 demos", bc_few),
                             ("BC, 13 demos", bc_many),
                             ("DAgger, 3+10", dag)):
        # bind per iteration, or every entry closes over the last policy
        runs[name] = lambda k, s, p=pol, q=prm: p.mean_action(q, s)

    visited = {}
    print(f"\nrolling {args.n_states} episodes of {args.horizon} steps each")
    print(f"\n{'':14} {'std x':>8} {'std tilt':>10} {'|tilt|>30':>10} {'|x|>4':>8}"
          f" {'off-plot':>10}")
    for name, fn in runs.items():
        traj = sample_trajectories(roll_key, fn, env,
                                   n=args.n_states, horizon=args.horizon)
        x = np.asarray(traj.obs[..., 0]).ravel()
        th = tilt_degrees(traj.obs).ravel()
        visited[name] = (x, th)
        off = np.mean((np.abs(x) > args.x_lim) | (np.abs(th) > args.th_lim))
        print(f"{name:14} {x.std():8.3f} {th.std():10.2f} "
              f"{np.mean(np.abs(th) > 30):9.1%} {np.mean(np.abs(x) > 4):7.1%}"
              f" {off:10.1%}")

    # ---- figure ---------------------------------------------------------
    names = list(visited)
    fig = plt.figure(figsize=(14.0, 7.8), facecolor=BG)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.34)
    top = outer[0].subgridspec(1, 2, wspace=0.2)
    bottom = outer[1].subgridspec(1, len(names), wspace=0.12)

    # Top row: the two marginals, every policy overlaid.
    for col, (vals, label, lim) in enumerate((
            (0, "cart position $x$ (m)", args.x_lim),
            (1, "tilt from upright (deg)", args.th_lim))):
        ax = fig.add_subplot(top[0, col])
        for name in names:
            v = visited[name][vals]
            # No clipping: folding the tail into the edge bin invents a spike
            # exactly where the interesting policies differ. Out-of-window
            # mass is reported in the table instead.
            ax.hist(v, bins=args.bins, range=(-lim, lim),
                    density=True, histtype="step", lw=2.1,
                    color=COLOURS[name], label=name)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("density", fontsize=11)
        ax.set_xlim(-lim, lim)
        style(ax)
        if col == 0:
            leg = ax.legend(frameon=False, fontsize=10, loc="upper right")
            for t in leg.get_texts():
                t.set_color(TEXT)

    # ---- how close is p_pi(s) to p_expert(s)? ---------------------------
    # Two distributions can share a standard deviation and still differ, so
    # compare them bin by bin. Jensen-Shannon is symmetric, finite where one
    # side is empty, and in bits: 0 identical, 1 disjoint. Mass outside the
    # window goes in its own bin so each policy still sums to one and the
    # runaways are not quietly discarded.
    rng_d = [[-args.x_lim, args.x_lim], [-args.th_lim, args.th_lim]]

    def occupancy_of(x, th):
        h, _, _ = np.histogram2d(x, th, bins=args.js_bins, range=rng_d)
        return np.append(h.ravel(), len(x) - h.sum()) / len(x)

    def occupancy(name):
        return occupancy_of(*visited[name])

    def js_bits(p_, q_):
        m = 0.5 * (p_ + q_)
        def kl(a, b):
            nz = a > 0
            return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
        return 0.5 * kl(p_, m) + 0.5 * kl(q_, m)

    # A finite sample never reproduces itself exactly, so split the expert's
    # own episodes in two and compare the halves. That is the floor: a
    # divergence at or near it is as close a match as this many rollouts can
    # resolve, not evidence of an actual difference.
    ex_x, ex_th = visited["expert"]
    per = args.horizon
    half = (args.n_states // 2) * per
    floor = js_bits(occupancy_of(ex_x[:half], ex_th[:half]),
                    occupancy_of(ex_x[half:], ex_th[half:]))

    ref = occupancy("expert")
    print(f"\n{'':14} {'JS vs expert (bits)':>22}   {'':>4}")
    js = {}
    for name in names:
        js[name] = js_bits(occupancy(name), ref)
        bar = "#" * int(round(js[name] * 40))
        print(f"{name:14} {js[name]:22.4f}   {bar}")
    print(f"{'(noise floor)':14} {floor:22.4f}   "
          f"expert vs itself, split in half")
    for name in names[1:]:
        print(f"    {name:18} {js[name] / floor:5.1f}x the floor")

    # Bottom row: the joint, one panel per policy. The panels share one colour
    # scale, or each normalises to its own peak and they cannot be compared.
    rng = [[-args.x_lim, args.x_lim], [-args.th_lim, args.th_lim]]
    grids = {}
    for name in names:
        x, th = visited[name]
        h, _, _ = np.histogram2d(x, th, bins=args.bins, range=rng)
        grids[name] = h.T
    vmax = max(g.max() for g in grids.values())

    for col, name in enumerate(names):
        ax = fig.add_subplot(bottom[0, col])
        g = np.where(grids[name] > 0, grids[name], np.nan)
        im = ax.imshow(g, origin="lower", aspect="auto", cmap="magma_r",
                       norm=LogNorm(vmin=1, vmax=vmax),
                       extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]])
        ax.set_title(name, fontsize=11.5, pad=7)
        ax.set_xlabel("$x$ (m)", fontsize=11)
        if col == 0:
            ax.set_ylabel("tilt (deg)", fontsize=11)
        else:
            ax.set_yticklabels([])
        ax.axhline(0, lw=0.8, color=GRID, zorder=3)
        ax.axvline(0, lw=0.8, color=GRID, zorder=3)
        style(ax)

    cb = fig.colorbar(im, ax=fig.axes[-len(names):], fraction=0.015, pad=0.012)
    cb.set_label("states per bin", color=TEXT, fontsize=10)
    cb.ax.tick_params(colors=TEXT, labelsize=8.5)
    cb.outline.set_edgecolor(GRID)

    fig.suptitle("Where each policy actually spends its time",
                 color=TEXT, fontsize=13, y=0.975)
    if args.save:
        fig.savefig(args.save, dpi=args.dpi, facecolor=BG,
                    bbox_inches="tight")
        print(f"\nwrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main(parse_args())
