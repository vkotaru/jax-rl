"""How much does behaviour cloning need to see?

Sweeps the number of expert demonstrations and measures two different things:

  rmse     -- how well the policy fits the expert ON THE EXPERT'S OWN STATES
  upright  -- how often it actually balances when it drives

They come apart, and the gap is the point. A low rmse with a low upright rate
means the fit is fine and the failures come from compounding error carrying
the policy off the demonstrated distribution -- which is what DAgger fixes.
A high rmse means it simply has not learned the controller, which more data
fixes and DAgger does not.

    python scripts/sweep_demos.py --demos 1 2 5 10 20 50
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.algos import behavior_cloning as bc
from jaxrl.envs.cart_pole import CartPoleEnv
from jaxrl.experts.lqr import LQRPolicy
from jaxrl.infra.rollout import sample_trajectories
from jaxrl.policies.gaussian import GaussianPolicy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demos", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 20, 50])
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=2500,
                   help="fixed, so every point gets the same optimisation budget")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--seeds", type=int, default=3,
                   help="repeats per point; the spread matters at small n")
    p.add_argument("--out", type=str, default="../rl_blog/03-demos-sweep")
    return p.parse_args()


def run_one(args, env, expert, n_demos, seed):
    """Train BC on `n_demos` episodes and return (rmse, upright fraction)."""
    dk, ik, tk, ek = jax.random.split(jax.random.key(seed), 4)

    demos = sample_trajectories(dk, expert.act, env, n=n_demos,
                                horizon=args.horizon)
    X = demos.obs.reshape(-1, 4)
    Y = demos.action.reshape(-1, 1)

    # normalisation statistics come from this run's own demonstrations, frozen
    policy = GaussianPolicy(4, 1, args.hidden,
                            obs_mean=X.mean(0), obs_std=X.std(0) + 1e-6)
    params = policy.init(ik)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)
    update = bc.make_update(policy, opt)

    for _ in range(args.n_steps):
        tk, sub = jax.random.split(tk)
        idx = jax.random.randint(sub, (args.batch_size,), 0, X.shape[0])
        params, opt_state, _ = update(params, opt_state, X[idx], Y[idx])

    mu, log_std = policy.forward(params, X)
    rmse = float(jnp.sqrt(jnp.mean((mu - Y) ** 2)))
    sigma = float(jnp.exp(log_std).mean())

    roll = sample_trajectories(ek, lambda k, s: policy.mean_action(params, s),
                               env, n=args.n_eval, horizon=args.horizon)
    tilt = jnp.abs(jnp.degrees(roll.obs[:, :, 1]) - 180.0)
    upright = float(jnp.mean(tilt.max(axis=1) < 30.0))
    return rmse, upright, sigma


def main(args):
    env = CartPoleEnv()
    expert = LQRPolicy.from_env(env,
                                Q=jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1])),
                                R=jnp.diag(jnp.array([1.0])))

    rows = []
    print(f"{'demos':>6} {'pairs':>8} {'rmse':>16} {'upright':>16}")
    for n in args.demos:
        r = np.array([run_one(args, env, expert, n, s) for s in range(args.seeds)])
        rows.append({"n_demos": n, "pairs": n * args.horizon,
                     "rmse": r[:, 0].tolist(), "upright": r[:, 1].tolist(),
                     "sigma": r[:, 2].tolist()})
        print(f"{n:>6} {n * args.horizon:>8} "
              f"{r[:,0].mean():>9.4f} +-{r[:,0].std():<5.4f} "
              f"{r[:,1].mean():>9.1%} +-{r[:,1].std():<5.1%}")

    json.dump({"args": vars(args), "rows": rows}, open(args.out + ".json", "w"),
              indent=2)
    plot(args, rows)


def plot(args, rows):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")
    plt.rcParams.update({"font.family": "Lato"})
    BG, GRID, ROD, BALL, TEXT = "#fbfbf9", "#e2e9e6", "#2f6f6a", "#ff7043", "#7f938f"

    n = np.array([r["n_demos"] for r in rows])
    rmse = np.array([r["rmse"] for r in rows])
    sig = np.array([r["sigma"] for r in rows])
    up = np.array([r["upright"] for r in rows]) * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=BG)
    for ax, y, colour, label in [(axes[0], rmse, ROD, "fit error on expert states  (N)"),
                                 (axes[1], sig, ROD, "learned sigma  (N)"),
                                 (axes[2], up, BALL, "episodes upright  (%)")]:
        ax.set_facecolor(BG)
        ax.plot(n, y.mean(1), "-o", color=colour, lw=2, ms=6)
        ax.fill_between(n, y.min(1), y.max(1), color=colour, alpha=.18, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(n)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("expert demonstrations", color=TEXT)
        ax.set_title(label.upper(), color=TEXT, fontsize=9.5, loc="left",
                     fontweight="bold", pad=10)
        ax.tick_params(colors=TEXT, labelsize=9.5, length=0)
        ax.grid(color=GRID, lw=.9)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[2].set_ylim(-3, 103)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=170, facecolor=BG)
    print("wrote", args.out + ".png")


if __name__ == "__main__":
    main(parse_args())
