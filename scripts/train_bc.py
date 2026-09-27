import argparse

import optax
import jax
import jax.numpy as jnp
from jaxrl.envs.cart_pole import CartPoleEnv
from jaxrl.experts.lqr import LQRPolicy
from jaxrl.policies.gaussian import GaussianPolicy
from jaxrl.algos import behavior_cloning as bc
from jaxrl.infra.rollout import sample_trajectories
from jaxrl.utils.cart_pole_render import compare_grid


def parse_args():
    p = argparse.ArgumentParser(description="Behaviour cloning on cart-pole.")
    # data
    p.add_argument("--n-demos", type=int, default=50, help="expert episodes")
    p.add_argument("--horizon",
                   type=int,
                   default=500,
                   help="steps per episode")
    # training
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="passes over the dataset; n_steps is derived from this")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--n-steps",
                   type=int,
                   default=None,
                   help="override the step count derived from --epochs")
    p.add_argument("--lr", type=float, default=1e-3)
    # net
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--n-eval",
                   type=int,
                   default=32,
                   help="episodes to evaluate on")
    p.add_argument("--n-animate",
                   type=int,
                   default=0,
                   help="episodes to animate side by side; 0 to skip")
    p.add_argument("--stride",
                   type=int,
                   default=6,
                   help="simulation steps per animation frame")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main(args):
    batch_size = args.batch_size

    env = CartPoleEnv()

    expert_lqr = LQRPolicy.from_env(env,
                                    Q=jnp.diag(jnp.array([1.0, 10., 0.1,
                                                          0.1])),
                                    R=jnp.diag(jnp.array([1.0])))
    # Confirm lqr Gain
    print(expert_lqr.K)

    # Generat expert dat
    key = jax.random.key(args.seed)
    data_key, init_key, train_key = jax.random.split(key, 3)
    expert_data = sample_trajectories(data_key,
                                      expert_lqr.act,
                                      env,
                                      n=args.n_demos,
                                      horizon=args.horizon)
    expert_states = expert_data.obs.reshape(-1, 4)
    expert_actions = expert_data.action.reshape(-1, 1)
    n_pairs = expert_states.shape[0]
    print(f"dataset: {expert_states.shape} {expert_actions.shape}")

    # Policy
    policy = GaussianPolicy(4,
                            1,
                            args.hidden,
                            obs_mean=expert_states.mean(0),
                            obs_std=expert_states.std(0) + 1e-6)
    params = policy.init(init_key)

    # Train
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)
    update = bc.make_update(policy, opt)

    # one epoch = one pass over the data, so n_steps follows from it
    steps_per_epoch = max(1, n_pairs // batch_size)
    n_steps = args.n_steps if args.n_steps is not None else args.epochs * steps_per_epoch
    print(f"{steps_per_epoch} steps/epoch, training for {n_steps} steps")

    losses = []
    for i in range(n_steps):
        train_key, sub = jax.random.split(train_key)
        idx = jax.random.randint(sub, (batch_size, ), 0,
                                 expert_states.shape[0])
        params, opt_state, loss = update(params, opt_state, expert_states[idx],
                                         expert_actions[idx])
        losses.append(float(loss))
        if i % max(1, n_steps // 10) == 0:
            print(f"  step {i:6}/{n_steps}  loss {float(loss):9.4f}")

    # ---- evaluate ---------------------------------------------------------
    # How well does it fit, on the states it was trained on?
    mu, log_std = policy.forward(params, expert_states)
    rmse = float(jnp.sqrt(jnp.mean((mu - expert_actions)**2)))
    print(f"\nfit on expert states: rmse {rmse:.4f} N   "
          f"(expert action std {float(jnp.std(expert_actions)):.4f} N)   "
          f"sigma {float(jnp.exp(log_std).mean()):.4f}")

    # How well does it actually balance? Same key for both, so the learner and
    # the expert start from identical states and the comparison is paired.
    eval_key = jax.random.key(args.seed + 1000)

    def tilt_deg(traj):
        """Degrees away from upright at each step, (n_episodes, horizon)."""
        return jnp.abs(jnp.degrees(traj.obs[:, :, 1]) - 180.0)

    rollouts = {
        "expert":
        sample_trajectories(eval_key,
                            expert_lqr.act,
                            env,
                            n=args.n_eval,
                            horizon=args.horizon),
        "BC":
        sample_trajectories(eval_key,
                            lambda k, s: policy.mean_action(params, s),
                            env,
                            n=args.n_eval,
                            horizon=args.horizon),
    }

    print(f"\n{'':8} {'final tilt':>18} {'max tilt':>12} {'upright':>10}")
    for name, traj in rollouts.items():
        t = tilt_deg(traj)
        final, worst = t[:, -1], t.max(axis=1)
        # "upright" = never tipped past 30 deg during the whole episode
        upright = float(jnp.mean(worst < 30.0))
        print(
            f"{name:8} {float(final.mean()):8.3f} +- {float(final.std()):5.3f} deg"
            f" {float(worst.mean()):9.2f} deg {upright:9.0%}")

    # ---- animate ----------------------------------------------------------
    if args.n_animate > 0:
        # Both series share eval_key, so panel i starts from the same state
        # for both policies and any divergence is down to the policy.
        anim = compare_grid({
            k: v.obs
            for k, v in rollouts.items()
        },
                            env.l,
                            n=args.n_animate,
                            dt=env.dt,
                            stride=args.stride)
        del anim  # plt.show() has already blocked by the time this returns


if __name__ == '__main__':
    main(parse_args())
