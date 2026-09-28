import argparse

import optax
import jax
import jax.numpy as jnp
from jaxrl.envs.cart_pole import CartPoleEnv
from jaxrl.experts.lqr import LQRPolicy
from jaxrl.policies.gaussian import GaussianPolicy
from jaxrl.algos import behavior_cloning as bc
from jaxrl.algos import dagger
from jaxrl.infra.rollout import sample_trajectories, sample_trajectory
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
                   default=8,
                   help="simulation steps per animation frame")
    p.add_argument("--pole-scale",
                   type=float,
                   default=2.0,
                   help="draw the pole longer than it is; display only")
    p.add_argument("--screen-frac",
                   type=float,
                   default=0.5,
                   help="window size as a fraction of the screen")
    p.add_argument("--span",
                   type=float,
                   default=None,
                   help="half-width of the animation window in metres; "
                   "pins the frame so clips of different policies match")
    p.add_argument("--save-gif",
                   type=str,
                   default=None,
                   help="also write the animation to this path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rollouts",
                   type=int,
                   default=1,
                   help="learner episodes collected per DAgger iteration")
    p.add_argument("--n-iters",
                   type=int,
                   default=10,
                   help="number of iterations for dagger")
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

    # Generat expert data
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

    params, opt_state, losses = bc.train(update, params, opt_state,
                                         expert_states, expert_actions,
                                         train_key, n_steps, batch_size)

    rollout_key = jax.random.key(args.seed + 2000)
    for i in range(args.n_iters):
        rollout_key, rollout_sub_key = jax.random.split(rollout_key)
        rollout_traj = sample_trajectories(
            rollout_sub_key,
            lambda k, s: policy.mean_action(params, s),
            env,
            n=args.n_rollouts,
            horizon=args.horizon)
        rollout_X, rollout_Y = dagger.relabel(expert_lqr,
                                              rollout_traj).flatten()
        expert_states = jnp.concatenate([expert_states, rollout_X])
        expert_actions = jnp.concatenate([expert_actions, rollout_Y])

        # a fresh key each round, or every retrain walks the same minibatch
        # sequence through a dataset that has changed underneath it
        train_key, iter_key = jax.random.split(train_key)
        params, opt_state, losses = bc.train(update,
                                             params,
                                             opt_state,
                                             expert_states,
                                             expert_actions,
                                             iter_key,
                                             n_steps,
                                             batch_size,
                                             verbose=False)
        print(f"  iter {i + 1:2}/{args.n_iters}  "
              f"pairs {expert_states.shape[0]:6}  loss {losses[-1]:8.4f}")

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
        "DAgger":
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
                            stride=args.stride,
                            pole_scale=args.pole_scale,
                            screen_frac=args.screen_frac,
                            span=args.span,
                            save=args.save_gif)
        del anim  # plt.show() has already blocked by the time this returns


if __name__ == '__main__':
    main(parse_args())
