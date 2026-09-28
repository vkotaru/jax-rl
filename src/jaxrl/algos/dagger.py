import jax
import optax
from jaxrl.infra.rollout import Trajectory


def relabel(expert, traj: Trajectory) -> Trajectory:
    actions = jax.vmap(jax.vmap(lambda s: expert.act(None, s)))(traj.obs)
    return traj._replace(action=actions)
