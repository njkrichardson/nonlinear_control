from pathlib import Path
from typing import List 

from jax.config import config
import jax.numpy as jnp
import jax
import matplotlib
matplotlib.use("Agg") 
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from trajax import integrators
from trajax.experimental.sqp import shootsqp, util


np = jnp 
ndarray: type = jnp.ndarray


def render_scene(obs, path: Path=None):
  # Setup obstacle environment for state constraint
  world_range = (jnp.array([-0.5, -0.5]), jnp.array([3.5, 3.5]))

  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(111)
  plt.grid(False)

  for ob in obs:
    ax.add_patch(plt.Rectangle((ob[0][0], ob[0][1]), 0.5, 0.5, color='k', alpha=0.3))

  ax.set_xlim([world_range[0][0], world_range[1][0]])
  ax.set_ylim([world_range[0][1], world_range[1][1]])
  ax.set_aspect('equal')

  if path is None: 
      return fig, ax
  else: 
      plt.savefig(path) 
      plt.close()

def car_ode(x, u, t):
  del t
  return jnp.array([x[3] * jnp.sin(x[2]),
                    x[3] * jnp.cos(x[2]),
                    x[3] * u[0],
                    u[1]])

def main(): 
    obs: List[ndarray] = [
            (jnp.array([1., 1.]), 0.5),
            (jnp.array([1, 2.5]), 0.5),
            (jnp.array([2.5, 2.5]), 0.5), 
            ]

    dt = 0.05
    dynamics = integrators.euler(car_ode, dt=dt)

    # Constants
    n, m, T = (4, 2, 40)

    # Indices of state corresponding to S1 sphere constraints
    s1_indices = (2,)
    state_wrap = util.get_s1_wrapper(s1_indices)


    # Cost function.
    R = jnp.diag(jnp.array([0.2, 0.1]))
    Q_T = jnp.diag(jnp.array([100., 100., 50., 10.]))
    goal_default = jnp.array([3., 3., jnp.pi/2, 0.])

    @jax.jit
    def cost(x, u, t, goal=goal_default):
      stage_cost = dt * jnp.vdot(u, R @ u)
      delta = state_wrap(x - goal)
      term_cost = jnp.vdot(delta, Q_T @ delta)
      return jnp.where(t == T, term_cost, stage_cost)

    # Control box bounds
    control_bounds = (jnp.array([-jnp.pi/3., -6.]),
                      jnp.array([jnp.pi/3., 6.]))

    # Obstacle avoidance constraint function
    def obs_constraint(pos):
      def avoid_obs(pos_c, ob):
        delta_body = pos_c - ob[0]
        delta_dist_sq = jnp.vdot(delta_body, delta_body) - (ob[1]**2)
        return delta_dist_sq
      return jnp.array([avoid_obs(pos, ob) for ob in obs])

    # State constraint function
    @jax.jit
    def state_constraint(x, t):
      del t
      pos = x[0:2]
      return obs_constraint(pos)

    solver_options = dict(method=shootsqp.SQP_METHOD.SENS,
                      ddp_options={'ddp_gamma': 1e-4},
                      hess="full", verbose=True,
                      max_iter=100, ls_eta=0.49, ls_beta=0.8,
                      primal_tol=1e-3, dual_tol=1e-3, stall_check="abs",
                      debug=False)
    solver = shootsqp.ShootSQP(n, m, T, dynamics, cost, control_bounds,
                           state_constraint, s1_ind=s1_indices, **solver_options)

    x0 = jnp.array([1.75, 1.0, 0., 0.])
    U0 = jnp.zeros((T, m))
    X0 = None
    solver.opt.proj_init = False

    # guess with waypoints 
    solver.opt.proj_init = True
    #waypoints = jnp.array([
    #    x0[:2], jnp.array([1.75, 3.0]), goal_default[:2]
    #])
    avg = (x0[:2] + goal_default[:2]) / 2 
    waypoints = jnp.array([
        x0[:2], avg, goal_default[:2]
    ])
    X0 = jnp.concatenate((
        jnp.linspace(waypoints[0], waypoints[1], int(T//2)),
        jnp.linspace(waypoints[1], waypoints[2], int(T//2) + 2)[1:]
    ))
# Augment with zeros
    X0 = jnp.hstack((X0, jnp.zeros((T+1, 2))))

    solver.opt.max_iter = 1
    _ = solver.solve(x0, U0, X0)
    solver.opt.max_iter = 100
    soln = solver.solve(x0, U0, X0)

    fig, ax = render_scene(obs)
    U, X = soln.primals
    ax.plot(X[:, 0], X[:, 1], 'k-', linewidth=1)

    for t in jnp.arange(0, solver._T+1, 3):
      x = X[t, :2]
      dx = jnp.array([0.2 * jnp.sin(X[t, 2]), 0.2 * jnp.cos(X[t, 2])])
      y = x + dx 
      L = jnp.linalg.norm(y - x) 
      heading = jnp.arccos(dx[0] / L)
      #ax.arrow(X[t, 0], X[t, 1],
      #    0.2 * jnp.sin(X[t, 2]), 0.2 * jnp.cos(X[t, 2]),
      #    width=0.05, color='c')
      make_car(np.array([x[0], x[1], heading]), np.array([U[t, 1], -U[t, 0]]), ax)

# Start
    ax.add_patch(plt.Circle([x0[0], x0[1]], 0.1, color='g', alpha=0.3))
# End
    ax.add_patch(plt.Circle([goal_default[0], goal_default[1]], 0.1, color='r', alpha=0.3))

    ax.set_aspect('equal')
    plt.savefig("solution") 
    plt.close()





if __name__=="__main__": 
    config.update('jax_enable_x64', True)
    main() 
