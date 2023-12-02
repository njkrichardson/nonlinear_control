import argparse
from pathlib import Path
from typing import List, Tuple
import time 
import sys 

from jax.config import config
import jax.numpy as np
import jax
import jax.random as npr
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from trajax.integrators import euler
from trajax.experimental.sqp import shootsqp, util

from constants import ndarray 
from visuals import render_scene, make_car 
from utils import setup_logger, serialize, setup_experiment_directory

parser = argparse.ArgumentParser()
parser.add_argument("--num-tries", default=10, type=int)
parser.add_argument("--slurm_id", default=0, type=int)

def car_dynamics(x: ndarray, u: ndarray, t: int) -> ndarray:
  del t
  return np.array([x[3] * np.sin(x[2]), x[3] * np.cos(x[2]), x[3] * u[0], u[1]])

def sample_obstacles(key: ndarray, num_obstacles: int, world_range: Tuple[float, float]=(0., .5)) -> ndarray: 
    x_key, y_key = npr.split(key) 
    x = npr.uniform(x_key, (num_obstacles,), minval=-0.5, maxval=1.5)
    y = npr.uniform(y_key, (num_obstacles,), minval=-0.25, maxval=0.25) 
    obstacles = np.vstack((x, y)).T
    return obstacles

def run_control(key: ndarray) -> Tuple[ndarray, ndarray]: 
    obstacles_key, goal_key = npr.split(key) 
    num_obstacles: int = 3
    obstacles: ndarray = sample_obstacles(obstacles_key, num_obstacles)
    obstacle_size: float = 0.2 

    # configure dynamics 
    dt: float = 0.05
    dynamics = euler(car_dynamics, dt=dt)

    # constants
    num_timesteps: int = 40 
    state_dimension: int = 4 
    control_dimension: int = 2
    n, m, T = (4, 2, 40)

    # indices of state corresponding to S1 sphere constraints
    s1_indices: Tuple[int] = (2,)
    state_wrap = util.get_s1_wrapper(s1_indices)

    # cost function.
    R: ndarray = np.diag(np.array([0.2, 0.1]))
    Q_T: ndarray = np.diag(np.array([100., 100., 50., 10.]))

    goal_location_key, goal_orientation_key, initial_position_key = npr.split(goal_key, 3) 

    goal_x_key, goal_y_key = npr.split(goal_location_key)
    goal_location: ndarray = np.concatenate((npr.uniform(goal_x_key, (1,), minval=0., maxval=1.) , npr.uniform(goal_y_key, (1,), minval=1., maxval=1.2)))
    goal_orientation: float = npr.uniform(goal_orientation_key, (1,), minval=0., maxval=np.pi/2)
    goal_default: ndarray = np.concatenate((goal_location, goal_orientation, np.zeros(1)))

    @jax.jit
    def cost(x: ndarray, u: ndarray, t: int, goal=goal_default) -> float:
        stage_cost: ndarray = dt * np.vdot(u, R @ u)
        delta: ndarray = state_wrap(x - goal)
        term_cost: ndarray = np.vdot(delta, Q_T @ delta)
        return np.where(t == num_timesteps, term_cost, stage_cost)

    # control box bounds
    control_bounds: Tuple[ndarray] = (np.array([-np.pi/3., -6.]), np.array([np.pi/3., 6.]))

    # obstacle avoidance constraint 
    def obs_constraint(position: ndarray) -> ndarray:
        def avoid_obs(pos_center: ndarray, obstacle: ndarray) -> float:
            delta_body = pos_center - obstacle
            #delta_dist_sq = np.vdot(delta_body, delta_body) - (obstacle_size**2)
            delta_dist_sq = np.vdot(delta_body, delta_body)
            return delta_dist_sq
        constraint = np.array([avoid_obs(position, ob) for ob in obstacles]) 
        return constraint

    # state constraint function
    @jax.jit
    def state_constraint(x: ndarray, t: int) -> float:
      del t
      position: ndarray = x[0:2]
      return obs_constraint(position)

    solver_options = dict(method=shootsqp.SQP_METHOD.SENS,
                      ddp_options={'ddp_gamma': 1e-4},
                      hess="full", verbose=False,
                      max_iter=100, ls_eta=0.49, ls_beta=0.8,
                      primal_tol=1e-3, dual_tol=1e-3, stall_check="abs",
                      debug=False)
    solver = shootsqp.ShootSQP(state_dimension, control_dimension, num_timesteps, dynamics, cost, control_bounds, state_constraint, s1_ind=s1_indices, **solver_options)

    init_pos_x, init_pos_y = npr.split(initial_position_key)
    x0 = np.concatenate((npr.uniform(init_pos_x, (1,), minval=0., maxval=1.), npr.uniform(init_pos_y, (1,), minval=-1., maxval=-.8)))
    x0 = np.concatenate((x0, np.zeros(2)))
    U0 = np.zeros((num_timesteps, control_dimension))

    # guess with waypoints 
    solver.opt.proj_init = True
    avg = (x0[:2] + goal_default[:2]) / 2 
    waypoints = np.array([
        x0[:2], avg, goal_default[:2]
    ])
    X0 = np.concatenate((
        np.linspace(waypoints[0], waypoints[1], int(T//2)),
        np.linspace(waypoints[1], waypoints[2], int(T//2) + 2)[1:]
    ))

    # augment with zeros
    X0 = np.hstack((X0, np.zeros((T+1, 2))))

    solver.opt.max_iter = 1
    _ = solver.solve(x0, U0, X0)

    solver.opt.max_iter = 100
    soln = solver.solve(x0, U0, X0)
    U, X = soln.primals

    return avg, solver, x0, goal_default, obstacles, U, X

def main(args): 
    experiment_directory = setup_experiment_directory("car_control")
    log = setup_logger(__name__, custom_handle=experiment_directory / "log.out")

    for t in range(args.num_tries): 
        # run control 
        key: ndarray = npr.PRNGKey(int(time.time()), + int(args.slurm_id))
        avg, solver, x0, goal_default, obstacles, U, X = run_control(key)

        final_error: float = np.linalg.norm(X[-1, :2] - goal_default[:2])
        if final_error >= 0.2: 
            log.info(f"Poor optimization, not saving")
            break 

        result = dict(
                key=key, 
                avg=avg, 
                T=solver._T, 
                x0=x0, 
                goal_default=goal_default, 
                obstacles=obstacles, 
                U=U, 
                X=X
                )
        serialize(result, (experiment_directory / f"result_{t}").as_posix())

        fig, ax = render_scene(obstacles)
        ax.plot(X[:, 0], X[:, 1], 'k-', linewidth=1)

        for t in np.arange(0, solver._T+1, 3):
          x = X[t, :2]
          dx = np.array([0.2 * np.sin(X[t, 2]), 0.2 * np.cos(X[t, 2])])
          y = x + dx 
          L = np.linalg.norm(y - x) 
          heading = np.arccos(dx[0] / L)
          make_car(np.array([x[0], x[1], heading]), np.array([U[t, 1], -U[t, 0]]), ax)

        ax.scatter(avg[0], avg[1], c="tab:red")
        
        # start
        ax.add_patch(plt.Circle([x0[0], x0[1]], 0.1, color='tab:blue', alpha=0.5))
        # end
        ax.add_patch(plt.Circle([goal_default[0], goal_default[1]], 0.1, color='r', alpha=0.5))

        ax.set_aspect('equal')
        plt.savefig(experiment_directory / f"solution_{t}") 
        plt.close()

if __name__=="__main__": 
    config.update('jax_enable_x64', True)
#    config.update('jax_disable_jit', True)
    args = parser.parse_args()
    main(args) 
