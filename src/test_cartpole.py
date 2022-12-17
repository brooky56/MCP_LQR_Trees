from controllers.lqr_tree import LQRTree
import numpy as np
from models.cartpole import symbolic_dynamics, dynamics, linearize_dynamics
import pickle
import matplotlib.pyplot as plt

xmax = np.array([3, np.pi, 1, 1])
nx = 4
nu = 1
xgoal = np.array([0, np.pi, 0, 0])
ugoal = np.array([0])
ulb = -5*np.ones(1)
uub = 5*np.ones(1)
dt = 0.1
branch_horizon = 50
tree = LQRTree(xmax, symbolic_dynamics, dynamics, linearize_dynamics, nx, nu, xgoal, ugoal, ulb, uub, dt, branch_horizon)
tree.build_tree()
xlabel = 'x'
ylabel = 'theta'
tree.plot_all_funnels(xlabel, ylabel, 'cartpole_funnels.png')

# Set Initial Conditions
x_init = [0.05, 1.2, 0, 0]

xs, us = tree.trace(np.array(x_init), xlabel, ylabel, 'cartpole_trace.png')
plt.figure()
plt.plot([x[0] for x in xs])
plt.figure()
plt.plot([x[1] for x in xs])
plt.figure()
plt.plot([x[2] for x in xs])
plt.figure()
plt.plot([x[3] for x in xs])
plt.figure()
plt.plot(us)
plt.show()
