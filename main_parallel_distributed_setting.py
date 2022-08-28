"""
Main code to compare the Set-wise Coordinate Descent algorithms of the paper

M. Costantini, N. Liakopoulos, P. Mertikopoulos, and T. Spyropoulos, “Pick your neighbor: Local Gauss-Southwell rule for fast asynchronous decentralized optimization,” in 61st IEEE Conference on Decision and Control (CDC), 2022.

in the *parallel distributed* setting.

Marina Costantini, July 2022
marina.costant@gmail.com
"""

import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from functions import *
from parallel_distributed_problem import Distrib_problem
from parallel_distributed_solvers import Distrib_solver

print('\nRunning code for comparison between SU-CD and SGS-CD in the parallel distributed setting \n')

# create dummy graph to distribute coordiantes
param_list = [(12,8), (24,4)] # (sets/nodes, coords_per_set/degree)
G_list = []
for params in param_list:
    B = nx.watts_strogatz_graph(params[0], params[1], 0)
    G_list.append( np.array( nx.to_numpy_matrix(B) ).astype(int) )
# distribute coordinates in sets
A_list = []
for G in G_list:
    A_list.append( (get_edge_matrix(G)).astype(bool).T )
coord_set_list = []
for A in A_list:
    coord_set = []
    for row in A:
        coord_set.append( np.where(row)[0] )
    coord_set_list.append(coord_set)

# find the edges linking consecutive pairs of graphs - those will be the non-zero coordinates
non_zero_list = []
E_list = []
for G in G_list:
    A = get_edge_matrix(G)
    E_list.append( np.shape(A)[0] )
    role, neighbors, N = create_node_variables(A)
    _, mat_edge_idxs, _ = create_edge_variables(A, neighbors)
    n = np.shape(G)[0]
    non_zero = []
    for i in range(0, n, 2):
        non_zero.append( mat_edge_idxs[i,i+1] )
    non_zero_list.append(non_zero)
# check that E is the same for both coordinate distributions (otherwise there is a problem)
if not all(E == E_list[0] for E in E_list):
    print("WARNING !! --> Elemens in E_list are not all equal")

# Create quadratic problem
E = E_list[0]
quad_factor = np.random.normal(loc=10, scale=3, size=(E,))
if any(quad_factor <= 0):
    raise Exception("Negative quadratic coefficient !!")
P = np.diag(quad_factor)
min_sz = 1 / max(quad_factor)
significant_digits = 1
sz = 0.9 * min_sz
alpha = round(sz, significant_digits - int(math.floor(math.log10(abs(sz)))) - 1)

# run algorithms
algos_to_run = ['SU-CD', 'SGS-CD']
results = [None for i in range(len(coord_set_list))]
steps = 10000
for idx_cs, coord_set in enumerate(coord_set_list):
    print("\n Running coordinate distribution", idx_cs+1, "/", len(coord_set_list))
    the_problem = Distrib_problem(coord_set, P)
    obj_vals = {}
    for algo_name in algos_to_run:
        print("\tRunning ", algo_name)
        the_solver = Distrib_solver(the_problem, algo_name, coord_set, non_zero_list[idx_cs], steps, alpha)
        obj_vals[algo_name] = the_solver.solve()
    results[idx_cs] = obj_vals

# plot settings
fontsz = {'legends':14, 'axes':16, 'titles':18, 'suptitles':20}
figsz = (6*len(coord_set_list), 5)
color = {'SU-CD':'blue', 'SGS-CD':'red'}
linestyle = {'SU-CD':'-', 'SGS-CD':'-'}
algo_labels = {'SU-CD':'SU-CD', 'SGS-CD':'SGS-CD'}
rho_strs = {'SU-CD': r", $\rho_U$ =", 'SGS-CD': r", $\rho_G$ ="}
dpi_paper = 150

# use the last third of the convergence curves to estimate the ratio of the convergence rate
plot_fitting = False
if plot_fitting:
    fig, ax = plt.subplots(1, len(G_list), figsize=figsz)
plot_fitting = False
if plot_fitting:
    fig, ax = plt.subplots(1, len(coord_set_list), figsize=figsz)
m_list = [[None for i in range(len(coord_set_list))] for j in range(len(algos_to_run))]
b_list = [[None for i in range(len(coord_set_list))] for j in range(len(algos_to_run))]
sel_idcs_list = [[None for i in range(len(coord_set_list))] for j in range(len(algos_to_run))]

for idx_cs in range(len(coord_set_list)):
    obj_vals = results[idx_cs]
    for idx_algo, algo in enumerate(algos_to_run):
        subopt = abs(1 - obj_vals[algo]/the_problem.opt_val)
        idcs_nz_subopt = np.argwhere(subopt > 0).flatten()
        non_zero_subopt = subopt[idcs_nz_subopt]
        third_steps = int(len(non_zero_subopt)/3)
        sel_idcs_subopt = idcs_nz_subopt[2*third_steps:]
        sel_subopt_vals = non_zero_subopt[2*third_steps:]
        m, b = np.polyfit(sel_idcs_subopt, np.log(sel_subopt_vals), 1)
        if plot_fitting:
            ax[idx_G].plot(sel_idcs_subopt, np.log(sel_subopt_vals), color=color[algo], linestyle=linestyle[algo], label=algo)
            ax[idx_G].plot( sel_idcs_subopt, m*sel_idcs_subopt+b, 'k--')
        m_list[idx_algo][idx_cs] = m
        b_list[idx_algo][idx_cs] = b
        sel_idcs_list[idx_algo][idx_cs] = sel_idcs_subopt

fig, ax = plt.subplots(1, len(G_list), figsize=figsz, dpi=70)
rate_list = [[None for i in range(len(coord_set_list))] for j in range(len(algos_to_run))]
for idx_cs in range(len(coord_set_list)):
    idx_ax = 1 - idx_cs
    obj_vals = results[idx_cs]
    for idx_algo, algo in enumerate(algos_to_run):
        rate = 1 - np.exp(m_list[idx_algo][idx_cs])
        suboptim = abs(1 - obj_vals[algo]/the_problem.opt_val)
        ax[idx_ax].plot(suboptim, color=color[algo], linestyle=linestyle[algo], label=algo + rho_strs[algo] + "{:.4f}".format(rate) )
        support = sel_idcs_list[idx_algo][idx_cs]
        subopt = suboptim[support[0]]
        support_2 = np.linspace(1, len(support), len(support))
        ax[idx_ax].plot(support, (1-rate)**support_2 * subopt, color=color[algo], linewidth=8, alpha=0.3)
        rate_list[idx_algo][idx_cs] = rate
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
for idx_cs in range(len(coord_set_list)):
    idx_ax = 1 - idx_cs
    ax[idx_ax].set_xlabel('Iteration index', fontsize=fontsz['axes']-2)
    ax[idx_ax].set_ylabel(r'$ \mid 1 - \frac{f^k}{f^*} \mid $', fontsize=fontsz['axes']+2)
    ax[idx_ax].yaxis.set_label_coords(-0.09,0.5)
    ax[idx_ax].grid()
    ax[idx_ax].set_yscale("log")
    params = param_list[idx_cs]
    title_str = 'Parallel Distributed, ' + r'$n = $' + str(params[0]) + r', $N_{max} =$ ' + str(params[1])
    ax[idx_ax].set_title(title_str, fontsize=fontsz['titles'])
    ax[idx_ax].legend(fontsize=fontsz['legends'])
    textstr = r'$\frac{\rho_G}{\rho_U}=%.2f$' % (rate_list[1][idx_cs]/rate_list[0][idx_cs], )
    ax[idx_ax].text(0.7, 0.68, textstr, transform=ax[idx_ax].transAxes, fontsize=fontsz['legends']+2,
        verticalalignment='top', bbox=props)
fig.subplots_adjust(
    left  = 0.125,  # the left side of the subplots of the figure
    right = 0.9,    # the right side of the subplots of the figure
    bottom = 0.05,   # the bottom of the subplots of the figure
    top = 0.85,      # the top of the subplots of the figure
    wspace = 0.25,   # the amount of width reserved for blank space between subplots
    hspace = 0.2)   # the amount of height reserved for white space between subplots
plt.show()
