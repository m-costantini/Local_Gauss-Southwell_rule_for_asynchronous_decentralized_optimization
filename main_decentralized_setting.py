"""
Main code to compare the Set-wise Coordinate Descent algorithms of the paper

M. Costantini, N. Liakopoulos, P. Mertikopoulos, and T. Spyropoulos, “Pick your neighbor: Local Gauss-Southwell rule for fast asynchronous decentralized optimization,” in 61st IEEE Conference on Decision and Control (CDC), 2022.

in the *decentralized* setting.

Marina Costantini, July 2022
marina.costant@gmail.com
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from functions import *
from decentralized_problem import Problem
from decentralized_solvers import Solver

print('\nRunning code for comparison between SU-CD and SGS-CD in two different graphs\n')

# create graphs
n = 24
G_list = []
graph_names = []
deg_list = [8,12]

steps_list = [8000 for i in range(len(deg_list))]

for deg in deg_list:
    G_list.append( make_graph(n, 'WS', deg, 0) )
    graph_names.append("Watts-Strogatz")

# parameter settings
algos_to_run = ['SU-CD', 'SGS-CD']
dim = 5

# create problem for each graph and run algorithms
results_primal = [None for i in range(len(G_list))]
results_dual = [None for i in range(len(G_list))]
for idx_G, G in enumerate(G_list):
    print("Running graph", idx_G+1, "/", len(G_list))
    max_degree = np.max(np.sum(G,axis=1)) # maximum degree in the network
    A = get_edge_matrix(G) # incidence matrix
    E = np.shape(A)[0] # number of edges
    role, neighbors, N = create_node_variables(A)
    mut_idcs, mat_edge_idxs, edge_to_nodes = create_edge_variables(A, neighbors)
    # generate problem
    the_problem = Problem(n, dim, A, edge_to_nodes, iid=False, degree=deg_list[idx_G])
    opt_sz = the_problem.get_optimal_stepsizes() # compute optimal stepsize based on coordinate smoothness
    significant_digits = 1 # round steospize to one significant digit
    sz =  min(opt_sz) # use the largest stepsize that guarantes convergence for *all* coordinates/edges
    alpha = round(sz, significant_digits - int(math.floor(math.log10(abs(sz)))) - 1)
    # store simulation setings
    simu_vars = {'dim': dim,
                'n': n,
                'E': E,
                'A': A,
                'role': role,
                'neighbors': neighbors,
                'N': N,
                'mat_edge_idxs': mat_edge_idxs,
                'edge_to_nodes': edge_to_nodes,
                'alpha': alpha,
                'mut_idcs': mut_idcs,
                'steps': steps_list[idx_G]}
    # run algorithms
    primal_obj_vals = {}
    dual_obj_vals = {}
    for algo in algos_to_run:
        print("\tRunning ", algo)
        the_solver = Solver(the_problem, simu_vars, algo)
        primal_obj_vals[algo], dual_obj_vals[algo] = the_solver.solve()
    results_primal[idx_G] = primal_obj_vals
    results_dual[idx_G] = dual_obj_vals

# plot settings
fontsz = {'legends':14, 'axes':16, 'titles':18, 'suptitles':20}
figsz = (6*len(G_list), 4)
color = {'SU-CD':'blue', 'SGS-CD':'red'}
linestyle = {'SU-CD':'-', 'SGS-CD':'-'}
algo_labels = {'SU-CD':'SU-CD', 'SGS-CD':'SGS-CD'}
rho_strs = {'SU-CD': r", $\rho_U$ =", 'SGS-CD': r", $\rho_G$ ="}
dpi_paper = 150

# use the last third of the convergence curves to estimate the ratio of the convergence rate
plot_fitting = False
if plot_fitting:
    fig, ax = plt.subplots(1, len(G_list), figsize=figsz)
m_list = [[None for i in range(len(G_list))] for j in range(len(algos_to_run))]
b_list = [[None for i in range(len(G_list))] for j in range(len(algos_to_run))]
sel_idcs_list = [[None for i in range(len(G_list))] for j in range(len(algos_to_run))]
for idx_G in range(len(G_list)):
    dual_obj_vals = results_dual[idx_G]
    for idx_algo, algo in enumerate(algos_to_run):
        dual_vals = dual_obj_vals[algo]
        subopt = abs(1 - dual_vals/the_problem.analy_opt_obj_val)
        idcs_nz_subopt = np.argwhere(subopt > 0).flatten()
        non_zero_subopt = subopt[idcs_nz_subopt]
        third_steps = int(len(non_zero_subopt)/3)

        sel_idcs_subopt = idcs_nz_subopt[2*third_steps:]
        sel_subopt_vals = non_zero_subopt[2*third_steps:]

        m, b = np.polyfit(sel_idcs_subopt, np.log(sel_subopt_vals), 1)
        if plot_fitting:
            ax[idx_G].plot(sel_idcs_subopt, np.log(sel_subopt_vals), color=color[algo], linestyle=linestyle[algo], label=algo)
            ax[idx_G].plot( sel_idcs_subopt, m*sel_idcs_subopt+b, 'k--')

        m_list[idx_algo][idx_G] = m
        b_list[idx_algo][idx_G] = b
        sel_idcs_list[idx_algo][idx_G] = sel_idcs_subopt

# plot
fig, ax = plt.subplots(1, len(G_list), figsize=figsz, dpi=70)
rate_list = [[None for i in range(len(G_list))] for j in range(len(algos_to_run))]
for idx_G in range(len(G_list)):
    dual_obj_vals = results_dual[idx_G]
    for idx_algo, algo in enumerate(algos_to_run):
        rate = 1 - np.exp(m_list[idx_algo][idx_G])
        suboptim = abs(1 - dual_obj_vals[algo]/the_problem.analy_opt_obj_val)
        ax[idx_G].plot(suboptim, color=color[algo], linestyle=linestyle[algo], label=algo_labels[algo] + rho_strs[algo] + "{:.4f}".format(rate) )
        support = sel_idcs_list[idx_algo][idx_G]
        subopt = suboptim[support[0]]
        support_2 = np.linspace(1, len(support), len(support))
        ax[idx_G].plot(support, (1-rate)**support_2 * subopt, color=color[algo], linewidth=8, alpha=0.3)
        rate_list[idx_algo][idx_G] = rate
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
for idx_G in range(len(G_list)):
    ax[idx_G].set_xlabel('Iteration index', fontsize=fontsz['axes']-2)
    ax[idx_G].set_ylabel(r'$ \mid 1 - \frac{F^k}{F^*} \mid $', fontsize=fontsz['axes']+2)
    ax[idx_G].yaxis.set_label_coords(-0.09,0.5)
    ax[idx_G].grid()
    ax[idx_G].set_yscale("log")
    title_str = 'Decentralized, ' + str(deg_list[idx_G]) + '-regular graph, ' + r'$n = $' + str(n)
    ax[idx_G].set_title(title_str, fontsize=fontsz['titles'])
    ax[idx_G].legend(fontsize=fontsz['legends'])
    textstr = r'$\frac{\rho_G}{\rho_U}=%.2f$' % (rate_list[1][idx_G]/rate_list[0][idx_G], )
    ax[idx_G].text(0.7, 0.68, textstr, transform=ax[idx_G].transAxes, fontsize=fontsz['legends']+2,
        verticalalignment='top', bbox=props)
fig.subplots_adjust(
    left  = 0.125,  # the left side of the subplots of the figure
    right = 0.9,    # the right side of the subplots of the figure
    bottom = 0.05,   # the bottom of the subplots of the figure
    top = 0.85,      # the top of the subplots of the figure
    wspace = 0.25,   # the amount of width reserved for blank space between subplots
    hspace = 0.2)   # the amount of height reserved for white space between subplots
plt.show()
