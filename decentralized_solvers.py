"""
Implement set-wise coordinate descent solver in their two variations: uniform and Gauss Southwell
"""

import numpy as np
from random import choice

class Solver():

    def __init__(self, the_problem, simu_vars, solver_name):

        self.the_problem = the_problem
        self.solver_name = solver_name

        self.dim = simu_vars['dim']
        self.n = simu_vars['n']
        self.E = simu_vars['E']
        self.A = simu_vars['A']
        self.role = simu_vars['role']
        self.neighbors = simu_vars['neighbors']
        self.N = simu_vars['N']
        self.mat_edge_idxs = simu_vars['mat_edge_idxs']
        self.edge_to_nodes = simu_vars['edge_to_nodes']
        self.alpha = simu_vars['alpha']
        self.mut_idcs = simu_vars['mut_idcs']
        self.steps = simu_vars['steps']

        # Variables needed only by all solvers
        self.thetas = np.ones(shape=(self.n,self.dim))
        self.lambdas = 10 * np.ones(shape=(self.E,self.dim))
        self.obj = np.zeros((self.steps,))
        self.dual = np.zeros((self.steps,))



    def choose_neighbor(self, i):
        if self.solver_name == 'SU-CD':
            idx_j = np.random.randint(self.N[i]) # random uniform selection
        elif self.solver_name == 'SGS-CD':
            # NOTE: we unnecesarily repeat the computation of theta_i and theta_j afterwards for this algorithm, but it's the price of recycling code
            theta_i = self.the_problem.arg_min_Lagran(i, self.lambdas).flatten() # compute theta of activated node
            # Ask the parameters of all of our neighbors to compute the largest gradient
            theta_neighs = np.zeros((self.N[i],self.dim))
            for idx_k, k in enumerate(self.neighbors[i]): # weighted already by the corresponding A[e,k]
                # theta_neighs[idx_k,:] = self.the_problem.PP_inv[k] @ ( - self.A[:,k].T @ self.lambdas - self.the_problem.Q_vec[k].flatten() )
                theta_neighs[idx_k,:] = self.the_problem.arg_min_Lagran(k, self.lambdas) # changed on 04/02/2022. Should allow loop to work for any problem
            # Compute magnitude of the gradients and choose neighbor that has the largest
            grads_i = self.role[i] @ np.expand_dims(theta_i, axis=0) - np.tile(self.role[i],(1,self.dim)) * theta_neighs
            mag_grads_i = np.linalg.norm(grads_i, axis=1)
            idx_j = np.where( mag_grads_i == np.max(mag_grads_i) )[0]
            if hasattr(idx_j, "__len__"): # tie breaking - random unifrom choice among possibilities
                idx_j = choice(idx_j)
        else:
            raise Exception("Invalid solver name")
        j = self.neighbors[i][idx_j]
        e = self.mat_edge_idxs[i,j]
        idx_i = self.mut_idcs[j][i]
        return idx_j, j, e, idx_i



    def solve(self,):
        for t in range(self.steps):
            # Activated nodes
            i = np.random.randint(self.n) # random node goes active
            idx_j, j, e, idx_i = self.choose_neighbor(i) # choose contacted neighbor
            # Compute primal variables that minimize the Lagrangian
            theta_i = self.the_problem.arg_min_Lagran(i, self.lambdas).flatten()
            theta_j = self.the_problem.arg_min_Lagran(j, self.lambdas).flatten()
            self.thetas[i,:] = theta_i
            self.thetas[j,:] = theta_j
            # Compute primal and dual function value
            self.obj[t] = self.the_problem.objective(self.thetas)
            self.dual[t] = self.the_problem.Lagrangian(self.thetas, self.lambdas)
            if self.obj[t] > 10**6: # thing is diverging !
                print(' --> Objective > breaking_thresh @ iteration', t, '--> break', end='\r')
                break
            # Update lambdas
            self.lambdas[e,:] = self.lambdas[e,:] + self.alpha * ( self.A[e,i] * theta_i + self.A[e,j] * theta_j )
            if abs(1 - self.obj[t]/self.the_problem.analy_opt_obj_val) < 10**(-9): # converged! stop!
                print(' --> Precision reached @ iteration',t ,'--> leave!')
                self.obj = self.obj[:t]
                self.dual = self.dual[:t]
                break
        return self.obj, self.dual
