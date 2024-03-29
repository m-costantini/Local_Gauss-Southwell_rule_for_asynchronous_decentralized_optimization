"""
Implement set-wise coordinate descent solvers (uniform and Gauss Southwell) for the parallel distributed problem problem

Initialize the starting point to get the most benefit from the Gauss-Southwell choice
"""

import numpy as np
from random import choice

class Distrib_solver():

    def __init__(self, the_problem, solver_name, coord_sets, non_zero_list, steps, alpha):

        self.the_problem = the_problem
        self.solver_name = solver_name
        self.coord_sets = coord_sets
        self.steps = steps
        self.alpha = alpha

        # Generate extra variables not s=explicitly given
        self.n = len(coord_sets) # number of sets
        E = np.shape(the_problem.P)[0] # number of coordiantes

        # Initialize parameter vector
        self.x = np.ones((E,))
        self.x[non_zero_list] = 100
        self.obj = np.zeros((steps,))



    # ----------------------------------------------------------------------------------
    # Functions

    def choose_coordinate(self, i):
        if self.solver_name == 'SU-CD':
            idx_e = np.random.randint(len(self.coord_sets[i])) # random uniform selection
            set_grads = self.the_problem.get_set_grads(self.x, i).flatten()
        elif self.solver_name == 'SGS-CD':
            set_grads = self.the_problem.get_set_grads(self.x, i).flatten()
            abs_set_grads = abs(set_grads)
            idx_e = np.where( abs_set_grads == np.max( abs_set_grads ) )[0]
            if hasattr(idx_e, "__len__"): # tie breaking - random unifrom choice among possibilities
                idx_e = choice(idx_e)
        else:
            raise Exception("Invalid solver name")
        e = self.coord_sets[i][idx_e]
        e_grad = set_grads[idx_e]
        return e, e_grad



    def solve(self,):
        for t in range(self.steps):
            i = np.random.randint(self.n) # random set goes active
            e, e_grad = self.choose_coordinate(i) # choose coordinate
            self.x[e] = self.x[e] - self.alpha * e_grad
            self.obj[t] = self.the_problem.objective(self.x)
            if self.obj[t] > 10**10: # thing is diverging !
                print(' --> Objective > breaking_thresh @ iteration', t ,'--> break')
                break
            if abs(1 - self.obj[t]/self.the_problem.opt_val) < 10**(-15): # converged! stop!
                print(' --> Precision reached @ iteration', t, '--> leave!')
                self.obj = self.obj[:t]
                break
        return self.obj
