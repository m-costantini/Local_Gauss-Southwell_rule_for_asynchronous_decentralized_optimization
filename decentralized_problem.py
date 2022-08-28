"""
Create decentralized problem where the f_i of each node is a quadratic 
"""

import numpy as np
from scipy.linalg import sqrtm

class Problem():

    def __init__(self, n, dim, A, edge_to_nodes, iid, degree):

        super(Problem, self).__init__()
        self.n = n
        self.dim = dim
        self.A = A
        self.edge_to_nodes = edge_to_nodes
        self.E = np.shape(A)[0]
        # Generate the data
        self.P_vec = [None] * n
        self.Q_vec = [None] * n
        self.R_vec = [None] * n

        self.P_inv = [None] * n
        self.PP_inv = [None] * n
        self.P_inv_sq = [None] * n

        # Give all nodes the same distribution, just change random matrix aux
        for i in range(n):

            # Create quadratic matrix
            aux_0 = np.random.uniform(low=-0.01, high=0.01, size=(dim,dim))
            if np.mod(i,degree) == 0:
                c = 50
            else:
                c = 1
            P = aux_0 @ aux_0.T + c * np.eye(dim) # add identity to increase eigenvalues and give more curvature

            eig_vals, _ = np.linalg.eig(P)
            if np.any(eig_vals < 0):
                raise Exception('Eigenvalues of created matrix are smaller than 0')

            # Create linear term and offset
            Q = np.zeros((dim,1))
            R = 1 # just for f(x^*) not to be 0 and have problems in the division

            self.P_vec[i] = P
            self.Q_vec[i] = Q
            self.R_vec[i] = R

            self.P_inv[i] = np.linalg.inv(P)
            self.PP_inv[i] = 0.5 * self.P_inv[i] # = np.linalg.inv(P + P.T)
            self.P_inv_sq[i] = sqrtm(self.P_inv[i])

        # All together
        Psum = sum(self.P_vec)
        Qsum = sum(self.Q_vec)
        Rsum = sum(self.R_vec)

        # Compute optimal objective value analytically
        theta_opt = np.linalg.inv(Psum + Psum.T) @ (-Qsum)
        self.analy_opt_obj_val = self.quadratic_function(theta_opt, Psum, Qsum, Rsum).flatten()

    # ----------------------------------------------------------------------------------
    # Functions

    def quadratic_function(self, theta, P, Q, R):
        theta = np.reshape(theta,(len(theta),1))
        return theta.T @ P @ theta + Q.T @ theta + R

    def objective(self, thetas):
        obj = 0
        for i in range(len(self.P_vec)):
            obj += self.quadratic_function(thetas[i,:], self.P_vec[i], self.Q_vec[i], self.R_vec[i])
        return obj

    def Lagrangian(self, thetas, lambdas):
        obj = self.objective(thetas)
        relax = 0
        for e in range(self.E):
            (i,j) = self.edge_to_nodes[e]
            relax += lambdas[e,:].reshape((1,self.dim)) @ ( self.A[e,i]*thetas[i,:] + self.A[e,j]*thetas[j,:] ).reshape((self.dim,1))
        return obj + relax

    def arg_min_Lagran(self, i, lambdas):
        return self.PP_inv[i] @ ( - self.A[:,i].T @ lambdas - self.Q_vec[i].flatten() )

    def get_optimal_stepsizes(self):
        opt_sz = [None] * self.E
        for k in range(self.E):
            (i,j) = self.edge_to_nodes[k]
            Ql = 0.5 * (self.P_inv[i] + self.P_inv[j])
            eig_vals, _ = np.linalg.eig(Ql)
            opt_sz[k] = 1/max(eig_vals)
        return opt_sz
