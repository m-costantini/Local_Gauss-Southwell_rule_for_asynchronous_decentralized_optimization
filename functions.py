"""
Auxiliary functions
"""

import numpy as np
import warnings
from scipy.linalg import sqrtm
import networkx as nx



def random_graph(n,p):
    G = nx.fast_gnp_random_graph(n,p)
    A = np.array( nx.to_numpy_matrix(G) )
    return A

def power_law_graph(n,m):
    G = nx.barabasi_albert_graph(n, m)
    A = np.array( nx.to_numpy_matrix(G) ).astype(int)
    return A

def watts_strogatz_graph(n,d,p):
    G = nx.watts_strogatz_graph(n,d,p)
    A = np.array( nx.to_numpy_matrix(G) ).astype(int)
    return A

def grid_graph(n):
    G = nx.grid_graph(dim=[n,n])
    A = np.array(nx.to_numpy_matrix(G))
    return A

def d_regular_random_graph(n,d):
    G = nx.random_regular_graph(d, n)
    A = np.array( nx.to_numpy_matrix(G) ).astype(int)
    return A

def make_graph(n, graph_name, *args):
    # Generate graph
    max_graph_apptempts = 100 # maximum number of tries allowed to generate a connected graph
    if graph_name == 'random':
        graph_fun = random_graph
    elif graph_name == 'power-law':
        graph_fun = power_law_graph
    elif graph_name == 'WS':
        graph_fun = watts_strogatz_graph
    elif graph_name == 'grid':
        graph_fun = grid_graph
    elif graph_name == 'd-regular':
        graph_fun = d_regular_random_graph
    else:
        raise Exception("graph name is not valid")
    G = graph_fun(n,*args)
    connected, _ = is_connected(G)
    counter = 0
    while not connected:
        counter += 1
        G = graph_fun(n,*args)
        connected, _ = is_connected(G)
        if counter == max_graph_apptempts:
            raise Exception('Could not generate connected graph in ' + str(max_graph_apptempts) + ' attempts')
            return
    return G



def is_connected(A):
    # Compute Laplacian to test if it is connected
    d = A.sum(axis=1)
    D = np.diag(d)
    L = D - A
    eig_L, _ = np.linalg.eig(L)
    n_connected_components = np.sum(eig_L < 1e-10)
    if n_connected_components == 1:
        connected = True
    else:
        connected = False
    return connected, n_connected_components



def get_edge_matrix(A):
    # Matrix of zeros with {1,-1} indicating each edge (direction is first node --> last node)
    # For N nodes with E edges returns an [E x N] matrix
    A = np.where(A != 0, 1, 0)
    A_UT = np.triu(A, k=0)
    e = np.sum(np.sum(A_UT)) # number of edges
    n = np.shape(A)[0] # number of nodes
    E = np.zeros((e,n))
    coords_edges = np.nonzero(A_UT)
    E[range(e),coords_edges[0]] = 1 # first node, or leader
    E[range(e),coords_edges[1]] = -1 # second node, or follower
    return E



def create_node_variables(A):
    n = np.shape(A)[1]
    role = [None] * n # stores for each node a vector fo the length of its neighbors stating whether the node is leader (1) or follower (-1)
    neighbors = [None] * n # states for each node who are its neighbors
    N = [None] * n # number of neighbors of each node
    for i in range(n):
        edges_i = A[A[:,i]!=0,:] # edges concerning node i (selects rows of A)
        role[i] = np.expand_dims( np.copy(edges_i[:,i]), axis=1 ) # +1 or -1, tells us if node i is leader of follower of the edge
        edges_i[:,i] = 0 # delete values column i
        neighbors[i] = np.nonzero(edges_i)[1] # save indeces of neighbor nodes (column indeces of non-zero entries edges_i)
        N[i] = len(neighbors[i])
    return role, neighbors, N



def create_edge_variables(A,neighbors):
    n = np.shape(A)[1]
    E = np.shape(A)[0]
    mut_idcs = np.zeros((n,n)).astype(int) # save mutual indeces in matrix form: (row i, col j) holds the index of node j for node i
    mat_edge_idxs = np.zeros((n,n)).astype(int)
    edge_to_nodes = [None] * E
    for idx_row, row in enumerate(A):
        i = np.where(row == 1)[0][0]
        j = np.where(row == -1)[0][0]
        idx_j = np.where(neighbors[i] == j)[0][0]
        idx_i = np.where(neighbors[j] == i)[0][0]
        mat_edge_idxs[i][j] = idx_row
        mat_edge_idxs[j][i] = idx_row
        mut_idcs[i][j] = idx_j
        mut_idcs[j][i] = idx_i
        edge_to_nodes[idx_row] = (i,j)
    return mut_idcs, mat_edge_idxs, edge_to_nodes
