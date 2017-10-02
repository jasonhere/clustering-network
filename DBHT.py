import cPickle as pickle
import matlab.engine
import progressbar
import networkx as nx
import igraph as ig
import numpy as np
import math

def Cluster_labelToList(cluster_labels, nodenames):
    result = [[] for i in range(max(cluster_labels))]
    for i in range(len(cluster_labels)):
        result[cluster_labels[i]-1].append(nodenames[i])
    return result

def construct_nxPMFG(PMFG_matrix, nodenames, method = 'gower'):
    corr_matrix = PMFG_matrix
    corr_matrix[np.where(PMFG_matrix != 0)] = abs(PMFG_matrix[np.where(PMFG_matrix != 0)] -1)
    # distance_matrix[np.where(PMFG_matrix != 0)] = (2 * (1- distance_matrix[np.where(PMFG_matrix != 0)]))**0.5
    G = nx.from_numpy_matrix(corr_matrix)
    mapping = dict(zip(list(range(len(nodenames))), list(nodenames)))
    G = nx.relabel_nodes(G, mapping, copy=False)
    # delete NAN weights
    for (u, v, d) in G.edges(data=True):
        if math.isnan(d["weight"]):
            G.remove_edges_from([(u, v)])
    # delete self-connected edges
    for (u, v, d) in G.edges(data=True):
        if u == v:
            G.remove_edges_from([(u, v)])
    # delete nodes whose degree is 0
    nodes = list(G.nodes())
    for node in nodes:
        if G.degree(node) == 0:
            G.remove_node(node)
    length_dict = {}
    if method == "gower":
        for edge in G.edges(data=True):
            length_dict[(edge[0],edge[1])] = (2 - 2 * edge[2]['weight']) ** 0.5  # gower
    elif method == "power":
        for edge in G.edges(data=True):
            length_dict[(edge[0],edge[1])]=1 - edge[2]['weight'] ** 2   # power
    nx.set_edge_attributes(G,'length',length_dict)
    weight_plus_one_dict = {}
    for edge in G.edges(data=True):
        weight_plus_one_dict[(edge[0],edge[1])] = 1+edge[2]['weight']
    nx.set_edge_attributes(G,'weight+1',weight_plus_one_dict)
    return G

def DBHT(corr_dict):
    clusterings = {}
    PMFG = {}
    bubble_cluster_membership = {}
    PMFG_shortest_path_length_matrix = {}
    bubble_membership_mattrix = {}
    DBHT_hierarchy = {}
    eng = matlab.engine.start_matlab()
    eng.cd(r'MATLAB')
    print("Computing DBHTs...")
    count = 0
    bar = progressbar.ProgressBar(max_value=len(corr_dict.keys()))
    for key in sorted(corr_dict.keys()):
        key = key[-10:]
        count+=1
        A = corr_dict[key]
        D = (2 * (1 - A)) ** 0.5
        S = A + 1
        D = matlab.double(D.as_matrix().tolist())
        S = matlab.double(S.as_matrix().tolist())
        T8, Rpm, Adjv, Dpm, Mv, Z = eng.DBHTs(D, S, nargout=6)
        clusterings[key] = Cluster_labelToList([int(x[0]) for x in np.array(T8)], corr_dict[key].columns.values)
        PMFG[key] = construct_nxPMFG(np.array(Rpm),corr_dict[key].columns.values)
        bubble_cluster_membership[key] = np.array(Adjv)
        PMFG_shortest_path_length_matrix[key] = np.array(Dpm)
        bubble_membership_mattrix[key] = np.array(Mv)
        DBHT_hierarchy[key] = Z
        bar.update(count)

    return {'DBHT_clusterings': clusterings, 'PMFG': PMFG,
            'bubble_cluster_membership_matrix': bubble_cluster_membership,
            'PMFG_shortest_path_length_matrix': PMFG_shortest_path_length_matrix,
            'bubble_membership_matrix': bubble_membership_mattrix,
            'DBHT_hierarchy': DBHT_hierarchy}
