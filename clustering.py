import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import time
from all_functions import importdata, MST, rolling_corr, weighted_degree_centrality
from scipy.stats import hypergeom
from sklearn.metrics import adjusted_rand_score


def NXtoIG(nxgraph):
    """Convert a networkx network to an igraph network"""
    edgelist = list(nx.to_edgelist(nxgraph))
    G = ig.Graph(len(list(nxgraph.nodes())))
    G.es["weight"] = 1.0
    G.vs["name"] = list(nxgraph.nodes())
    for i in range(0, len(edgelist)):
        G[edgelist[i][0], edgelist[i][1]] = edgelist[i][2]['weight']
    return G


def createlabel(clustering, names):
    """Label nodes in a clustering with their respective clusters"""
    i = 0
    labels = np.empty(len(names))
    labeldict = dict()
    for cluster in clustering:
        for node in cluster:
            labeldict[node] = i
        i = i + 1
    for i in range(0, len(names)):
        labels[i] = int(labeldict[names[i]])
    return labeldict, labels


def label_clusters(clustering1, clustering2, p_value=0.01):
    # type: (dict, list, float) -> dict
    """inputs: clustering1: dict of clusters,
        clustering2: list of clusters, method: 'HGT' or 'maxoverlap',
        p_value: the threshold for HGT.
    returns a clustering2 as a dictionary of clusters with similar clusters in the same order as clustering1"""
    N = 0
    for cluster in clustering1.values():
        N = N + len(cluster)
    result = {}
    ordered_ts = sorted(clustering1, key=lambda k: len(clustering1[k]), reverse=True)
    c2 = clustering2[:]
    c2.sort(key=len, reverse=True)
    for t in ordered_ts:
        if not c2:
            break
        for c in c2:
            overlap = len(set(clustering1[t]).intersection(c))
            p = hypergeom.pmf(overlap, N, len(c), len(clustering1[t]))
            if p < p_value:
                result[t] = c
                c2.remove(c)
                break
    lastindex = max(ordered_ts) + 1
    for c in c2:
        result[lastindex] = c
        lastindex = lastindex + 1
    return result


def total_clustering(enddate, startdate, filename="SP100_20170612.csv", method='Newman'):
    """helper to create a clustering with the whole period starting at startdate, ending at enddate, as the window.
    method can be 'Newman' or 'ClausetNewman' """
    df = importdata(filename)[1]
    end = int(np.where(df.index == enddate)[0])
    start = int(np.where(df.index == startdate)[0])
    total_tree = MST(filename=filename, window=end - start + 1,
                     enddate=enddate,
                     startdate=df.index[int(np.where(df.index == startdate)[0]) - 1].strftime('%Y-%m-%d'),
                     # here we pass the date one day before startdate to MST
                     space=1)[df.index[int(np.where(df.index == startdate)[0])].strftime('%Y-%m-%d')]
    IGtree = NXtoIG(total_tree)
    if method == 'Newman':
        C = IGtree.community_leading_eigenvector(weights="weight")
        clustersNewman = list(C)
        for i in range(0, len(C)):
            clustersNewman[i] = [IGtree.vs["name"][j] for j in C[i]]
        clustersNewman.sort(key=len, reverse=True)
        return {i + 1: clustersNewman[i] for i in range(len(clustersNewman))}
    elif method == 'ClausetNewman':
        C = IGtree.community_fastgreedy(weights="weight").as_clustering()
        clustersClausetNewman = list(C)
        for i in range(0, len(C)):
            clustersClausetNewman[i] = [IGtree.vs["name"][j] for j in C[i]]
        clustersClausetNewman.sort(key=len, reverse=True)
        return {i + 1: clustersClausetNewman[i] for i in range(len(clustersClausetNewman))}
    else:
        print("method can only be 'Newman' or 'ClausetNewman', your input is '%s'." % method)


def label_clustering_series(series, baseline_clustering=None, option='continuous', p_value=0.01):
    """inputs: series: a dictionary of clusterings corresponding to dates
                baseline_clustering (optional): a dictionary, baseline clustering for labeling
                option: can be either 'continuous', which assigns date t's labeling according to date t-1's
                        or 'baseline', which labels the clustering in every timestamp according to the baseline clustering
        returns a dictionary of dictionary of clusterings: {date: {label:cluster,label:cluster...}...}"""
    sorteddates = sorted(series.keys(), key=lambda d: map(int, d.split('-')))
    results = {}
    if option == 'continuous':
        temp = series[sorteddates[0]][:]
        temp.sort(key=len, reverse=True)
        results[sorteddates[0]] = {i + 1: temp[i] for i in range(len(series[sorteddates[0]]))}
        for t in range(1, len(sorteddates)):
            results[sorteddates[t]] = label_clusters(results[sorteddates[t - 1]],
                                                     series[sorteddates[t]], p_value=p_value)
        return results
    elif option == 'baseline':

        for t in range(len(sorteddates)):
            results[sorteddates[t]] = label_clusters(baseline_clustering,
                                                     series[sorteddates[t]], p_value=p_value)
        return results
    else:
        print("option can only be 'continuous' or 'baseline', your input is %s." % option)
        return None


def clustering_universe(trees, clusterings, c_measure, quantile=0.25):
    """compute the central and peripheral universes according to a given dict of {date: clustering}"""
    result = {}
    sorteddates = sorted(trees.keys(), key=lambda d: map(int, d.split('-')))
    for k in sorteddates:
        T = trees[k]
        subresult = {}
        C = clusterings[k]
        peripheral = []
        central = []
        for c in C:
            if len(list(T.subgraph(c).edges())) == 0:
                # elements in clusters with no edges will be considered peripheral
                peripheral.extend(c)
            else:
                peripheral.extend(portfolio(T.subgraph(c), c_measure, quantile, "lower"))
                central.extend(portfolio(T.subgraph(c), c_measure, quantile, "upper"))
        subresult["central"] = central
        subresult["peripheral"] = peripheral
        result[k] = subresult
    return result


def construct_clusters(trees, method='Newman'):
    """input: trees: could be either iGraph trees or Networkx trees
              method: 'Newman' or 'ClausetNewman'"""
    sorteddates = sorted(trees.keys(), key=lambda d: map(int, d.split('-')))
    if type(trees[sorteddates[0]]) != ig.Graph:
        usabletrees = {}
        for t in sorteddates:
            usabletrees[t] = NXtoIG(trees[t])
    else:
        usabletrees = trees
    ig.arpack_options.maxiter = 500000
    clusters = {}
    IGclusters = {}
    if method == 'Newman':
        if type(trees[sorteddates[0]]) != ig.Graph:
            usabletrees = {}
            for t in sorteddates:
                usabletrees[t] = NXtoIG(trees[t])
        else:
            usabletrees = trees
        for t in sorteddates:
            c = usabletrees[t].community_leading_eigenvector(weights="weight")
            clusters[t] = list(c)
            IGclusters[t] = c
            for i in range(0, len(c)):
                clusters[t][i] = [usabletrees[t].vs["name"][j] for j in c[i]]
        return clusters, IGclusters
    elif method == 'ClausetNewman':
        if type(trees[sorteddates[0]]) != ig.Graph:
            usabletrees = {}
            for t in sorteddates:
                usabletrees[t] = NXtoIG(trees[t])
        else:
            usabletrees = trees
        for t in sorteddates:
            c = usabletrees[t].community_fastgreedy(weights="weight").as_clustering()
            clusters[t] = list(c)
            IGclusters[t] = c
            for i in range(0, len(c)):
                clusters[t][i] = [usabletrees[t].vs["name"][j] for j in c[i]]
        return clusters, IGclusters
    else:
        print ("'method' can only be 'Newman' or 'ClausetNewman'. Your input was '%s'" % method)
        return None


def find_cluster_diameter(labeling, trees, IGclusters, cluster_label):
    """find the historical diameter on each date for a given cluster (identified by 'cluster_label'"""
    sorteddates = sorted(labeling.keys(), key=lambda d: map(int, d.split('-')))
    result = {}
    for t in sorteddates:
        try:
            rep_stock = labeling[t][cluster_label][0]
            for i in range(len(labeling[t])):
                if rep_stock in trees[t].subgraph(IGclusters[t][i]).vs["name"]:
                    result[t] = trees[t].subgraph(IGclusters[t][i]).diameter(directed=False, unconn=False,
                                                                             weights="weight")
                    break
        except:
            result[t] = np.nan
    return result
