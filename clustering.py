import igraph as ig
import networkx as nx
import numpy as np
from trees import importdata, MST
from scipy.stats import hypergeom
from sklearn.metrics import adjusted_rand_score
import progressbar
import matplotlib.pyplot as plt
import pandas as pd


def NXtoIG(nxgraph):
    """Convert a networkx network to an igraph network"""
    edgelist = list(nx.to_edgelist(nxgraph))
    length_dict = {(edge[0],edge[1]):edge[2]['length'] for edge in edgelist}
    weight_dict = {(edge[0],edge[1]):edge[2]['weight'] for edge in edgelist}
    weight_plus_one_dict = {(edge[0],edge[1]):edge[2]['weight+1'] for edge in edgelist}
    G = ig.Graph(len(list(nxgraph.nodes())))
    G.vs["name"] = list(nxgraph.nodes())
    G.add_edges(nxgraph.edges())
    G.es["weight"] = 1.0
    G.es['length'] = 1.0
    G.es['weight+1'] = 1.0
    for e in G.es:
        e['length'] = length_dict[(G.vs[e.source]['name'],G.vs[e.target]['name'])]
        e['weight'] = weight_dict[(G.vs[e.source]['name'],G.vs[e.target]['name'])]
        e['weight+1'] = weight_plus_one_dict[(G.vs[e.source]['name'],G.vs[e.target]['name'])]
    return G

def NXdicttoIGdict(NXtrees):
    IGtrees = {}
    print "Converting NXtrees to IGtrees..."
    sorteddates = sorted(NXtrees.keys(), key=lambda d: map(int, d.split('-')))
    bar = progressbar.ProgressBar(max_value=len(sorteddates))
    count = 0
    for t in sorteddates:
        IGtrees[t] = NXtoIG(NXtrees[t])
        count = count+1
        bar.update(count)
    return IGtrees

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


def total_clustering(enddate, startdate, filename="SP100_20170612.csv", method='Newman', n_of_clusters = None):
    """helper to create a clustering with the whole period starting at startdate, ending at enddate, as the window.
    method can be 'Newman' or 'ClausetNewman' """
    df = importdata(filename)
    end = int(np.where(df.index == enddate)[0])
    start = int(np.where(df.index == startdate)[0])
    total_tree = MST(thresh=95,filename=filename,\
                 window=end-start+1,\
                 enddate=enddate,\
                 startdate=df.index[start - 1].strftime('%Y-%m-%d'),\
                     # here we pass the date one day before startdate to MST
                 space=end-start+1)[df.index[end].strftime('%Y-%m-%d')]
    IGtree = NXtoIG(total_tree)
    if method == 'Newman':
        if n_of_clusters == None:
            C = IGtree.community_leading_eigenvector(weights="weight")
        else:
            extra = 0
            length = 0
            while length != n_of_clusters:
                C = IGtree.community_leading_eigenvector(weights="weight", clusters=n_of_clusters+extra)
                length = len(C)
                extra=extra+1
        clustersNewman = list(C)
        for i in range(0, len(C)):
            clustersNewman[i] = [IGtree.vs["name"][j] for j in C[i]]
        clustersNewman.sort(key=len, reverse=True)
        return {i + 1: clustersNewman[i] for i in range(len(clustersNewman))}
    elif method == 'ClausetNewman':
        C = IGtree.community_fastgreedy(weights="weight").as_clustering(n=n_of_clusters)
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




def construct_clusters(trees, method='Newman', n_of_clusters=None):
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
    print("Computing clusterings using method=%s, n_of_clusters=%s" %(method, n_of_clusters))
    bar = progressbar.ProgressBar(max_value=len(sorteddates))
    count = 0
    if method == 'Newman':
        for t in sorteddates:
            if n_of_clusters == None:
                c = usabletrees[t].community_leading_eigenvector(weights="weight+1")
            else:
                extra = 0
                length = 0
                while length != n_of_clusters:
                    c = usabletrees[t].community_leading_eigenvector(weights="weight+1", clusters=n_of_clusters+extra)
                    if len(c) == length:
                        break
                    length = len(c)
                    extra=extra+1
            IGclusters[t] = c
            clusters[t] = list(c)
            for i in range(0, len(c)):
                clusters[t][i] = [usabletrees[t].vs["name"][j] for j in c[i]]
            count = count+1
            bar.update(count)
        return clusters, IGclusters
    elif method == 'ClausetNewman':
        for t in sorteddates:
            c = usabletrees[t].community_fastgreedy(weights="weight+1").as_clustering(n=n_of_clusters)
            clusters[t] = list(c)
            IGclusters[t] = c
            for i in range(0, len(c)):
                clusters[t][i] = [usabletrees[t].vs["name"][j] for j in c[i]]
            count = count+1
            bar.update(count)
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
                                                                             weights="length")
                    break
        except:
            result[t] = np.nan
    return result

def average_cluster_diameter(IGfilters, IGclusters):
    """find the average diameter of clusters for each date"""
    sorteddates = sorted(IGclusters.keys(), key=lambda d: map(int, d.split('-')))
    result = {}
    print "Finding the average diameter of clusters..."
    bar = progressbar.ProgressBar(max_value=len(sorteddates))
    count = 0
    for t in sorteddates:
        sum = 0.0
        for i in range(len(IGclusters[t])):
            sum += IGfilters[t].subgraph(IGclusters[t][i]).diameter(directed=False, unconn=True,
                                                                     weights="length")
        result[t] = sum/len(IGclusters[t])
        count+=1
        bar.update(count)
    return result

def number_of_clusters(clusterings):
    """find the number of clusters in a dictionary of clustering for each day"""
    sorteddates = sorted(clusterings.keys(), key=lambda d: map(int, d.split('-')))
    result = {}
    print "Finding the number of clusters in clusterings..."
    bar = progressbar.ProgressBar(max_value=len(sorteddates))
    count = 0
    for t in sorteddates:
        result[t] = len(clusterings[t])
        count+=1
        bar.update(count)
    return result

def movingARI(IGclusterings):
    """Compute the moving adjusted Rand index"""
    print("Computing Adjusted Rand index...")
    sorteddates = sorted(IGclusterings.keys(), key=lambda d: map(int, d.split('-')))
    ARI = {}
    bar = progressbar.ProgressBar(max_value=len(sorteddates)-1)
    count = 0
    for i in range(1, len(sorteddates)):
        ARI[sorteddates[i]] = adjusted_rand_score(IGclusterings[sorteddates[i]].membership,IGclusterings[sorteddates[i-1]].membership)
        count = count+1
        bar.update(count)
    return ARI

def ARImatrix(IGclusterings):
    """Compute the moving adjusted Rand index"""
    print("Computing Adjusted Rand index matrix...")
    sorteddates = sorted(IGclusterings.keys(), key=lambda d: map(int, d.split('-')))
    ARI = np.zeros((len(sorteddates),len(sorteddates)))
    bar = progressbar.ProgressBar(max_value=len(sorteddates))
    count = 0
    for i in range(len(sorteddates)):
        for j in range(len(sorteddates)):
            ARI[i][j] = adjusted_rand_score(IGclusterings[sorteddates[i]].membership,IGclusterings[sorteddates[j]].membership)
        count = count+1
        bar.update(count)
    return ARI

def plot_ari_matrix(arimatrix, IGclusterings):
    sorteddates = sorted(IGclusterings.keys(), key=lambda d: map(int, d.split('-')))
    fig, ax1 = plt.subplots(1,1, figsize = (15,15))
    cax = ax1.imshow(arimatrix, cmap='jet')
    tks = range(0,len(sorteddates),50)
    ax1.set_xticklabels(['']+[sorteddates[i] for i in tks])
    ax1.set_yticklabels(['']+[sorteddates[i] for i in tks])
    fig.colorbar(cax, fraction=0.0455, pad=0.04)
    return fig
