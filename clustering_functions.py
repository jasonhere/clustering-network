# get_ipython().magic(u'matplotlib inline')
import cvxopt as opt
import igraph as ig
import networkx as nx
import numpy as np
from cvxopt import blas, solvers
from scipy.stats import hypergeom
from sklearn.metrics import adjusted_rand_score
from pandas_datareader import data
import time

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

def HGT_clustering(total_clusters, clusters, nodenames):
    """Clustering using hypergeometric test, keeping the labeling of the clusters throughout"""
    sorteddates = sorted(clusters.keys(), key=lambda d: map(int, d.split('-')))
    # The total number of elements within all the clusters is N for the whole time frame
    N = 0
    for cluster in total_clusters:
        N = N + len(cluster)

    new_current_date_clusters = {}
    for i in range(0, len(sorteddates)):
        current_date_clusters = clusters[sorteddates[i]]
        current_cluster_list = {}
        for j in range(0, len(total_clusters)):
            current_cluster = tuple()
            for k in range(0, len(current_date_clusters)):
                overlap = len(set((current_date_clusters[k])).intersection(total_clusters[j]))
                p = hypergeom.pmf(overlap, N, len(current_date_clusters[k]), len(total_clusters[j]))
                if (p < 0.01) & (len(current_cluster) < len(current_date_clusters[k])):
                    current_cluster = current_date_clusters[k]
            current_cluster_list[j] = current_cluster
        new_current_date_clusters[sorteddates[i]] = current_cluster_list
    return new_current_date_clusters

def metacorrelation(time_window_a, time_window_b):
    df = importdata("SP100_prices.csv")[1]
    rc = rolling_corr(df)
    corr_matrix_a = rc[time_window_a]
    corr_matrix_b = rc[time_window_b]
    sum_dot = np.sum(np.dot(corr_matrix_a, corr_matrix_b), axis=(0, 1))
    sum_a = np.sum(corr_matrix_b, axis=(0, 1))
    sum_b = np.sum(corr_matrix_b, axis=(0, 1))
    sum_a2 = sum_a ** 2
    sum_b2 = sum_b ** 2
    sum2_a = np.sum(np.dot(corr_matrix_a, corr_matrix_a), axis=(0, 1))
    sum2_b = np.sum(np.dot(corr_matrix_b, corr_matrix_b), axis=(0, 1))
    metacorrelation = sum_dot / (((sum2_a - sum_a2) * (sum2_b - sum_b2)) ** (1 / 2))
    return metacorrelation

def movingARI(clusters, nodenames):
    """Compute the moving adjusted Rand index"""
    sorteddates = sorted(clusters.keys(), key=lambda d: map(int, d.split('-')))
    ARI = np.empty(len(sorteddates) - 1)
    for i in range(1, len(sorteddates)):
        ARI[i - 1] = adjusted_rand_score(createlabel(clusters[sorteddates[i]], nodenames)[1],
                                         createlabel(clusters[sorteddates[i - 1]], nodenames)[1])
    return ARI

def portfolio(MST, c_measure, quantile=0.25, option='upper'):
    """Return a list of the upper or lower 25%(default) of stocks sorted by centrality.
    If the quantile is 0, would return the single one with the highest (lowest) centrality"""
    if c_measure == 'closeness':
        centrality = nx.closeness_centrality(MST, distance="weight")
    elif c_measure == 'degree':
        centrality = weighted_degree_centrality(MST)
    elif c_measure == 'betweenness':
        centrality = nx.betweenness_centrality(MST)
    else:
        print('wrong centrality measure.')
        return None
    items = centrality.items()
    items.sort(key=lambda item: (item[1], item[0]))
    v = [item[1] for item in items]
    sorted_keys = [item[0] for item in items]
    number = int(len(sorted_keys) * quantile)
    if number == 0:
        number = 1
    if option == 'lower':
        pos = len(v) - v[::-1].index(v[number])
        stock_list = sorted_keys[:pos]
    elif option == 'upper':
        if v[-number] == 0:
            pos = len(v) - v[::-1].index(0)
        else:
            pos = v.index(v[-number])
        stock_list = sorted_keys[pos:]
    else:
        print "option must be upper or lower"
        return None
    return stock_list

def clustering_universe(trees, clusterings, c_measure, quantile=0.25):
    """compute the central and peripheral universe according to a given list of clusterings"""
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

def cov_matrix(df, stocklist, window=250, enddate="2017-02-28"):
    """To generate correlation matrix for a certain period, method = 'gower' or 'power',
     differs from the one in 'all_functions.py' by the last argument"""
    end = int(np.where(df.index == enddate)[0])
    start = end - window + 1
    sub = df[start:end + 1][stocklist]
    # print(sub)
    cov_mat = np.cov(sub.T)
    return cov_mat

## still need to deal with the situation when enddate is not in the index.
def min_variance_weights(cov):
    S = opt.matrix(cov)
    n = cov.shape[0]
    q = opt.matrix(0.0, (n, 1))
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    weights = solvers.qp(S, q, G, h, A, b)["x"]
    risk = np.sqrt(blas.dot(weights, S * weights))
    return np.asarray(weights), risk

def clustering_performance(universes, weighted='TRUE'):
    price, log_ret = importdata("SP100_prices.csv")
    ret = price / price.shift(1)
    ret = ret.iloc[1:]
    univdates = sorted(universes.keys(), key=lambda d: map(int, d.split('-')))
    pricedates = sorted(pd.read_csv("SP100_prices.csv")["Date"], key=lambda d: map(int, d.split('-')))
    space = pricedates.index(univdates[1]) - pricedates.index(univdates[0])
    result = {'central': {}}
    result['central'][univdates[0]] = 1
    result['peripheral'] = {}
    result['peripheral'][univdates[0]] = 1
    for t in univdates:
        for j in ['central', 'peripheral']:
            cov = cov_matrix(ret, universes[t][j], 250, t)
            if weighted == 'TRUE':
                weights = np.transpose(min_variance_weights(cov)[0])[0]
            else:
                weights = np.divide(np.ones(len(cov)), len(cov))
            for tt in pricedates[pricedates.index(t) + 1:
                            pricedates.index(t) + 1 + space]:
                result[j][tt] = result[j][t] * np.dot(weights, np.divide(price[tt:tt][universes[t][j]].as_matrix()[0],
                                                                         price[t:t][universes[t][j]].as_matrix()[0]))
    return result

def download_daily_data(symbollist, filename):
    store = pd.HDFStore(filename)
    for i in symbollist:
        try:
            temp = data.DataReader(i, 'yahoo', '1900-01-01', time.strftime("%Y-%m-%d"))
            store[i] = temp
        except:
            pass
    store.close()
