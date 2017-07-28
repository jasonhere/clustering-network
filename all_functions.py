import math

import cvxopt as opt
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import numpy as np
import pandas as pd
from cvxopt import blas, solvers
from scipy.stats import hypergeom
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.metrics import adjusted_rand_score

'''
Producing the correlation matrix and minimum variance weights
'''


def corr_matrix(df, window=250, enddate="2017-01-24", method="gower"):
    """To generate correlation matrix for a certain period, method = 'gower' or 'power'"""
    end = int(np.where(df.index == enddate)[0])
    start = end - window + 1
    sub = df[start:end + 1].dropna(axis=1, how='any')  # dropna in case it is too early for some tickers to exist
    # print(sub)
    corr_mat = sub.corr(min_periods=1)
    if method == "gower":
        corr_mat = (2 - 2 * corr_mat[corr_mat.notnull()]) ** 0.5  # gower
    elif method == "power":
        corr_mat = 1 - corr_mat[corr_mat.notnull()] ** 2  # power
    # corr_mat.apply(lambda x:1-x**2 if not math.isnan(x) else np.nan)
    return corr_mat


# still need to deal with the situation when enddate is not in the index.
def rolling_corr(df, window=250, enddate="2017-01-24", startdate='2005-01-03', space=10):
    """Return a dictionary of correlation matrices.
    The key is the enddate of the window, the value is corresponding correlation matrix"""
    end = int(np.where(df.index == enddate)[0])
    start = int(np.where(df.index == startdate)[0])
    space = -space
    dates = df.index.values
    dates = dates[end:start:space]
    result = {}
    for d in dates:
        d = pd.to_datetime(d).strftime("%Y-%m-%d")
        result[str(d)] = corr_matrix(df, window, enddate=d)
    return result


def constructgraph(corr_matrix):
    """Convert a correlation matrix to a graph"""
    G = nx.from_numpy_matrix(corr_matrix.values)
    mapping = dict(zip(list(range(127)), list(corr_matrix.index)))
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
    return G


def draw_network(G):
    # Define a layout for the graph
    # pos=nx.spring_layout(G) # positions for all nodes
    pos = graphviz_layout(G)  # positions for all nodes
    fig = plt.figure(1, figsize=(40, 40))  # Let's draw a big graph so that it is clearer

    # draw the nodes: red, sized, transperancy
    nx.draw_networkx_nodes(G, pos,
                           node_color='r',
                           node_size=100,
                           alpha=.8)

    # draw the edges
    nx.draw_networkx_edges(G, pos,
                           edgelist=list(G.edges()),
                           width=0.5, alpha=0.5, edge_color='b')

    node_name = {}
    for node in G.nodes():
        node_name[node] = str(node)

    nx.draw_networkx_labels(G, pos, node_name, font_size=16)

    plt.axis('off')
    plt.show()


# connect to quandl
def importdata(filename):
    """imports data from a .csv file, returns a pandas dataframe of prices and another of log returns"""
    df = pd.read_csv(filename)
    # set date as index and sort by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df = df.drop(['Date'], axis=1)
    df.sort_index(inplace=True)
    # forward fill for NA
    df.fillna(method='ffill', axis=0, inplace=True)
    # Log return
    log_ret = np.log(df) - np.log(df.shift(1))
    return df, log_ret


def MST(filename="SP100_prices.csv", window=250, enddate="2017-01-24", startdate='2015-12-30', space=1):
    """Returns a dictionary of Minimum Spanning Tree for each end date,
    space means the interval between two sample updates"""
    log_ret = importdata(filename)[1]
    dic = rolling_corr(log_ret, window, enddate, startdate, space)
    trees = {}
    for key in sorted(dic.keys()):
        corr_matrix = dic[key]
        G = constructgraph(corr_matrix)
        T = nx.minimum_spanning_tree(G, "weight")
        trees[key] = T
    return trees


# unfinished: need DBHT
def construct_trees(filename="SP100_prices.csv", window=250, enddate="2017-01-24", startdate='2015-12-30', space=1,
                    tree_type='MST'):
    """construct a dictionary of trees {date: tree}. Based on tree_type, can return MST or DBHT"""
    if tree_type == 'MST':
        return MST(filename=filename, window=window, enddate=enddate, startdate=startdate, space=space)
    elif tree_type == 'DBHT':
        pass
            # ......
    else:
        print ("'tree_type' can only be 'MST' or 'DBHT'. Your input was '%s'." % tree_type)
        return None


# one function to do different centrality
def weighted_degree_centrality(T):
    """Return a dictionary of weighted degree centrality for each node,
    the weighted degree is defined as (2-gower weight)"""
    for u, v, d in T.edges(data=True):
        d['weight2'] = 2 - d['weight']
    degree = nx.degree(T, weight='weight2')
    degree = dict(degree)
    degree_c = {}
    total = T.size(weight='weight2')
    for i in degree:
        degree_c[i] = degree[i] / total
    return degree_c


def portfolio(MST, c_measure, quantile=0.25, option='upper'):
    """Return a list of the upper or lower 25%(default) of stocks sorted by centrality"""
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
        print "option must be lower or upper"
        return None
    return stock_list
    # return {k:centrality[k] for k in stock_list}


def port_change(list1, list2):
    """Calculate the percentage change between two stock lists"""
    count = 0
    if not list1:
        return None
    else:
        for x in list2:
            if x not in list1:
                count += 1.0
    return count / len(list2)


def get_portfolios(dic, c_measure, quantile=0.25):
    """dic: dictionary of MST, which can be generated by function MST()
    returns a dictionary of central and peripheral stock list for each window, the key is enddate of the window.
    apply get_portfolios function to c_measure in ['degree','closeness','betweenness']
    to get three dictionaries for these three measure of centrality."""
    result = {}
    for k, T in dic.items():
        subresult = {}
        central = portfolio(T, c_measure, quantile, "upper")
        peripheral = portfolio(T, c_measure, quantile, "lower")
        subresult["central"] = central
        subresult["peripheral"] = peripheral
        result[k] = subresult
    return result


def portfolio_change(dic):
    """dic: dictonary of central and peripheral stock list for each window, generated from get_portfolio() function
    returns a dataframe of percentage change for stock list"""
    dates = sorted(dic.keys())
    change_dict_c = {}
    change_dict_p = {}
    for i in range(0, len(dates) - 1):
        list1c = dic[dates[i]]["central"]
        list2c = dic[dates[i + 1]]["central"]
        list1p = dic[dates[i]]["peripheral"]
        list2p = dic[dates[i + 1]]["peripheral"]
        change_dict_c[dates[i + 1]] = port_change(list1c, list2c)
        change_dict_p[dates[i + 1]] = port_change(list1p, list2p)
    c = pd.DataFrame(data=change_dict_c, index=["Central"]).T
    p = pd.DataFrame(data=change_dict_p, index=["Peripheral"]).T
    change = pd.concat([c, p], axis=1, join='inner')
    return change


def cov_matrix(df, window=250, enddate="2017-02-28"):
    """To generate correlation matrix for a certain period, method = 'gower' or 'power'"""
    end = int(np.where(df.index == enddate)[0])
    start = end - window + 1
    sub = df[start:end + 1]
    # print(sub)
    cov_mat = np.cov(sub)
    return cov_mat


## still need to deal with the situation when enddate is not in the index.

# needs inspection
def measure_performance(pricedf, stocklist, startdate, space=1, weights=None):
    start = int(np.where(pricedf.index == startdate)[0])
    end = start + space
    try:
        df = pricedf.iloc[[start, end]][stocklist]
        date = sorted(df.index)[1]
        p_current = df.iloc[0]
        p_next = df.iloc[1]
        a = (p_next / p_current).values
        r = np.average(a, weights=weights)
        return date, r
    except:
        return None, None


def performance(dic, pricedf, space=1, weights=False):
    """dic: dictonary of central and peripheral stock list for each window, generated from get_portfolio() function"""
    colnames = ["Upper", "Lower"]
    if weights:
        colnames.extend([str(x) + '_Weighted' for x in colnames])
        ret = pricedf / pricedf.shift(1)
        ret = ret.iloc[1:]
    colnames.sort()
    dates = sorted(dic.keys())
    performance = pd.DataFrame(columns=colnames)
    for d in dates:
        clist = dic[d]["central"]
        plist = dic[d]["peripheral"]
        date, cr = measure_performance(pricedf, clist, d, space, weights=None)
        pr = measure_performance(pricedf, plist, d, space, weights=None)[1]
        if date:
            performance.set_value(date, "Upper", cr)
            performance.set_value(date, "Lower", pr)
        if weights:
            date, cr = measure_performance(pricedf, clist, d, space, weights=None)
            pr = measure_performance(pricedf, plist, d, space, weights=None)[1]
            performance.set_value(date, "Upper", cr)
            performance.set_value(date, "Lower", pr)
            # use cov_matrix() in clustering_functions.py
            c_cov = cov_matrix(ret, clist, 250, date)
            p_cov = cov_matrix(ret, plist, 250, date)
            cw = min_variance_weights(c_cov)
            pw = min_variance_weights(p_cov)
            crw = measure_performance(pricedf, clist, d, space, weights=cw)[1]
            prw = measure_performance(pricedf, plist, d, space, weights=pw)[1]
            if date:
                performance.set_value(date, "Upper_Weighted", crw)
                performance.set_value(date, "Lower_Weighted", prw)
    performance = performance.sort_index()
    return performance


'''
Clustering functions
'''


def NXtoIG(nxgraph):
    edgelist = list(nx.to_edgelist(nxgraph))
    G = ig.Graph(len(list(nxgraph.nodes())))
    G.es["weight"] = 1.0
    G.vs["name"] = list(nxgraph.nodes())
    for i in range(0, len(edgelist)):
        G[edgelist[i][0], edgelist[i][1]] = edgelist[i][2]['weight']
    return G


def createlabel(clustering, names):
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


def HGT_clustering(total_clusters, clusters):
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


def movingARI(clusters, nodenames):
    sorteddates = sorted(clusters.keys(), key=lambda d: map(int, d.split('-')))
    ARI = np.empty(len(sorteddates) - 1)
    for i in range(1, len(sorteddates)):
        ARI[i - 1] = adjusted_rand_score(createlabel(clusters[sorteddates[i]], nodenames)[1],
                                         createlabel(clusters[sorteddates[i - 1]], nodenames)[1])
    return ARI

    # testing remote jupyternotebook via ssh
