import networkx as nx
import numpy as np
import pandas as pd
import math
from sklearn.covariance import ledoit_wolf


def corr_matrix(df, thresh, window=250, enddate="2017-01-24", method="gower", shrinkage="None"):
    """To generate correlation matrix for a certain period, method = 'gower' or 'power'"""
    end = int(np.where(df.index == enddate)[0])
    start = end - window
    sub = df[start:end + 1].dropna(thresh=thresh, axis=1)  # drop whole column when there are less than or equal to
    # thresh number of non-nan entries in the window
    # print(sub)
    sub = sub.ffill()
    sub = sub.bfill()
    subret = np.log(sub) - np.log(sub.shift(1))
    subret = subret[1:]
    if shrinkage == "None":
        corr_mat = subret.corr(min_periods=1)
    elif shrinkage == "LedoitWolf":
        cov = ledoit_wolf(subret, assume_centered=True)[0]
        std = np.sqrt(np.diagonal(cov))
        corr_mat = (cov / std[:, None]).T / std[:, None]
        np.fill_diagonal(corr_mat, 1.0)
        corr_mat = pd.DataFrame(data=corr_mat, index=subret.columns, columns=subret.columns)
    else:
        print "'shrinkage' can only be 'None' or 'LedoitWolf'"
        return None
    if method == "gower":
        corr_mat = (2 - 2 * corr_mat[corr_mat.notnull()]) ** 0.5  # gower
    elif method == "power":
        corr_mat = 1 - corr_mat[corr_mat.notnull()] ** 2  # power
    # corr_mat.apply(lambda x:1-x**2 if not math.isnan(x) else np.nan)
    return corr_mat


def rolling_corr(df, thresh, window=250, enddate="2017-01-24", startdate='2005-01-03', space=10, shrinkage="None"):
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
        result[str(d)] = corr_matrix(df, thresh, window, enddate=d, shrinkage=shrinkage)
    return result


def constructgraph(corr_matrix):
    """Convert a correlation matrix to a graph"""
    G = nx.from_numpy_matrix(corr_matrix.values)
    mapping = dict(zip(list(range(len(corr_matrix.index))), list(corr_matrix.index)))
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


def importdata(filename):
    """imports data from a .csv file, returns a pandas dataframe of prices and another of log returns"""
    df = pd.read_csv(filename)
    # set date as index and sort by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df = df.drop(['Date'], axis=1)
    df.sort_index(inplace=True)
    # forward fill for NA
    # df.fillna(method='ffill', axis=0, inplace=True)
    # Log return
    # log_ret = np.log(df) - np.log(df.shift(1))
    return df


def MST(thresh, filename="SP100_prices.csv", window=250, enddate="2017-01-24", startdate='2015-12-30', space=1,
        shrinkage="None"):
    """Returns a dictionary of Minimum Spanning Tree for each end date,
    space means the interval between two sample updates"""
    price = importdata(filename)
    dic = rolling_corr(price, thresh, window, enddate, startdate, space, shrinkage=shrinkage)
    trees = {}
    for key in sorted(dic.keys()):
        corr_matrix = dic[key]
        G = constructgraph(corr_matrix)
        T = nx.minimum_spanning_tree(G, "weight")
        trees[key] = T
    return trees


# unfinished: need DBHT
def construct_trees(thresh, filename="SP100_prices.csv", window=250, enddate="2017-01-24", startdate='2015-12-30',
                    space=1,
                    tree_type='MST', shrinkage="None"):
    """construct a dictionary of trees {date: tree}. Based on tree_type, can return MST or DBHT"""
    if tree_type == 'MST':
        return MST(thresh=thresh, filename=filename, window=window, enddate=enddate, startdate=startdate, space=space,
                   shrinkage=shrinkage)
    elif tree_type == 'DBHT':
        pass
        # ......
    else:
        print ("'tree_type' can only be 'MST' or 'DBHT'. Your input was '%s'." % tree_type)
        return None
