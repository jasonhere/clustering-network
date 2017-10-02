import networkx as nx
import numpy as np
import pandas as pd
import math
from sklearn.covariance import ledoit_wolf
import progressbar

def corr_matrix(ret, window=250, enddate="2017-01-24", shrinkage=None, exp_shrinkage_theta = 125):
    """To generate correlation matrix for a certain period, method = 'gower' or 'power'"""
    end = int(np.where(ret.index == enddate)[0])
    start = end - window
    # sub = ret[start:end].dropna(thresh=thresh*window, axis=1)  # drop whole column when there are less than or equal to
    # # thresh number of non-nan entries in the window
    sub = ret[start:end]
    sub = sub.ffill()
    sub = sub.bfill()
    subret = sub
    if shrinkage == None:
        corr_mat = subret.corr(min_periods=1)
    # elif shrinkage == "LedoitWolf":
    #     cov = ledoit_wolf(subret, assume_centered=True)[0]
    #     std = np.sqrt(np.diagonal(cov))
    #     corr_mat = (cov / std[:, None]).T / std[:, None]
    #     np.fill_diagonal(corr_mat, 1.0)
    #     corr_mat = pd.DataFrame(data=corr_mat, index=subret.columns, columns=subret.columns)
    elif shrinkage == "Exponential":
        stocknames = subret.columns
        weight_list = [np.exp((i - window) / exp_shrinkage_theta) for i in range(1, window + 1)]
        weight_list = np.array(weight_list / sum(weight_list))
        cov = np.cov(subret.values, rowvar=False, aweights=weight_list)
        cov_diag = np.sqrt(np.diag(cov))
        corr = (cov / cov_diag).T / cov_diag
        corr_mat = pd.DataFrame(corr)
        corr_mat.columns = stocknames
        corr_mat.index = stocknames
    else:
        print "'shrinkage' can only be None or 'Exponential'"
        return None
    # corr_mat.apply(lambda x:1-x**2 if not math.isnan(x) else np.nan)
    return corr_mat


def all_corr(ret, window=250, shrinkage=None,exp_shrinkage_theta=125):
    print("Computing all correlations with window=%s, shrinkage=%s, theta=%s..." % (window, shrinkage, exp_shrinkage_theta))
    allcorr = {}
    alldates = ret.index
    alldates.sort_values()
    bar = progressbar.ProgressBar(max_value=len(alldates[window:]))
    count = 0
    for d in alldates[window:]:
        d = d.strftime("%Y-%m-%d")
        allcorr[str(d)] = corr_matrix(ret, window, enddate=d, shrinkage=shrinkage, exp_shrinkage_theta = exp_shrinkage_theta)
        count = count+1
        bar.update(count)
    alldates = np.array(sorted([s[-10:] for s in allcorr.keys()]))
    adjusted_R = {}
    print("Computing adjusted_R")
    bar = progressbar.ProgressBar(max_value=len(alldates))
    count = 0
    for d in alldates:
        corr = allcorr[str(d)].values
        stocknames = allcorr[str(d)].index.values
        target = (sum(sum(corr))-sum(np.diag(corr)))/(corr.shape[0]*(corr.shape[0]-1))
        temp= pd.DataFrame(corr+target)
        temp.columns = stocknames
        temp.index = stocknames
        adjusted_R[str(d)] = temp
        count = count+1
        bar.update(count)
    return allcorr, adjusted_R

def rolling_corr(allcorr, dates, adjusted_R=None, average=False, tau=125):
    """Return a dictionary of correlation matrices.
    The key is the enddate of the window, the value is corresponding correlation matrix"""
    result = {}
    alldates = np.array(sorted([s[-10:] for s in allcorr.keys()]))
    if average==False:
        print("Computing corrs without average...")
        bar = progressbar.ProgressBar(max_value=len(dates))
        count = 0
        for d in dates:
            d = pd.to_datetime(d).strftime("%Y-%m-%d")
            result[str(d)] = allcorr[str(d)]
            count = count+1
            bar.update(count)
    else:
        print("Computing corr with average...")
        bar = progressbar.ProgressBar(max_value=len(dates))
        count = 0
        for d in dates:
            d = d.strftime("%Y-%m-%d")
            windowend = int(np.where(alldates==d)[0])+1
            windowstart = windowend-tau
            shrinkage_corr = sum(adjusted_R[str(dd)] for dd in alldates[windowstart:windowend])/(2*(tau+1))
            result[str(d)] = shrinkage_corr
            count = count+1
            bar.update(count)
    return result


def constructgraph(corr_matrix, method='gower'):
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


def importdata(filename):
    """imports data from a .csv file, returns a pandas dataframe of daily returns"""
    df = pd.read_csv(filename, nrows=10)
    nodenames = df.columns[1:]
    dtypedict = {stock: np.float64 for stock in nodenames}
    dtypedict['Date'] = str
    df = pd.read_csv(filename, dtype=dtypedict)
    # set date as index and sort by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df = df.drop(['Date'], axis=1)
    df.sort_index(inplace=True)
    # forward fill for NA
    #df.fillna(method='ffill', axis=0, inplace=True)
    # Log return
    # log_ret = np.log(df) - np.log(df.shift(1))
    return df


def MST(corrs, method="gower"):
    """Returns a dictionary of Minimum Spanning Tree for each end date"""
    dic = corrs
    trees = {}
    print("Creating MSTs...")
    bar = progressbar.ProgressBar(max_value=len(sorted(dic.keys())))
    count = 0
    for key in sorted(dic.keys()):
        corr_matrix = dic[key]
        G = constructgraph(corr_matrix, method)
        T = nx.minimum_spanning_tree(G, "length")
        trees[key[-10:]] = T
        count = count+1
        bar.update(count)
    return trees

def compute_dates(alldates, startdate, enddate, space):
    end = int(np.where(alldates==enddate)[0])
    start = int(np.where(alldates==startdate)[0])
    return alldates[end:start:-space]

# unfinished: need DBHT
# def construct_trees(thresh, filename="SP100_prices.csv", window=250, enddate="2017-01-24", startdate='2015-12-30',
#                     space=1,
#                     tree_type='MST', shrinkage="None"):
#     """construct a dictionary of trees {date: tree}. Based on tree_type, can return MST or DBHT"""
#     if tree_type == 'MST':
#         return MST(thresh=thresh, filename=filename, window=window, enddate=enddate, startdate=startdate, space=space,
#                    shrinkage=shrinkage)
#     elif tree_type == 'DBHT':
#         pass
#         # ......
#     else:
#         print ("'tree_type' can only be 'MST' or 'DBHT'. Your input was '%s'." % tree_type)
#         return None
