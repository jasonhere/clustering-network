import pandas as pd
import numpy as np
import datetime as dt
import math
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
#import pygraphviz
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import random

def corr_matrix(df, window=250, enddate="2017-01-24", method="gower"):
    """To generate correlation matrix for a certain period, method = 'gower' or 'power'"""
    end = int(np.where(df.index==enddate)[0])
    start = end - window + 1
    sub = df[start:end+1]
    #print(sub)
    corr_mat = sub.corr(min_periods=250)
    corr_mat = corr_mat.dropna(axis=0, how='all')
    corr_mat = corr_mat.dropna(axis=1, how='all')    
    if method == "gower":
        corr_mat = (2-2*corr_mat[corr_mat.notnull()])**0.5# gower
    elif method == "power":
        corr_mat = 1-corr_mat[corr_mat.notnull()]**2# power
    #corr_mat.apply(lambda x:1-x**2 if not math.isnan(x) else np.nan)
    return corr_mat
## still need to deal with the situation when enddate is not in the index.


def rolling_corr(df, window=250, enddate="2017-01-24", startdate='2005-01-03',space=10):
    """Return a dictionary of correlation matrices.
    The key is the enddate of the window, the value is corresponding correlation matrix"""
    end = int(np.where(df.index==enddate)[0])
    start = int(np.where(df.index==startdate)[0])
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
    mapping = dict(zip(list(range(127)),list(corr_matrix.index)))
    G = nx.relabel_nodes(G, mapping, copy=False)
    #delete NAN weights
    for (u,v,d) in G.edges(data=True):
        if math.isnan(d["weight"]):
            G.remove_edges_from([(u,v)])
    #delete self-connected edges
    for (u,v,d) in G.edges(data=True):
        if u==v:
            G.remove_edges_from([(u,v)])
    #delete nodes whose degree is 0
    nodes = list(G.nodes())
    for node in nodes:
        if G.degree(node) == 0:
            G.remove_node(node)
    return G


def draw_network(G):
    # Define a layout for the graph
    #pos=nx.spring_layout(G) # positions for all nodes
    pos=graphviz_layout(G) # positions for all nodes
    fig=plt.figure(1,figsize=(40,40)) #Let's draw a big graph so that it is clearer

    # draw the nodes: red, sized, transperancy
    nx.draw_networkx_nodes(G, pos, 
                           node_color='r',
                           node_size=100,
                           alpha=.8)

    # draw the edges
    nx.draw_networkx_edges(G,pos,
                           edgelist=list(G.edges()),
                           width=0.5,alpha=0.5,edge_color='b')


    node_name={}
    for node in G.nodes():
        node_name[node]=str(node)


    nx.draw_networkx_labels(G,pos,node_name,font_size=16)

    plt.axis('off')
    plt.show()


def importdata(filename):
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


def exponential_weighted_corr(rdata, t, theta = 250, window = 250):
    """
    Compared to the formula in paper, tou is the length of time window, so use window to stand
        for tou
    t is the date of correlation(number in the list)
    
    """
    weight_list = [ np.exp((i-window)/theta)/window for i in range(1,window+1)]
    weight_list = np.array(weight_list/sum(weight_list))    
    t_list = rdata.index.format()
    
    end = t_list.index(t)
    start = end - window + 1
    startdate = rdata.index[start]
    enddate = rdata.index[end]
    
    tmp = rdata.loc[startdate:enddate]
    nrow, ncol = tmp.shape

    cov = np.cov(tmp.values, rowvar = False, aweights = weight_list)
    cov = pd.DataFrame(cov, index = tmp.columns, columns = tmp.columns)
    cov = cov.dropna(axis=0, how='all')
    cov = cov.dropna(axis=1, how='all')
    cov_values = cov.values
    cov_diag = np.sqrt(np.diag(cov_values))
    corr = (cov.values/cov_diag).T/cov_diag
    corr = pd.DataFrame(corr, index = cov.columns, columns = cov.columns)   
    return corr

def rolling_expweighted_corr(rdata, t, theta = 250, window = 250):
    dic0 = {}
    #for i in range(window-1, len(rdata.index)):
    t_list = rdata.index.format()
    for i in range(t-window, t+1):
        date = t_list[i]
        #dic0[date] = pd.DataFrame(exponential_weighted_corr(rdata, date, theta, window), index =\
        #    rdata.columns, columns = rdata.columns)
        dic0[date] = exponential_weighted_corr(rdata, date, theta, window)
    return dic0


def shrinkage(t, log_ret, theta=250, window = 250, method = "gower"):
    dates=sorted(log_ret.index)
    dates=[str(x)[:10] for x in dates]
    # dic0 is a dictionary. Keys are dates; values are the exponential-weighted pearson correlation matrixs for a 250-day window ending at that date.
    t_index = dates.index(t)
    dic0 = rolling_expweighted_corr(log_ret, t_index, theta, window)
    # create an empty dataframe to calculate the first part in the formula
    new=pd.DataFrame(0,columns=log_ret.columns,index=log_ret.columns)
    # create an zero number to calculate the second part in the formula
    num=0

    # window = 125, for the shrinkage correlation at time t, we need the exponential-weighted pearson correlation coefficient over the last tou days
    for i in range(dates.index(t)-window, dates.index(t)+1):
        d=dates[i]
        # the first part of the formula: sum of the correlation coefficent of the two specific stocks during the 125 days 
        a = dic0[d]
        #a = a.dropna(axis=0, how='all')
        #a = a.dropna(axis=1, how='all')
        #new+=a
        new = new.add(a, fill_value = 0)
        # the second part of the formula: sum of the correlation coefficent of all pairs of stocks during the 125 days 
        num+=(sum(a.sum())-log_ret.shape[1])/2 
    # normalization corresponding to the formular (2/(N*(N-1)))
    num=num*2/((log_ret.shape[1])*(log_ret.shape[1]-1))
    new = new.dropna(axis=0, how='all')
    new = new.dropna(axis=1, how='all')
    new=new+num
    # normalization corresponding to the formular (2/(window+1))
    new=new/(2*(window+1))
    np.fill_diagonal(new.values,1)
    
    if method == "gower":
        corr_mat = (2-2*new[new.notnull()])**0.5# gower
    elif method == "power":
        corr_mat = 1-new[new.notnull()]**2# power
      
    return corr_mat


def MST(log_ret, window=250, enddate="2017-01-24", startdate='2015-12-30',space=1, corr=None):
    """Return a dictionary of Minimum Spanning Tree for each end date"""
    #log_ret = importdata(filename)[1]
    if corr == "ew":
        dates=sorted(log_ret.index)
        dates=[str(x)[:10] for x in dates]
        dic0 = log_ret.ewm(span=window/2, min_periods=250).corr()
        dic1={}
        end = dates.index(enddate)
        start = dates.index(startdate)
        dates_space = dates[end:start:-space]
        for t in dates_space:
            dic1[t]=shrinkage(t,log_ret=log_ret,dic0=dic0,window=window)     
        dic = {}
        for i in dic1:
            s = str(i)[:10]
            dic[s] = (2-2*dic1[i][dic1[i].notnull()])**0.5
    else:
        dic = rolling_corr(log_ret, window, enddate, startdate, space)
    trees = {}
    for key in sorted(dic.keys()):
        corr_matrix = dic[key]
        G = constructgraph(corr_matrix)
        T = nx.minimum_spanning_tree(G, "weight")    
        trees[key] = T
    return trees


def weighted_degree_centrality(T):
    """Return a dictionary of weighted degree centrality for each node,
    the weighted degree is defined as (2-gower weight)"""
    for u,v,d in T.edges(data=True):
        d['weight2']=2-d['weight']
    degree = nx.degree(T,weight='weight2')
    degree = dict(degree)
    degree_c={}
    total=T.size(weight='weight2')
    for i in degree:
        degree_c[i]=degree[i]/total
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
    items = centrality.items()
    items.sort(key=lambda item: (item[1], item[0]))
    v=[item[1] for item in items]
    sorted_keys = [item[0] for item in items]
    number = int(len(sorted_keys)*quantile)
    if  option == 'lower':
        pos=len(v)-v[::-1].index(v[number])
        stock_list = sorted_keys[:pos]
    elif option == 'upper':
        # print(v, number)
        if v[-number]==0:
            pos=len(v)-v[::-1].index(0)
        else: 
            pos=v.index(v[-number])
        stock_list = sorted_keys[pos:]
    return stock_list
    #return {k:centrality[k] for k in stock_list}


def port_change(list1,list2):
    """Calculate the percentage change between two stock lists"""
    count = 0
    if not list1:
        return None
    else:
        for x in list2:
            if x not in list1:
                count+=1.0
    return count/len(list2)


def getportfolio(dic,measure, option):
    """Return a dict of portfolio lists"""
    performance=pd.DataFrame(index=sorted(dic.keys()))
    portfolio_dict={}
    for key in sorted(dic.keys()):
        corr_matrix = dic[key]
        G=constructgraph(corr_matrix)
        T=nx.minimum_spanning_tree(G, "weight")
        #T = nx.algorithms.tree.mst.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')
        portfolio_dict[key]=portfolio(T,c_measure=measure,option=option)
    return portfolio_dict


def get_portfolios(dic, c_measure, quantile=0.25):
    """dic: dictionary of MST, which can be generated by function MST()
    returns a dictionary of central and peripheral stock list for each window, the key is enddate of the window"""
    result = {}
    for k,T in dic.items():
        subresult = {}
        central = portfolio(T, c_measure, quantile, "upper")
        peripheral = portfolio(T, c_measure, quantile, "lower")
        subresult["central"] = central
        subresult["peripheral"] = peripheral
        result[k] = subresult
    return result

## apply get_portfolios function to c_measure in ['degree','closeness','betweenness'] 
## to get three dictionaries for these three measure of centrality.


def portfolio_change(dic):
    """dic: dictonary of central and peripheral stock list for each window, generated from get_portfolio() function
    returns a dataframe of percentage change for stock list"""
    dates=sorted(dic.keys())
    change_dict_c = {}
    change_dict_p = {}
    for i in range(0,len(dates)-1):
        list1c = dic[dates[i]]["central"]
        list2c = dic[dates[i+1]]["central"]
        list1p = dic[dates[i]]["peripheral"]
        list2p = dic[dates[i+1]]["peripheral"]
        change_dict_c[dates[i+1]] = port_change(list1c,list2c)
        change_dict_p[dates[i+1]] = port_change(list1p,list2p)
    c = pd.DataFrame(data = change_dict_c, index=["Central"]).T
    p = pd.DataFrame(data = change_dict_p, index=["Peripheral"]).T
    change = pd.concat([c,p], axis=1, join='inner')
    return change


def cov_matrix(df, stocklist, window=250, enddate="2017-02-28"):
    """To generate correlation matrix for a certain period, method = 'gower' or 'power'"""
    end = int(np.where(df.index==enddate)[0])
    start = end - window + 1
    sub = df[start:end+1][stocklist]
    #print(sub)
    cov_mat = np.cov(sub.T)
    return cov_mat
## still need to deal with the situation when enddate is not in the index.


def min_variance_weights(cov):
    try:
        mask = np.all(np.isnan(cov), axis=1)
        cov = cov[~mask]
        cov = cov.T
        mask = np.all(np.isnan(cov), axis=1)
        cov = cov[~mask]
        cov = cov.T
        S = opt.matrix(cov)
        n = cov.shape[0]
        q = opt.matrix(0.0, (n, 1))
        G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
        h = opt.matrix(0.0, (n ,1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
        weights = solvers.qp(S, q, G, h, A, b)["x"]
        risk = np.sqrt(blas.dot(weights, S*weights))
    except:
        weights=np.asarray([[1]])
        risk = np.sqrt(cov)
    return np.asarray(weights), risk


def min_variance_perform(pricedf, stocklist, weights, startdate, space=1):
    start = int(np.where(pricedf.index==startdate)[0])
    end = start + space
    df = pricedf.iloc[[start,end]][stocklist]
    date = sorted(df.index)[1]
    p_current = df.iloc[0]
    p_next = df.iloc[1]
    a = (p_next/p_current).values
    #print(a, weights)
    r = np.average(a, weights = weights)
    return (date, r)


def equal_perform(pricedf, stocklist, startdate, space=1):
    start = int(np.where(pricedf.index==startdate)[0])
    end = start + space
    df = pricedf.iloc[[start,end]][stocklist]
    date = sorted(df.index)[1]
    p_current = df.iloc[0]
    p_next = df.iloc[1]
    a = (p_next/p_current).values
    r = np.average(a)
    return (date, r)


def measure_performance(pricedf, stocklist, startdate, space=1, weights = None):
    start = int(np.where(pricedf.index==startdate)[0])
    end = start + space
    try:
        df = pricedf.iloc[[start,end]][stocklist]
        date = sorted(df.index)[1]
        p_current = df.iloc[0]
        p_next = df.iloc[1]
        a = (p_next/p_current).values
        r = np.average(a, weights = weights)
        return (date, r)
    except:
        return (None, None)


def performance(dic, pricedf, window=250, space=1, weights = False):
    """dic: dictonary of central and peripheral stock list for each window, generated from get_portfolio() function"""
    colnames = ["Upper","Lower"]
    if weights == True:
        colnames.extend([str(x)+'_Weighted' for x in colnames])
        ret = pricedf/pricedf.shift(1)
        ret = ret.iloc[1:]
    colnames.sort()
    dates = sorted(dic.keys())
    performance = pd.DataFrame(columns=colnames)
    for d in dates:
        clist = dic[d]["central"]
        plist = dic[d]["peripheral"]
        date, cr = measure_performance(pricedf, clist, d, space, weights = None)
        pr = measure_performance(pricedf, plist, d, space, weights = None)[1]
        if date:
            performance.set_value(date, "Upper", cr)
            performance.set_value(date, "Lower", pr)    
        if weights == True:
            date, cr = measure_performance(pricedf, clist, d, space, weights = None)
            pr = measure_performance(pricedf, plist, d, space, weights = None)[1]
            performance.set_value(date, "Upper", cr)
            performance.set_value(date, "Lower", pr)                      
            c_cov = cov_matrix(ret, clist, window, date)
            p_cov = cov_matrix(ret, plist, window, date)
            cw = min_variance_weights(c_cov)[0]
            cw = list(np.concatenate(cw).ravel())
            pw = min_variance_weights(p_cov)[0]
            pw = list(np.concatenate(pw).ravel())
            crw = measure_performance(pricedf, clist, d, space, cw)[1]
            prw = measure_performance(pricedf, plist, d, space, pw)[1]
            if date:
                performance.set_value(date, "Upper_Weighted", crw)
                performance.set_value(date, "Lower_Weighted", prw)             
    performance = performance.sort_index()
    return performance


def daily_performance(dic, pricedf, window=250, space=1, weights = False):
    """to calculate daily performance used for sharpe ratio"""
    """dic: dictonary of central and peripheral stock list for each window, generated from get_portfolio() function"""
    colnames = ["Upper","Lower"]
    if weights == True:
        colnames.extend([str(x)+'_Weighted' for x in colnames])
        ret = pricedf/pricedf.shift(1)
        ret = ret.iloc[1:]
    colnames.sort()
    dates = sorted(dic.keys())
    performance = pd.DataFrame(columns=colnames)
    for d in dates:
        clist = dic[d]["central"]
        plist = dic[d]["peripheral"]
        
        date, cr = measure_performance(pricedf, clist, d, 1, weights = None)
        pr = measure_performance(pricedf, plist, d, 1, weights = None)[1]
        if date:
            performance.set_value(date, "Upper", cr)
            performance.set_value(date, "Lower", pr)    
        if weights == True:
            date, cr = measure_performance(pricedf, clist, d, space, weights = None)
            pr = measure_performance(pricedf, plist, d, space, weights = None)[1]
            performance.set_value(date, "Upper", cr)
            performance.set_value(date, "Lower", pr)                      
            c_cov = cov_matrix(ret, clist, window, date)
            p_cov = cov_matrix(ret, plist, window, date)
            cw = min_variance_weights(c_cov)[0]
            cw = list(np.concatenate(cw).ravel())
            pw = min_variance_weights(p_cov)[0]
            pw = list(np.concatenate(pw).ravel())
            crw = measure_performance(pricedf, clist, d, space, cw)[1]
            prw = measure_performance(pricedf, plist, d, space, pw)[1]
            if date:
                performance.set_value(date, "Upper_Weighted", crw)
                performance.set_value(date, "Lower_Weighted", prw)             
    performance = performance.sort_index()
    return performance



def perform_all(log_ret, pricedf, window=250, enddate="2017-01-24", startdate='2015-12-30',space=1, corr=None):
    """Run performance() function for all methods"""
    # colnames = ["Degree_Lower", "Degree_Lower_Weighted", "Degree_Upper", "Degree_Upper_Weighted", "Closeness_Lower", "Closeness_Lower_Weighted", "Closeness_Upper", "Closeness_Upper_Weighted", "Betweenness_Lower", "Betweenness_Lower_Weighted", "Betweenness_Upper", "Betweenness_Upper_Weighted"]
    # colnames.extend([str(x)+'_w' for x in colnames])
    # colnames.sort()
    # dates = sorted(list(dic.keys()))
    # performancedf=pd.DataFrame()
    trees = MST(log_ret=log_ret, window=window, enddate=enddate, startdate=startdate,space=space, corr=corr)
    degree = get_portfolios(trees, "degree")
    degreep = performance(dic=degree, pricedf=pricedf, space=space, weights = True)
    #index = degreep.index
    degreep.columns = ["Degree_Lower", "Degree_Lower_Weighted", "Degree_Upper", "Degree_Upper_Weighted"]
    closeness = get_portfolios(trees, "closeness")
    closenessp = performance(dic=closeness, pricedf=pricedf, space=space, weights = True)
    #closenessp = closenessp.set_index(index)
    closenessp.columns = ["Closeness_Lower", "Closeness_Lower_Weighted", "Closeness_Upper", "Closeness_Upper_Weighted"]
    betweenness = get_portfolios(trees, "betweenness")
    betweennessp = performance(dic=betweenness, pricedf=pricedf, space=space, weights = True)
    #betweennessp = betweennessp.set_index(index)
    betweennessp.columns = ["Betweenness_Lower", "Betweenness_Lower_Weighted", "Betweenness_Upper", "Betweenness_Upper_Weighted"]
    performancedf = pd.concat([degreep, closenessp, betweennessp],axis=1, join='inner')   
    return performancedf



