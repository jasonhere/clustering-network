
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import datetime as dt
import math
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers


# In[2]:

def corr_matrix(df, window=250, enddate="2017-02-28", method="gower"):
    """To generate correlation matrix for a certain period, method = 'gower' or 'power'"""
    end = int(np.where(df.index==enddate)[0])
    start = end - window + 1
    sub = df[start:end+1]
    #print(sub)
    corr_mat = sub.corr(min_periods=150)
    if method == "gower":
        corr_mat = (2-2*corr_mat[corr_mat.notnull()])**0.5# gower
    elif method == "power":
        corr_mat = 1-corr_mat[corr_mat.notnull()]**2# power
    #corr_mat.apply(lambda x:1-x**2 if not math.isnan(x) else np.nan)
    return corr_mat
## still need to deal with the situation when enddate is not in the index.


# In[3]:

def rolling_corr(df, window=250, enddate="2017-02-28", startdate='2010-01-05',space=10):
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


# In[4]:

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


# In[5]:

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


# In[6]:

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


# In[7]:

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
        if v[-number]==0:
            pos=len(v)-v[::-1].index(0)
        else: 
            pos=v.index(v[-number])
        stock_list = sorted_keys[pos:]
    return stock_list
    #return {k:centrality[k] for k in stock_list}


# In[8]:

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


# In[9]:

def getdata(dic):
    """Return a dataframe of portfolio change percentage for upper or lower quantile of three measures of centrality"""
    performance=pd.DataFrame(index=sorted(dic.keys()))
    for k in ['degree','closeness','betweenness']:
        print(k)
        for j in ['upper','lower']:
            print(j)
            change_dict={}
            portfolio_dict={}
            for key in sorted(dic.keys()):
                corr_matrix = dic[key]
                G=constructgraph(corr_matrix)
                T=nx.minimum_spanning_tree(G, "weight")
                #T = nx.algorithms.tree.mst.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')
                portfolio_dict[key]=portfolio(T,c_measure=k,option=j)
            dates=sorted(portfolio_dict.keys())
            for i in range(0,len(dates)-1):
                list1 = portfolio_dict[dates[i]]
                list2 = portfolio_dict[dates[i+1]]
                change_dict[dates[i+1]] = port_change(list1,list2)
            p=pd.DataFrame(data=change_dict,index=[k+'_'+j]).T
            performance=pd.concat([performance,p],axis=1,join='inner')
    return performance


# In[10]:

def importdata(filename):
    df = pd.read_csv(filename)
    # set date as index and sort by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df = df.drop(['Date'], axis=1)
    df.sort_index(inplace=True)
    # forward fill for NA
    df.fillna(method='ffill')
    # Log return
    log_ret = np.log(df) - np.log(df.shift(1))
    return df, log_ret


# In[11]:

def do(filename="SP100_prices.csv", window=250, enddate="2017-02-28", startdate='2015-12-30',space=1):
    log_ret = importdata(filename)[1]
    dic = rolling_corr(log_ret, window, enddate, startdate, space)
    x = getdata(dic)
    upper = x[['betweenness_upper','closeness_upper','degree_upper']]
    lower = x[['betweenness_lower','closeness_lower','degree_lower']]
    #upper.plot(figsize=(20,8))
    #lower.plot(figsize=(20,8))
    return upper, lower


# In[12]:

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


# In[13]:

def perform(filename="SP100_prices.csv", window=250, enddate="2017-01-28", startdate='2015-12-30',space=1):
    pricedf, log_ret = importdata(filename)
    dic = rolling_corr(log_ret, window, enddate, startdate,space)
    dates = sorted(dic.keys())
    colnames = ["degree_upper", "degree_lower","closeness_upper","closeness_lower","betweenness_upper","betweenness_lower"]
    performance=pd.DataFrame(index=dates, columns=colnames)
    for k in ['degree','closeness','betweenness']:
        print(k)
        for j in ['upper','lower']:
            print(j)
            col = k+"_"+j
            portfolios = getportfolio(dic, k, j)
            for i in range(len(dates)-1):
                c = dates[i]
                n = dates[i+1]
                pf = portfolios[c]
                p_current = pricedf.loc[c, pf]
                p_next = pricedf.loc[n, pf]
                returns = np.nanmean(p_next/p_current)
                performance.set_value(n, col, returns)
    return performance


# In[14]:

def avgreturn(filename="SP100_prices.csv", enddate="2017-02-28", startdate='2015-12-30', space=1):
    pricedf, log_ret = importdata(filename)
    end = int(np.where(pricedf.index==enddate)[0])
    start = int(np.where(pricedf.index==startdate)[0])
    space = -space
    dates = pricedf.index.values
    dates = dates[end:start:space]
    subdf = pricedf.loc[dates,:].sort_index()
    re = subdf/subdf.shift(1)
    avg = re.mean(axis=1, skipna=True)
    return avg, avg.cumprod()


# In[15]:

def draw_performance(filename="SP100_prices.csv", window=250, enddate="2017-02-28", startdate='2015-12-30',space=1):
    returns = perform(filename, window, enddate, startdate, space)
    cumreturn = returns.cumprod()
    print(returns.cumprod().iloc[-1,:])
    avg = avgreturn(enddate=enddate, startdate=startdate, space=space)
    cumreturn["Avg_of_all_stocks"] = pd.Series(avg[1]).values
    cumreturn.plot(figsize=(20,8), title="Biweekly Cumulative Return "+startdate+" - "+enddate)


# In[16]:

def cov_matrix(df, stocklist, window=250, enddate="2017-02-28"):
    """To generate correlation matrix for a certain period, method = 'gower' or 'power'"""
    end = int(np.where(df.index==enddate)[0])
    start = end - window + 1
    sub = df[start:end+1][stocklist]
    #print(sub)
    cov_mat = np.cov(sub.T)
    return cov_mat
## still need to deal with the situation when enddate is not in the index.


# In[17]:

def min_variance_weights(cov):
    S = opt.matrix(cov)
    n = cov.shape[0]
    q = opt.matrix(0.0, (n, 1))
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    weights = solvers.qp(S, q, G, h, A, b)["x"]
    risk = np.sqrt(blas.dot(weights, S*weights))
    return np.asarray(weights), risk

def min_variance_perform(pricedf, stocklist, weights, startdate, space=1):
    start = int(np.where(pricedf.index==startdate)[0])
    end = start + space
    df = pricedf.iloc[[start,end]][stocklist]
    date = sorted(df.index)[1]
    p_current = df.iloc[0]
    p_next = df.iloc[1]
    a = (p_next/p_current).values
    r = np.average(a, weights = weights)
    return (date, r)

# In[ ]:



