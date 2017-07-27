import pandas as pd
import numpy as np
import matlab.engine
from clustering import *
from downloaddata import *
from portfolio_and_performance import *
from trees import *
from functions_final import *
%load_ext autoreload
%autoreload 2

eng = matlab.engine.start_matlab()
eng.cd(r'C:\Users\RockLIANG\Desktop\pmfg&DBHTs') #set up matlab wdir to where the matlab scripts are located
data, log_ret = importdata("SP500_full.csv")


thresh = 95
window = 1000
t_list = data.index.format()
space = 30
date_index = t_list.index('2014-12-31')
dic0 = {}
result = {}
while date_index > window - 1:
    enddate = t_list[date_index]
    corr = shrinkage(enddate, data, theta=250, window = 250)
    tmp = corr.as_matrix()
    tmp =  matlab.double(tmp.tolist()) #change python np.array to matlab double
    T8,Rpm,Adjv,Dpm,Mv,Z = eng.DBHT(tmp, nargout=6)
    
    result[enddate] = {}
    result[enddate]['clusterings'] = {}
    result[enddate]['PMFG'] = Rpm
    result[enddate]['bubble_cluster_membership'] = Adjv
    result[enddate]['PMFG_shortest_path_length_matrix'] = Dpm
    result[enddate]['bubble_membership_mattrix'] = Mv
    result[enddate]['DBHT_hierarchy'] = Z
      
    tmp_tree = np.unique(T8)
    tree_membership = np.array(T8)
    #Calculate dic0
    nclusters = len(tree)
    dic0[enddate] = (nclusters, corr.shape[0])
      
    for i in tmp_tree:
        trees[enddate]['clusterings'][i] = [corr.index[A] for A in np.where(tree_membership == i)[0].tolist()]
        
    date_index -= space
    print enddate, dic0[enddate]