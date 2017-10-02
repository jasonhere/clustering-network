import os

retfilename = 'data/WRDS/SP500_ret_1985_tickers.csv'
experiment_name = 'SP500_1985_exponential_no_average_weight_plus_1'
space = 20
startdate = '1988-01-04'
enddate = '2016-12-30'
datefile = ''
window = 250
shrinkage = "Exponential"
theta = window/3
tau = theta
average=False
clustering_algorithm = "Newman"
n_of_clusters = None
quantile = 0.0

for window in [1000,250,500]:
    theta = int(round(float(window)/3))
    tau = theta
    for clustering_algorithm in ['Newman','ClausetNewman','DBHT']:
        systemscript = "python construct_clusterings.py \
        --retfilename %s \
        --experiment_name %s \
        --space %s \
        --startdate %s \
        --enddate %s \
        --window %s \
        --shrinkage %s \
        --theta %s \
        --average %s \
        --tau %s \
        --clustering_algorithm %s \
        --n_of_clusters %s" % (retfilename, experiment_name, space, startdate, enddate, window,\
         shrinkage, theta, average, tau, clustering_algorithm, n_of_clusters)

        os.system(systemscript)

        systemscript = "python clustering_analysis.py \
        --experiment_name %s \
        --space %s \
        --window %s \
        --shrinkage %s \
        --theta %s \
        --average %s \
        --tau %s \
        --clustering_algorithm %s \
        --n_of_clusters %s" % (experiment_name, space, window,\
         shrinkage, theta, average, tau, clustering_algorithm, n_of_clusters)

        os.system(systemscript)

        systemscript = "python compute_universes.py \
        --experiment_name %s \
        --space %s \
        --window %s \
        --shrinkage %s \
        --theta %s \
        --average %s \
        --tau %s \
        --clustering_algorithm %s \
        --n_of_clusters %s \
        --quantile %s" % (experiment_name, space, window,\
         shrinkage, theta, average, tau, clustering_algorithm, n_of_clusters, quantile)

        os.system(systemscript)
