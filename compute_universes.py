from portfolio_and_performance import *
import cPickle as pickle
import os
import sys, getopt

def main(argv):
    retfilename = ''
    experiment_name = ''
    space = 10
    startdate = '1985-01-04'
    enddate = '2016-12-30'
    datefile = ''
    window = 250
    shrinkage = None
    theta = 125
    tau = 125
    average=False
    clustering_algorithm = "Newman"
    n_of_clusters = None
    quantile = 0.0

    try:
        opts, args = getopt.getopt(argv,"",[
        "experiment_name=",
        "space=",
        "window=",
        "shrinkage=",
        "theta=",
        "average=",
        "tau=",
        "clustering_algorithm=",
        "n_of_clusters=",
        "quantile="
        ])
    except getopt.GetoptError:
        print 'compute_universes.py --retfilename <inputfile>...'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "--experiment_name":
            experiment_name = arg
        elif opt == "--space":
            space = int(arg)
        elif opt == "--window":
            window = int(arg)
        elif opt == "--shrinkage":
            shrinkage = arg
        elif opt == "--theta":
            theta = int(arg)
        elif opt == "--tau":
            tau = int(arg)
        elif opt == "--average":
            average = eval(arg)
        elif opt == "--clustering_algorithm":
            clustering_algorithm = arg
        elif opt == "--n_of_clusters":
            if str(arg)=='None':
                n_of_clusters = None
            else:
                n_of_clusters = int(arg)
        elif opt == "--quantile":
            quantile = float(arg)

    corr_matrix_specs = "_shrinkage_"+str(shrinkage)+"_window_"+str(window)+"_theta_"+str(theta)
    tree_specs = corr_matrix_specs + "_space_"+str(space)+"_tau_"+str(tau)+"_average_"+str(average)
    clustering_specs = "_algorithm_"+str(clustering_algorithm)+"_n_of_clusters_"+str(n_of_clusters)+tree_specs
    universe_specs = "_quantile_"+str(quantile)+clustering_specs
    if os.path.isfile("experiments/"+experiment_name+"/output/universes/universes_centrality_"+str('betweenness')+clustering_specs+".p"):
        print "Universes already exists!"
    elif os.path.isfile("experiments/"+experiment_name+"/output/clusterings/clusterings"+clustering_specs+".p"):
        clusterings = pickle.load(open("experiments/"+experiment_name+"/output/clusterings/clusterings"+clustering_specs+".p", "rb"))
        if clustering_algorithm != "DBHT":
            filters = pickle.load(open("experiments/"+experiment_name+"/output/filters/MSTs"+tree_specs+".p", "rb"))
        else:
            filters = pickle.load(open("experiments/"+experiment_name+"/output/filters/PMFGs"+tree_specs+".p", "rb"))
        for measure in ["betweenness","degree","closeness"]:
            print("Computing universes with centrality measure %s and quantile= %s" %(measure, quantile))
            universes = clustering_universe(filters, clusterings, measure, quantile=quantile)
            pickle.dump(universes, open("experiments/"+experiment_name+"/output/universes/universes_centrality_"+str(measure)+clustering_specs+".p","wb"))
    else:
        print "The clustering with these specs does not exist!"

if __name__ == "__main__":
    main(sys.argv[1:])
