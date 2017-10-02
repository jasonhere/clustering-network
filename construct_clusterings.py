from clustering import *
import cPickle as pickle
import os
import sys, getopt
from DBHT import DBHT
import pandas as pd

def main(argv):
    # system arguments
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

    try:
        opts, args = getopt.getopt(argv,"",[
        "retfilename=",
        "experiment_name=",
        "space=",
        "startdate=",
        "enddate=",
        "datefile=",
        "window=",
        "shrinkage=",
        "theta=",
        "average=",
        "tau=",
        "theta=",
        "clustering_algorithm=",
        "n_of_clusters="
        ])
    except getopt.GetoptError:
        print 'construct_clusterings.py --retfilename <inputfile>...'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "--retfilename":
            retfilename = arg
        elif opt == "--experiment_name":
            experiment_name = arg
        elif opt == "--space":
            space = int(arg)
        elif opt == "--startdate":
            startdate = arg
        elif opt == "--enddate":
            enddate = arg
        elif opt == "--datefile":
            datefile = arg
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

    # store the specifications of the corrisponding correlation dictionary, tree dictionary and clustering dictionary
    corr_matrix_specs = "_shrinkage_"+str(shrinkage)+"_window_"+str(window)+"_theta_"+str(theta)
    tree_specs = corr_matrix_specs + "_space_"+str(space)+"_tau_"+str(tau)+"_average_"+str(average)
    clustering_specs = "_algorithm_"+str(clustering_algorithm)+"_n_of_clusters_"+str(n_of_clusters)+tree_specs

    # check if the clustering already exists
    if os.path.isfile("experiments/"+experiment_name+"/output/clusterings/clusterings"+clustering_specs+".p"):
        print "The clusteirng with these specs already exists!"
    # for DBHT we use PMFG instead of MST
    elif clustering_algorithm != 'DBHT':
        if os.path.isfile("experiments/"+experiment_name+"/output/filters/IGMSTs"+tree_specs+".p"):
            IGtrees = pickle.load(open("experiments/"+experiment_name+"/output/filters/IGMSTs"+tree_specs+".p","rb"))
        else:
            #create system script for create_trees.py
            systemscript = "python create_trees.py"
            for opt, arg in opts:
                if opt not in ["--clustering_algorithm", "--n_of_clusters"]:
                    systemscript = systemscript +" "+ str(opt) +" "+ str(arg)
            os.system(systemscript)
            #read necessary trees and stuff
            trees = pickle.load(open("experiments/"+experiment_name+"/output/filters/MSTs"+tree_specs+".p","rb"))
            IGtrees = NXdicttoIGdict(trees)
            pickle.dump(IGtrees, open("experiments/"+experiment_name+"/output/filters/IGMSTs"+tree_specs+".p", "wb"))
        # construct clusteirngs
        clusterings, IGclusterings = construct_clusters(IGtrees, method=clustering_algorithm, n_of_clusters=n_of_clusters)
        pickle.dump(clusterings,open("experiments/"+experiment_name+"/output/clusterings/clusterings"+clustering_specs+".p", "wb"))
        pickle.dump(IGclusterings,open("experiments/"+experiment_name+"/output/clusterings/IGclusterings"+clustering_specs+".p", "wb"))
    else:
        # for DBHT, the tree_specs can only be different in the specs of PMFG (i.e. tree_specs) in n_of_clusters!=none.
        # still need to find a way to add the ability to deal with when n_of_clusters != None
        if os.path.isfile("experiments/"+experiment_name+"/output/filters/IGPMFGs"+tree_specs+".p"):
            pass
            #find a way to do n_of_clusters!=None
        else:
            if not os.path.isfile("experiments/"+experiment_name+"/output/correlation_matrices/correlation_matrices"+tree_specs+".h5"):
                # prepare systemscript for creating PMFG
                systemscript = "python create_trees.py"
                for opt, arg in opts:
                    if opt not in ["--clustering_algorithm", "--n_of_clusters"]:
                        systemscript = systemscript +" "+ str(opt) +" "+ str(arg)
                os.system(systemscript)
            corr_dict = pd.HDFStore("experiments/"+experiment_name+"/output/correlation_matrices/correlation_matrices"+tree_specs+".h5")
            dbht_results = DBHT(corr_dict)
            IGPMFGs = NXdicttoIGdict(dbht_results['PMFG'])
            # create clusterings in IG format
            IGclusterings = {}
            for i in IGPMFGs.keys():
                IGclusterings[i] = ig.VertexClustering(IGPMFGs[i],createlabel(dbht_results['DBHT_clusterings'][i], corr_dict[i].columns.values)[1])
            pickle.dump(IGclusterings, open("experiments/"+experiment_name+"/output/clusterings/IGclusterings"+clustering_specs+".p", "wb"))
            pickle.dump(IGPMFGs, open("experiments/"+experiment_name+"/output/filters/IGPMFGs"+tree_specs+".p", 'wb'))
            pickle.dump(dbht_results['PMFG'], open("experiments/"+experiment_name+"/output/filters/PMFGs"+tree_specs+".p", 'wb'))
            pickle.dump(dbht_results['DBHT_clusterings'], open("experiments/"+experiment_name+"/output/clusterings/clusterings"+clustering_specs+".p", "wb"))
            pickle.dump(dbht_results['bubble_cluster_membership_matrix'], open("experiments/"+experiment_name+"/output/DBHTs/bubble_cluster_membership_matrix"+tree_specs+".p", 'wb'))
            pickle.dump(dbht_results['PMFG_shortest_path_length_matrix'], open("experiments/"+experiment_name+"/output/DBHTs/PMFG_shortest_path_length_matrix"+tree_specs+".p", 'wb'))
            pickle.dump(dbht_results['DBHT_hierarchy'], open("experiments/"+experiment_name+"/output/DBHTs/DBHT_hierarchy"+tree_specs+".p", 'wb'))
            pickle.dump(dbht_results['bubble_membership_matrix'], open("experiments/"+experiment_name+"/output/DBHTs/bubble_membership_matrix"+tree_specs+".p", 'wb'))




if __name__ == "__main__":
    main(sys.argv[1:])
