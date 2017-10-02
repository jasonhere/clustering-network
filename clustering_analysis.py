from clustering import *
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
        print 'clustering_analysis.py --retfilename <inputfile>...'
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
            if str(arg)=='None':
                n_of_clusters = None
            else:
                n_of_clusters = int(arg)
    corr_matrix_specs = "_shrinkage_"+str(shrinkage)+"_window_"+str(window)+"_theta_"+str(theta)
    tree_specs = corr_matrix_specs + "_space_"+str(space)+"_tau_"+str(tau)+"_average_"+str(average)
    clustering_specs = "_algorithm_"+str(clustering_algorithm)+"_n_of_clusters_"+str(n_of_clusters)+tree_specs
    if os.path.isfile("experiments/"+experiment_name+"/output/clusterings/clusterings"+clustering_specs+".p"):
        if (not os.path.isfile("experiments/"+experiment_name+"/output/clustering_analysis/ARI"+clustering_specs+".p")) or \
        (not os.path.isfile("experiments/"+experiment_name+"/output/clustering_analysis/ARImatrix"+clustering_specs+".p")) or\
        (not os.path.isfile("experiments/"+experiment_name+"/output/clustering_analysis/avg_cluster_diameter"+clustering_specs+".p")) or\
        (not os.path.isfile("experiments/"+experiment_name+"/output/clustering_analysis/n_clusters"+clustering_specs+".p")):
            print "Conducting clustering analysis..."
            clusterings = pickle.load(open("experiments/"+experiment_name+"/output/clusterings/clusterings"+clustering_specs+".p", "rb"))
            IGclusterings = pickle.load(open("experiments/"+experiment_name+"/output/clusterings/IGclusterings"+clustering_specs+".p", "rb"))
            if clustering_algorithm != 'DBHT':
                IGfilters = pickle.load(open("experiments/"+experiment_name+"/output/filters/IGMSTs"+tree_specs+".p", "rb"))
            else:
                IGfilters = pickle.load(open("experiments/"+experiment_name+"/output/filters/IGPMFGs"+tree_specs+".p", "rb"))
            if not os.path.isfile("experiments/"+experiment_name+"/output/clustering_analysis/ARI"+clustering_specs+".p"):
                ARI = movingARI(IGclusterings)
                pickle.dump(ARI, open("experiments/"+experiment_name+"/output/clustering_analysis/ARI"+clustering_specs+".p","wb"))
            elif not os.path.isfile("experiments/"+experiment_name+"/output/clustering_analysis/ARImatrix"+clustering_specs+".p"):
                arimatrix = ARImatrix(IGclusterings)
                pickle.dump(arimatrix, open("experiments/"+experiment_name+"/output/clustering_analysis/ARImatrix"+clustering_specs+".p","wb"))
                fig = plot_ari_matrix(arimatrix, IGclusterings)
                fig.savefig("experiments/"+experiment_name+"/output/clustering_analysis/ARImatrix"+clustering_specs+".pdf", bbox_inches='tight')
            elif not os.path.isfile("experiments/"+experiment_name+"/output/clustering_analysis/avg_cluster_diameter"+clustering_specs+".p"):
                avg_cluster_diameter = average_cluster_diameter(IGfilters, IGclusterings)
                pickle.dump(avg_cluster_diameter, open("experiments/"+experiment_name+"/output/clustering_analysis/avg_cluster_diameter"+clustering_specs+".p","wb"))
            elif n_of_clusters==None:
                n_clusters = number_of_clusters(clusterings)
                pickle.dump(n_clusters,open("experiments/"+experiment_name+"/output/clustering_analysis/n_clusters"+clustering_specs+".p","wb"))

    else:
        print "The clustering with these specs does not exist!"

if __name__ == "__main__":
    main(sys.argv[1:])
