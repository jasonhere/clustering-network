from trees import *
import sys, getopt
import os
import cPickle as pickle
# from time import gmtime, strftime
import warnings
import tables
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

def main(argv):
    experiment_name = ''
    retfilename = ''
    space = 10
    startdate = '1985-01-04'
    enddate = '2016-12-30'
    datefile = ''
    window = 250
    shrinkage = None
    theta = 125
    tau = 125
    average=False

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
        "theta="
        ])
    except getopt.GetoptError:
        print 'create_trees.py --retfilename <inputfile>...'
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

    if experiment_name == '':
        experiment_name = currenttime
    if average:
        shrinkage = 'Exponential'

    directory = "experiments/"+ experiment_name#+"_"+currenttime
    if not os.path.exists(directory):
        os.makedirs(directory)
    ret = importdata(retfilename)

    if not os.path.exists(directory+"/output"):
        os.makedirs(directory+"/output")
    for subfolder in ['all_correlation_matrices', 'correlation_matrices', 'filters', 'DBHTs', 'clusterings', 'clustering_analysis','universes','performance']:
        if not os.path.exists(directory+"/output/"+subfolder):
            os.makedirs(directory+"/output/"+subfolder)

    corr_matrix_specs = "_shrinkage_"+str(shrinkage)+"_window_"+str(window)+"_theta_"+str(theta)
    all_corr_store = "experiments/"+experiment_name+"/output/all_correlation_matrices/all_correlation_matrices"+corr_matrix_specs+".h5"
    adjusted_R_store = "experiments/"+experiment_name+"/output/all_correlation_matrices/adjusted_R"+corr_matrix_specs+".h5"
    allcorr = pd.HDFStore(all_corr_store)
    adjusted_R = pd.HDFStore(adjusted_R_store)
    if not allcorr.keys():
        allcorr, adjusted_R = all_corr(ret=ret, window=window, shrinkage=shrinkage,exp_shrinkage_theta = theta)
        store = pd.HDFStore(all_corr_store)
        for key in allcorr.keys():
            store[key] = allcorr[key]
        store.close()
        store = pd.HDFStore(adjusted_R_store)
        for key in allcorr.keys():
            store[key] = adjusted_R[key]
        store.close()

    corr_dict_specs = corr_matrix_specs + "_space_"+str(space)+"_tau_"+str(tau)+"_average_"+str(average)
    corr_dict_store = "experiments/"+experiment_name+"/output/correlation_matrices/correlation_matrices"+corr_dict_specs+".h5"
    corr_dict = pd.HDFStore(corr_dict_store)
    if not corr_dict.keys():
        if datefile:
            dates = open(datefile,"rb").read().split('\n')
            dates = pd.to_datetime(dates).strftime("%Y-%m-%d")
            dates.sort()
        else:
            alldates = ret.index
            alldates.sort_values()
            dates = compute_dates(alldates, startdate, enddate, space)
        if average:
            corr_dict= rolling_corr(allcorr, dates, adjusted_R=adjusted_R, average=average, tau=tau)
        else:
            corr_dict = rolling_corr(allcorr, dates, average=average, tau=tau)
        store = pd.HDFStore(corr_dict_store)
        for key in corr_dict.keys():
            store[key] = corr_dict[key]
        store.close()

    trees = MST(corr_dict)
    pickle.dump(trees,open("experiments/"+experiment_name+"/output/filters/MSTs"+corr_dict_specs+".p","wb"))

if __name__ == "__main__":
    main(sys.argv[1:])
