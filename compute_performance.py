from portfolio_and_performance import *
import cPickle as pickle
import os
import sys, ast, getopt, types
import easygui
import cPickle as pickle
from trees import importdata

def main(argv):
    retfilename = ''
    experiment_name = ''
    pricefile = ''
    divfile = ''
    cprfile = ''
    markowitz = '' # True or False
    return_target = ''
    risk_free_asset = '' # True or False
    risk_free_rate = ''
    allow_short_selling = '' # True or False
    long_cap = ''
    short_cap = ''

    try:
        opts, args = getopt.getopt(argv,"",[
        "experiment_name=",
        "retfilename=",
        "pricefile=",
        "divfile=",
        "cprfile=",
        "markowitz=",
        "return_target=",
        "risk_free_asset=",
        "risk_free_rate=",
        "allow_short_selling=",
        "long_cap=",
        "short_cap=",
        "window="
        ])
    except getopt.GetoptError:
        print 'compute_performance.py --retfilename <inputfile>...'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "--experiment_name":
            experiment_name = arg
        elif opt == "--retfilename":
            retfilename = arg
        elif opt == "--pricefile":
            pricefile = arg
        elif opt == "--divfile":
            divfile = arg
        elif opt == "--cprfile":
            cprfile = arg
        elif opt == "--markowitz":
            markowitz = eval(arg)
        elif opt == "--return_target":
            return_target = float(arg)
        elif opt == "--risk_free_asset":
            risk_free_asset = eval(arg)
        elif opt == "--risk_free_rate":
            risk_free_rate = float(arg)
        elif opt == "--allow_short_selling":
            allow_short_selling = eval(arg)
        elif opt == "--long_cap":
            if str(arg)=='None':
                long_cap = None
            else:
                long_cap = float(arg)
        elif opt == "--short_cap":
            if str(arg)=='None':
                short_cap = None
            else:
                short_cap = float(arg)
        elif opt == "--window":
            window = int(arg)
    if experiment_name == '':
        experiment_name = easygui.diropenbox("Please select your experiment folder.","", default = "experiments")
        experiment_name = experiment_name[experiment_name.rfind("/")+1:]
    if retfilename == '':
        retfilename = easygui.fileopenbox("Please select the ret file.","", default = "data/WRDS/", filetypes="*.csv")
    if pricefile == '':
        pricefile = easygui.fileopenbox("Please select the price file.","", default = "data/WRDS/", filetypes="*.csv")
    if divfile == '':
        divfile = easygui.fileopenbox("Please select the div file.","", default = "data/WRDS/", filetypes="*.csv")
    if cprfile == '':
        cprfile = easygui.fileopenbox("Please select the cpr file.","", default = "data/WRDS/", filetypes="*.csv")
    if markowitz == '':
        markowitz = easygui.ynbox("Are we using Markowitz?","")
    ret = importdata(retfilename)
    price = importdata(pricefile)
    div = importdata(divfile)
    cpr = importdata(cprfile)
    if True:
        universe_list = easygui.fileopenbox("Please select the universe file to compute performance.","",\
        default = "experiments/", filetypes="*.p", multiple=True)
        window = easygui.integerbox("Please enter the window size for the cov matrix.","",\
        default = 250, lowerbound = 10, upperbound = 50000)
        for f in universe_list:
            universe_specs = "_"+f[f.rfind("universes_"):-2]
            performance_specs = "_markowitz_"+str(markowitz)+"_covwindow_"+str(window)+universe_specs
            if not os.path.isfile("experiments/"+experiment_name+"/output/performance/performance_central_"+performance_specs+".p"):
                universes = pickle.load(open(f,'rb'))
                performance = clustering_performance(price=price, ret=ret, div=div, cpr=cpr, universes=universes, weighted=markowitz, window=window)
                pickle.dump(performance['peripheral'], open("experiments/"+experiment_name+"/output/performance/performance_peripheral_"+performance_specs+".p", "wb"))
                pickle.dump(performance['central'], open("experiments/"+experiment_name+"/output/performance/performance_central_"+performance_specs+".p", "wb"))
        if easygui.ynbox("Are we also doing benchmark?",""):
            universe = universe_list[0]
            universe_specs = "_"+universe[universe.rfind("universes_"):-2]
            performance_specs = "_markowitz_"+str(markowitz)+"_covwindow_"+str(window)+universe_specs
            universes = pickle.load(open(universe,'rb'))
            weighted_benchmark, unweighted_benchmark = benchmark_performance(price=price, ret=ret, div=div, cpr=cpr, universes=universes, window=window)
            pickle.dump(weighted_benchmark, open("experiments/"+experiment_name+"/output/performance/weighted_benchmark_"+performance_specs+".p", "wb"))
            pickle.dump(unweighted_benchmark, open("experiments/"+experiment_name+"/output/performance/unweighted_benchmark_"+performance_specs+".p", "wb"))

    else:
        pass

if __name__ == '__main__':
    main(sys.argv[1:])
