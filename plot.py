import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, ast, getopt, types
import easygui
import cPickle as pickle



def main(argv):
    files = easygui.fileopenbox("Please select files to plot", "", default = "experiments/", filetypes= "*.p", multiple=True)
    tags = easygui.multenterbox("Please enter the tags for the files you selected", "", fields=[i[i.rfind("/"):] for i in files], values=[])
    df = pd.DataFrame.from_dict(pickle.load(open(files[0],'rb')), orient='index')
    df = df.sort_index()
    df.index.name = 'Date'
    df.columns = [tags[0]]
    for i in range(1,len(files)):
        tempdf = pd.DataFrame.from_dict(pickle.load(open(files[i],'rb')), orient='index')
        tempdf = tempdf.sort_index()
        df[tags[i]] = tempdf
    filename = easygui.filesavebox("Please enter where to save the plot.","", default=files[0][:files[0].rfind("/")])
    df.plot(figsize=(20,8))
    plt.savefig(filename, bbox_inches='tight')
if __name__ == '__main__':
    main(sys.argv[1:])
