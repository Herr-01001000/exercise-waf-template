"""In this part of plotting, we used the regplot from seaborn to draw
   regression plots in different polynomial orders with the runtimes
   of fast_batch_update on the y-axis and the number of observations
   (from 1 to 2207) on the x-axis.
"""


import json
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt


from bld.project_paths import project_paths_join as ppj

def plot_regressions(runtimes, order, model_name):
    """
    This function is used for regression plots.
    """
    sns.set(color_codes=True)
    sns.regplot(x='obs', y='times', data=runtimes, order=order, marker='.')
    plt.savefig(ppj("OUT_FIGURES", "timing_{}_order_{}.png".format(model_name, order)))
        
if __name__ == "__main__":
    model_name = sys.argv[1]
    model = json.load(open(ppj("IN_MODEL_SPECS", model_name + ".json"), encoding="utf-8"))

    # Load locations after each round
    with open(ppj("OUT_ANALYSIS", "timing_{}.pickle".format(model_name)), "rb") as in_file:
        runtimes = pickle.load(in_file)

    for order in model["order"]:
        plot_regressions(runtimes, order, model_name)
