"""We tried numpy, numba and TensorFlow for speed improvements. On average,
pandas took about 8 seconds, numpy took about 0.2 seconds, TensorFlow took
about 0.08 seconds and numba took 0.007 seconds. The biggest improvement we
achieved was about 1500 times faster with numba compared to pandas.

In our experiments TensorFlow driven by GPU was not stably performing. We had 
achieved at most 1000 times' speedup compared to pandas. The speed improvements
 varied when Tensorflow was used in different hardware environments.
"""


import json
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt


from bld.project_paths import project_paths_join as ppj

def plot_regressions(runtimes, model_name):
    sns.set(color_codes=True)
    sns.regplot(x='obs', y='times', data=runtimes, order=3, marker='.')
    plt.savefig(ppj("OUT_FIGURES", "timing_{}.png".format(model_name)))
        
if __name__ == "__main__":
    model_name = sys.argv[1]
    model = json.load(open(ppj("IN_MODEL_SPECS", model_name + ".json"), encoding="utf-8"))

    # Load locations after each round
    with open(ppj("OUT_ANALYSIS", "timing_{}.pickle".format(model_name)), "rb") as in_file:
        runtimes = pickle.load(in_file)

    plot_regressions(runtimes, model_name)
