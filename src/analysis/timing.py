"""We tried numpy, numba and TensorFlow for speed improvements. On average,
pandas took about 8 seconds, numpy took about 0.2 seconds, TensorFlow took
about 0.08 seconds and numba took 0.007 seconds. The biggest improvement we
achieved was about 1500 times faster with numba compared to pandas.

In our experiments TensorFlow driven by GPU was not stably performing. We had 
achieved at most 1000 times' speedup compared to pandas. The speed improvements
 varied when Tensorflow was used in different hardware environments.
"""

import sys
import json
import logging
import pickle
import numpy as np
import pandas as pd
from time import time

from src.model_code.update import fast_batch_update
from bld.project_paths import project_paths_join as ppj


data = np.loadtxt(ppj("OUT_DATA", "data_clean.csv"), delimiter=",")

# fix dimensions
nobs = len(data)
state_names = ["cog", "noncog", "mother_cog", "mother_noncog", "investments"]
nstates = len(state_names)

# construct initial states
states_np = np.zeros((nobs, nstates))

# construct initial covariance matrices
root_cov = np.linalg.cholesky(
    [
        [0.1777, -0.0204, 0.0182, 0.0050, 0.0000],
        [-0.0204, 0.2002, 0.0592, 0.0261, 0.0000],
        [0.0182, 0.0592, 0.5781, 0.0862, -0.0340],
        [0.0050, 0.0261, 0.0862, 0.0667, -0.0211],
        [0.0000, 0.0000, -0.0340, -0.0211, 0.0087],
    ]
)

root_covs_np = np.zeros((nobs, nstates, nstates))
root_covs_np[:] = root_cov

# construct measurements
meas_bwght_np = data[:,1]

# construct loadings
loadings_bwght_np = np.array([1.0, 0, 0, 0, 0])

# construct the variance
meas_var_bwght = 0.8    


def run_analysis(states, root_covs, measurements, loadings, meas_var):
    runtimes = pd.DataFrame(columns=['obs','times'])
    for i in range(len(states_np)):
        for j in range(11):
            start = time()
            out_states_fast, out_root_covs_fast = fast_batch_update(
                states[:i],
                root_covs[:i],
                measurements[:i],
                loadings,
                meas_var,
            )
            stop = time()
            runtimes = runtimes.append({'obs':i,'times':(stop-start)},ignore_index=True)
    
    return runtimes


if __name__ == "__main__":
    model_name = sys.argv[1]
    model = json.load(open(ppj("IN_MODEL_SPECS", model_name + ".json"), encoding="utf-8"))

    logging.basicConfig(
        filename=ppj("OUT_ANALYSIS", "log", "timing_{}.log".format(model_name)),
        filemode="w",
        level=logging.INFO
    )
    np.random.seed(model["rng_seed"])
    logging.info(model["rng_seed"])

    # Run the main analysis
    runtimes = run_analysis(states_np, root_covs_np, meas_bwght_np, loadings_bwght_np, meas_var_bwght)
    
    for i in range(len(runtimes)):
        if i % 11 == 0:
            runtimes = runtimes.drop([i])
            
    # Store list with locations after each round
    with open(ppj("OUT_ANALYSIS", "timing_{}.pickle".format(model_name)), "wb") as out_file:
        pickle.dump(runtimes, out_file)
