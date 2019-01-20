"""Draw simulated samples from two uncorrelated uniform variables
(locations in two dimensions) for two types of agents and store
them in a 3-dimensional NumPy array.

*Note:* In principle, one would read the number of dimensions etc.
from the "IN_MODEL_SPECS" file, this is to demonstrate the most basic
use of *run_py_script* only.

"""

import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj


def save_data(sample):
    sample.to_csv(ppj("OUT_DATA", "data_clean.csv"), sep=",")


if __name__ == "__main__":
    data = pd.read_stata(ppj("IN_DATA", "chs_data.dta"))
    data.replace(-100, np.nan, inplace=True)
    data = data.query("age == 0")
    data.reset_index(inplace=True)
    data = data["weightbirth"]
    data.fillna(data.mean(), inplace=True)
    save_data(data)
