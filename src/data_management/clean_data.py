"""This function cleans the data from the chs_data.dta. It
   prepares the dataset we use for further analysis.
"""

import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj


def save_data(sample):
    """
    Save clean data as .csv file.
    """
    sample.to_csv(ppj("OUT_DATA", "data_clean.csv"), sep=",")



if __name__ == "__main__":
    data = pd.read_stata(ppj("IN_DATA", "chs_data.dta"))
    data.replace(-100, np.nan, inplace=True)
    data = data.query("age == 0")
    data.reset_index(inplace=True)
    data = data["weightbirth"]
    data.fillna(data.mean(), inplace=True)
    save_data(data)
