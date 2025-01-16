#!/usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("pickles", type=str, nargs="+")
args = parser.parse_args()

data = None
for pkl in args.pickles:
    if data is None:
        data = pd.read_pickle(pkl)
    else:
        data = pd.concat( (data, pd.read_pickle(pkl)), ignore_index=True)

data.dropna(inplace=True)

for idx, row in data.iterrows():
    # make config names shorter
    data.loc[idx, "config"] = os.path.basename(row["config"])


# Agregated output table
# config | #throws | mgt_obj | diag(Igt_obj) | mean(x_obj)  |  stddev(x_obj) |  mean(eps)  |  max(eps)  |  mean(Psi)  |  max(Psi)
# Aggregation
aggregated_data = data.groupby('config').agg(
    mgt_obj=('mgt_obj', 'first'),
    diag_Igt_obj=('Igt_obj', lambda x: np.diag(x.iloc[0])),  # Assuming diagonal is the first entry
    throws=('id', 'count'),
    mean_eps=('eps', 'mean'),
    max_eps=('eps', 'max'),
    mean_Psi=('Psi', 'mean'),
    max_Psi=('Psi', 'max'),
    mean_x_obj=('x_obj', 'mean'),
    stddev_x_obj=('x_obj', lambda x: np.asarray(x.to_list()).std(axis=0)),
).reset_index().sort_values(by="mgt_obj").reset_index(drop=True)

aggregated_rounded = aggregated_data.copy()
aggregated_rounded["mgt_obj"] = aggregated_rounded["mgt_obj"].map(lambda x: round(x, 3))
aggregated_rounded["diag_Igt_obj"] = aggregated_rounded["diag_Igt_obj"].map(lambda x: (1e6*x).round(0))
aggregated_rounded["mean_x_obj"] = aggregated_rounded["mean_x_obj"].map(lambda x: (1e3*x).round(1))
aggregated_rounded["stddev_x_obj"] = aggregated_rounded["stddev_x_obj"].map(lambda x: (1e3*x).round(2))
aggregated_rounded[["mean_eps", "max_eps"]] = (100*aggregated_rounded[["mean_eps", "max_eps"]]).map('{:.1f}\\%'.format)
aggregated_rounded[["mean_Psi", "max_Psi"]] = (180/np.pi*aggregated_rounded[["mean_Psi", "max_Psi"]]).map('{:.1f}\\si{{\\degree}}'.format)

# Display the resulting DataFrame
print(aggregated_rounded.to_latex(index=False, float_format="%.3f"))
