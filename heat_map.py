import argparse

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
import numpy as np


parser = argparse.ArgumentParser(description='Generate heat map.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('data', type=str, help='data file')
args = parser.parse_args()

data = pd.read_csv(args.data, sep="\t")
print(data.columns)
print(data.head())
print(data.shape)
print(data.isnull().sum())

cm = np.corrcoef(data.values.T)
hm = heatmap(cm, row_names=data.columns, column_names=data.columns)
plt.tight_layout()
plt.show()


