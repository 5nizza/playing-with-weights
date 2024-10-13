import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np


parser = argparse.ArgumentParser(description='Generate basic graphs.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('data', type=str, help='data file')
args = parser.parse_args()

data = pd.read_csv(args.data, sep="\t")
print(data.columns)
print(data.head())
print(data.shape)
print(data.isnull().sum())

X = data["t1"].values
y = data["diff"].values
scatterplotmatrix(data.values, figsize=(12, 10), names=data.columns, alpha=0.5)
plt.tight_layout()
plt.show()
