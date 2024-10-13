import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hpsklearn import HyperoptEstimator, any_regressor


parser = argparse.ArgumentParser(description='Search for a best regression model.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('data', type=str, help='data file')
args = parser.parse_args()
data = pd.read_csv(args.data, sep="\t")

print(data.columns)
print(data.head())
print(data.shape)
print(data.isnull().sum())
print()

# prepare features

target = data[['diff']]
features = data[["t1","t2","t3","w1","w2","w3"]]

X = features
y = target

print(X.shape)
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# regression
regressor = HyperoptEstimator(regressor=any_regressor('my any regressor'),
                              max_evals=1000,
                              trial_timeout=20,
                              preprocessing=[],
                              n_jobs=6,
                              verbose=True)

regressor.fit(X_train, y_train)
print(regressor.best_model())

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# MSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')
print()

# MAE
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')
print()

# coefficient of determination (fraction of response variance captured by the model)
# R^2 is 0...1 on the training set, but can become negative on the test set.
# Decrease of R^2 on the test set means overfitting on the training set.
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f'R^2 train: {train_r2:.3f}, R^2 test: {test_r2:.3f}')
print()

# plot actual-vs-predicted values

x_max = np.max([np.max(y_train), np.max(y_test)])
x_min = np.min([np.min(y_train), np.min(y_test)])

fig, axs = plt.subplots(2, 3, figsize=(12, 9), sharey='row')

ax1 = axs[0,0]
ax2 = axs[0,1]
ax3 = axs[1,0]
ax4 = axs[1,1]
ax5 = axs[1,2]

ax1.set_ylim(x_min, x_max)
ax2.set_ylim(x_min, x_max)
ax1.set_xlim(x_min, x_max)
ax2.set_xlim(x_min, x_max)

ax1.scatter(y_train,
            y_train_pred,
            #(y_train_pred - y_train),
            c='steelblue',
            marker='o',
            edgecolor='white',
            label='Training data')
ax1.set_xlabel('Actual values of diff')
ax2.scatter(y_test,
            y_test_pred,
            #(y_test_pred - y_test),
            c='limegreen', marker='s',
            edgecolor='white',
            label='Test data')
ax1.set_xlabel('Actual values of diff')
ax1.set_ylabel('Predicted values of diff')

ax3.scatter(data[["real_average"]],
            data[["perfect_average"]],
            c='steelblue',
            marker='o',
            edgecolor='white',
            label='All data')
ax3.set_xlabel('real_average (with noise)')
ax3.set_ylabel('perfect average (no noise)')

ax4.scatter(data[["real_average"]],
            data[["average"]],
            c='steelblue',
            marker='o',
            edgecolor='white',
            label='All data')
ax4.set_xlabel('real_average (with noise)')
ax4.set_ylabel('average')


def compute_average_y(data_rows):
    result = np.zeros((len(data_rows),))
    for i,row in enumerate(data_rows):
        result[i] = row[0]*row[3] + row[1]*row[4] + row[2]*row[5]
    return result


ax5.scatter(compute_average_y(X.to_numpy()) + y.to_numpy().flatten(),
            compute_average_y(X.to_numpy()) + regressor.predict(X).flatten(),
            c='steelblue',
            marker='o',
            edgecolor='white',
            label='All data')
ax5.set_xlabel('real_average (with noise)')
ax5.set_ylabel('learned average')

for ax in (ax1, ax2, ax3, ax4, ax5):
    ax.legend(loc='upper left')
    if ax.collections:
        scatter = ax.collections[0]
        offsets = scatter.get_offsets()
        hx = offsets[:, 0]
        for coeff in (0.9,1,1.1):
            hy = coeff * hx
            ax.plot(hx, hy, color='grey', lw=1 if coeff == 1 else 0.5)

plt.tight_layout()
plt.show()

