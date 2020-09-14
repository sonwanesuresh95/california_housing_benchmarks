import os
import pickle
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from preprocessing import preprocess
from utils import benchmark, print_data, update, train_on

# for calculating script execution time
start_time = time.time()
# benchmark dictionary
bench_dict = {"models": [],
              "mses": [],
              "training_times": [],
              "inference_times": []}

# set seed
np.random.seed(0)

# load and preprocess dataset
housing = pd.read_csv('data/housing.csv')
features, target = preprocess(housing)

# Training Linear Regression Model
linreg = LinearRegression(n_jobs=-1)
train_on(linreg, features, target, bench_dict)
# print('\nUpdated benchmark {}'.format(bench_dict))

# Ridge Regression (L2 regularized linear regression)
ridge = Ridge(alpha=0.1)
train_on(ridge, features, target, bench_dict)
# print('\nUpdated benchmark {}'.format(bench_dict))

# Lasso Regression (L1 regularized linear regression)
lasso = Lasso(alpha=0.01)
train_on(lasso, features, target, bench_dict)
# print('\nUpdated benchmark {}'.format(bench_dict))

# SVR
svr = SVR()
train_on(svr, features, target, bench_dict)
# print('\nUpdated benchmark {}'.format(bench_dict))

# Kernel Ridge Regression (L2 Reg with kernel trick)
kr = KernelRidge(0.1)
train_on(kr, features, target, bench_dict)
# print('\nUpdated benchmark {}'.format(bench_dict))

# Gaussian Process Regression
gpr = GaussianProcessRegressor()
train_on(gpr, features, target, bench_dict)
# print('\nUpdated benchmark {}'.format(bench_dict))

# DecisionTreeRegressor
dtregressor = DecisionTreeRegressor()
train_on(dtregressor, features, target, bench_dict)
# print('\nUpdated benchmark {}'.format(bench_dict))

# RandomForestRegressor
rndfrst = RandomForestRegressor(n_estimators=200)
train_on(rndfrst, features, target, bench_dict)
# print('\nUpdated benchmark {}'.format(bench_dict))

# Adaboost classifier (Iteratively fits n models on underfit data)
adbclf = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=200)
train_on(adbclf, features, target, bench_dict)
# print('\nUpdated benchmark {}'.format(bench_dict))

# Gradient Boosting Regressor (Minimizes Residual errors)
gbreg = GradientBoostingRegressor(n_estimators=200)
train_on(gbreg, features, target, bench_dict)
# print('\nUpdated benchmark = {}'.format(bench_dict))

# save benchmark
with open('benchmark.json', 'w') as f:
    json.dump(bench_dict, f)

print('Benchmarks are \n{}'.format(pd.DataFrame(bench_dict)))
print('\nTotal time taken by train.py script is {} seconds'.format(round(time.time() - start_time), 4))
