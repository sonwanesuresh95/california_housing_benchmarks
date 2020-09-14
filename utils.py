import time
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


# Function to produce benchmarks
def benchmark(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    # save train.csv and test.csv

    model_name = str(model).split('(')[0]
    print('\nNow training {} model...'.format(model_name))
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    mse = mean_squared_error(y_pred, y_test)
    return model, round(mse, 4), round(train_time, 4), round(inference_time, 4) / len(y_test)


# update benchmark
def update(d: dict, benchmarks):
    model, mse, train_time, inference_time = benchmarks
    model_name = str(model).split('(')[0]
    d['models'].append(model_name)
    d['mses'].append(mse)
    d['training_times'].append(train_time)
    d['inference_times'].append(inference_time)
    print("\nBenchmark Updated Succesfully.")
    return d


# print data
def print_data(model_name, mse, train_time, inference_time):
    string = '''\nFinished training {0}.
    \tBenchmarks are
    \tmse = {1}
    \ttraining time = {2} seconds
    \tinference time = {3} seconds'''.format(model_name, mse, train_time, inference_time)
    return string


# train function
def train_on(model, X, y, bench_dict):
    # train model and update benchmark
    benchmarks = benchmark(model, X, y)

    # printing data
    model, mse, train_time, inference_time = benchmarks
    model_name = str(model).split('(')[0]
    print(print_data(model_name, mse, train_time, inference_time))

    # Saving model after training
    filename = model_name + '.pkl'
    pickle.dump(model, open('./models/' + filename, 'wb'))
    print('\n{} model saved Successfully.'.format(model_name))

    # updating benchmark
    update(bench_dict, benchmarks)


# API for inference
def predict_instance(feature):
    prediction = None
    test = pd.read_csv('./data/test.csv')
    if feature == 'example1':
        prediction = sub_predict_instance(test[:1])
    elif feature == 'example2':
        prediction = sub_predict_instance(test[1:2])
    elif feature == 'example3':
        prediction = sub_predict_instance(test[2:3])
    elif feature == 'example4':
        prediction = sub_predict_instance(test[3:4])
    elif feature == 'example5':
        prediction = sub_predict_instance(test[4:5])
    return prediction


# sub-API for inference
def sub_predict_instance(feature):
    std_features = pickle.load(open('./models/feature_scaling/std_features.pkl', 'rb'))
    std_target = pickle.load(open('./models/feature_scaling/std_target.pkl', 'rb'))
    model = pickle.load(open('./models/AdaBoostRegressor.pkl', 'rb'))
    scaled_feature = std_features.transform(feature.iloc[:, :8])
    features = np.hstack([scaled_feature, feature.iloc[:, 8:-1]])
    prediction = model.predict(features)
    prediction = std_target.inverse_transform(prediction)
    return prediction
