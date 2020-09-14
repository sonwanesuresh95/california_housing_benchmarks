import os
import time

import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# preprocess data
def preprocess(housing):
    # separating target variable
    target = np.array(housing.pop('median_house_value'))

    # applying one-hot-encoding to categorical variable: housing['ocean_proximity']
    ocean_proximity = housing.pop('ocean_proximity')
    ocean_proximity = pd.get_dummies(ocean_proximity, drop_first=True)

    # Imputing missing values
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_housing = pd.DataFrame(imp.fit_transform(housing), columns=housing.columns)

    # Scaling/Standardizing Features and targets i.e. mean=0 and std_dev=1
    std_features = StandardScaler()
    std_target = StandardScaler()
    housing = std_features.fit_transform(imputed_housing)
    target = std_target.fit_transform(target.reshape(-1, 1))  # overwriting standardizer for target

    # Saving scaler after training
    filename = 'std_features.pkl'
    pickle.dump(std_features, open('./models/feature_scaling/' + filename, 'wb'))
    filename = 'std_target.pkl'
    pickle.dump(std_target, open('./models/feature_scaling/' + filename, 'wb'))

    # concatenating imputed dataset and ocean_proximity
    features = np.hstack([housing, ocean_proximity])

    return features, target.reshape(target.shape[0])

