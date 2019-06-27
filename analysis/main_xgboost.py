import numpy as np
import pandas as pd
import random as rd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.sparse.csr

from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression, Ridge

from xgboost import XGBRegressor

import importlib
import kaggle_forecast 
importlib.reload( kaggle_forecast )

from kaggle_forecast import *

kg = KaggleData()

features = { 'lag' : [1,2,3,6,12], 'mean' : [2,3,6,12], 'shop_id' : [], 'item_id' : [], 'shop_item_id' : [], \
             'cat_id' : [], 'date_block_num' : [], 'month' : [], 'year' : [] }

X, Y = get_labels_and_features( kg, features=features )

# Get the combined features in matrix form for the train, validation and test sets
xx_raw, yy = combine_features( X, Y )

# Preprocess the features (e.g. winsorize, clip values, etc.)
xx = preprocess_features(xx_raw)

# Convert features to sparse matrices
xx_sparse = convert_to_sparse( xx )

# Replace shop_id with one-hot encodings
xx_sparse = add_one_hot_encoding( xx_sparse, "shop_id", binarizer=kg.binarizer['shop'] )

# Replace category_id with one-hot encodings
xx_sparse = add_one_hot_encoding( xx_sparse, "cat_id", binarizer=kg.binarizer['cat'] )

# Replace month with one-hot encodings
xx_sparse = add_one_hot_encoding( xx_sparse, "month", binarizer=kg.binarizer['month'] )

####################################################################################
# Create constructor

model_constructor_fun = lambda : XGBRegressor()
model_train = fit_model( model_constructor_fun, TRAIN, xx_sparse, yy, date_block_cutoff=30 )