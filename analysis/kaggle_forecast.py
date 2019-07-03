
import numpy as np
import pandas as pd
import random as rd
import datetime
import os

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.sparse.csr

from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

import importlib
import config 
importlib.reload( config )
from config import *

class KaggleData():

    def __init__(self):
        """Read in csv files."""

        self.load_raw_data()
        self.add_derived_features()
        self.aggregate_monthly_sales()        
        self.create_time_series()
        self.add_binarizers()


    def load_raw_data(self):
        self.csv = { 'sales' : [], 'item_cat' : [], 'item' : [], 'sub' : [], 'shops' : [], 'test' : [] }
        
        # Import all csv file data
        self.csv[ 'sales' ] = pd.read_csv( os.path.join( PROJECT_PATH, 'input/sales_train.csv') )

        # Warnings
        import warnings
        warnings.filterwarnings('ignore')

        self.csv['item_cat'] = pd.read_csv( os.path.join( PROJECT_PATH, 'input/item_categories.csv') )
        self.csv['item'] = pd.read_csv( os.path.join( PROJECT_PATH, 'input/items.csv') )
        self.csv['sub'] = pd.read_csv( os.path.join( PROJECT_PATH, 'input/sample_submission.csv') )
        self.csv['shops'] = pd.read_csv( os.path.join( PROJECT_PATH, 'input/shops.csv') )
        self.csv['test'] = pd.read_csv( os.path.join( PROJECT_PATH, 'input/test.csv') )

    def add_derived_features(self):
        # Reformat the date column
        self.csv['sales'].date = self.csv['sales'].date.apply( lambda x: datetime.datetime.strptime( x, '%d.%m.%Y' ) )

        # Add month and year columns
        self.csv['sales']['month'] = [ x.month for x in self.csv['sales'].date ]
        self.csv['sales']['year'] = [ x.year for x in self.csv['sales'].date ]
        self.csv['sales']['year_month'] = self.csv['sales'].year * 100 + self.csv['sales'].month

        # Add the item_category_id to the training set
        self.csv['sales'] = self.csv['sales'].set_index('item_id').join( self.csv['item'].set_index('item_id'))
        self.csv['sales'] = self.csv['sales'].drop('item_name', axis=1).reset_index()
        
        self.csv['test'] = self.csv['test'].set_index('item_id').join( self.csv['item'].set_index('item_id'))
        self.csv['test'] = self.csv['test'].drop('item_name', axis=1).reset_index()

        # Add a unique id for the shop + item combo
        self.csv['sales']['shop_item_id'] = self.csv['sales'].shop_id + self.csv['sales'].item_id * 100
        self.csv['test']['shop_item_id'] = self.csv['test'].shop_id + self.csv['test'].item_id * 100
        self.csv['sales']['shop_cat_id'] = self.csv['sales'].shop_id + self.csv['sales'].item_category_id * 100
        self.csv['test']['shop_cat_id'] = self.csv['test'].shop_id + self.csv['test'].item_category_id * 100

        # Add the revenue
        self.csv['sales'][ "revenue" ] = self.csv['sales'].item_price * self.csv['sales'].item_cnt_day

    def aggregate_monthly_sales(self):
        """Create a data frame with aggregated montly sales."""

        # Aggregate the monthly data
        agg_rules = {'item_price' : "mean", "revenue" : "sum", "item_cnt_day" : "sum" }
        groupby_cols = [ "year_month", 'date_block_num', "shop_item_id", "item_id", "shop_id", "item_category_id" ]
        self.monthly_shop_item = self.csv['sales'].groupby( groupby_cols ).agg( agg_rules ).reset_index()
        self.monthly_shop_item.rename( columns={ 'item_cnt_day' : 'item_cnt_month'}, inplace=True )

    def create_time_series(self):
        """Create time series of variables that we will use for prediction."""

        self.ts = { 'sales' : [], 'shop_id' : [], 'cat_id' : [], 'date_block_num' : [], 'month' : [], 'year' : [] }

        # Time series of the prices
        self.ts['item_cnt_month'] = create_pivot_ts( self.monthly_shop_item, "shop_item_id", "item_cnt_month", "sum", 'zero'  )

        # Time series of the shop and item ids
        shop_id, item_id = decompose_shop_item_id( self.ts['item_cnt_month'].columns )
        self.ts['shop_id'] = pd.DataFrame( np.vstack( [ shop_id.values ] * self.ts['item_cnt_month'].shape[0] ), \
                                                                     index=self.ts['item_cnt_month'].index )
        self.ts['shop_id'].columns = self.ts['item_cnt_month'].columns

        self.ts['item_id'] = pd.DataFrame( np.vstack( [ item_id.values ] * self.ts['item_cnt_month'].shape[0] ), \
                                                                     index=self.ts['item_cnt_month'].index )
        self.ts['item_id'].columns = self.ts['item_cnt_month'].columns        
        self.ts['shop_item_id'] = self.ts['shop_id'] + self.ts['item_id'] * 100

        # Time series for the category ids. First, we must look up the category ID for each item
        uniq_items = item_id.unique()
        cat_ids_for_uniq_ids = [ self.csv['item'][ self.csv['item'].item_id == x ].item_category_id.iloc[0] for x in uniq_items ]
        id_map = dict(zip( list(uniq_items), cat_ids_for_uniq_ids ))
        cat_ids = pd.Series( [ id_map[x] for x in item_id ] )
        self.ts['cat_id'] = pd.DataFrame( np.vstack( [ cat_ids.values ] * self.ts['item_cnt_month'].shape[0] ), \
                                                                    index=self.ts['item_cnt_month'].index )
        self.ts['cat_id'].columns = self.ts['item_cnt_month'].columns

        date_block_nums = np.array(list(range(0,self.ts['shop_id'].shape[0])))[:,np.newaxis]
        self.ts['date_block_num'] = pd.DataFrame( np.hstack( [ date_block_nums ] * self.ts['item_cnt_month'].shape[1] ), \
                                                                    index=self.ts['item_cnt_month'].index )
        self.ts['date_block_num'].columns = self.ts['item_cnt_month'].columns

        months = np.array([ x % 100 for x in sorted( self.csv['sales'].year_month.unique()) ] )[:,np.newaxis]
        self.ts['month'] = pd.DataFrame( np.hstack( [ months ] * self.ts['item_cnt_month'].shape[1] ), \
                                                           index=self.ts['item_cnt_month'].index )
        self.ts['month'].columns = self.ts['item_cnt_month'].columns

        years = np.array([ x // 100 for x in sorted( self.csv['sales'].year_month.unique()) ] )[:,np.newaxis]
        self.ts['year'] = pd.DataFrame( np.hstack( [ years ] * self.ts['item_cnt_month'].shape[1] ), \
                                                         index=self.ts['item_cnt_month'].index )
        self.ts['year'].columns = self.ts['item_cnt_month'].columns


    def add_binarizers(self):

        self.binarizer = dict()

        # Replace shop_id with one-hot encodings
        self.binarizer['shop']= LabelBinarizer(sparse_output=True)
        self.binarizer['shop'].fit( self.csv['sales'].shop_id.unique())

        # Replace category_id with one-hot encodings
        self.binarizer['cat'] = LabelBinarizer(sparse_output=True)
        self.binarizer['cat'].fit( self.csv['sales'].item_category_id.unique())

        # Replace month with one-hot encodings
        self.binarizer['month'] = LabelBinarizer(sparse_output=True)
        self.binarizer['month'].fit(np.arange(1,13))


def write_forecast_to_csv( obj, yhat_test, output_file ):
    # Format the forecast as a Pandas Series and write the output to .csv
    fcst = pd.Series( yhat_test.ravel(), index=pd.Index( obj.ts['item_cnt_month'].columns, dtype="int64") )
    fcst_df = format_forecast( obj.csv['test'], fcst, fill_na=True )

    # Write the results to the output file
    fcst_df.to_csv( os.path.join( PROJECT_PATH, 'forecasts', output_file ), index=False, header=True )


def get_lagged_features( obj, X, lags ):
    """Calculate lagged feature vectors based on the time series of monthly sales."""

    for L in lags:
        x_lag_mtx = obj.ts['item_cnt_month'].shift(periods=L-1)
        X = get_regression_vectors_from_matrix( X, x_lag_mtx, is_X=True, descrip='sales_lag_{:02}'.format(L) )
    
    return X

def get_lagged_mean_features( obj, X, means ):
    """Calculate lagged mean feature vectors based on the time series of monthly sales."""

    for M in means:
        x_mean_mtx = obj.ts['item_cnt_month'].rolling(window=M).mean()    
        X = get_regression_vectors_from_matrix( X, x_mean_mtx, is_X=True, descrip='sales_mean_{:02}'.format(M) )

    return X

def get_labels_and_features( obj, features=dict() ):
    
    # Initialize dictionaries to store the labels and features
    X = { TRAIN : [], VALID : [], TEST : [], DESC : [] }
    Y = { TRAIN : [], VALID : [], TEST : [] }

    # Create the test and train sets from the observation matrices
    Y = get_regression_vectors_from_matrix( Y, obj.ts['item_cnt_month'], is_X=False )

    # Use different sales lags as features
    if 'lag' in features:
        X = get_lagged_features( obj, X, lags=features['lag'] )

    # Use different means as features
    if 'mean' in features:
        get_lagged_mean_features( obj, X, means=features['mean'] )

    # Add the shop id as a feature
    if 'shop_id' in features:
        X = get_regression_vectors_from_matrix( X, obj.ts['shop_id'], is_X=True, descrip='shop_id' )

    # Add the item id as a feature
    if 'item_id' in features:
        X = get_regression_vectors_from_matrix( X, obj.ts['item_id'], is_X=True, descrip='item_id' )

    # Add the shop/item id as a feature
    if 'shop_item_id' in features:
        X = get_regression_vectors_from_matrix( X, obj.ts['shop_item_id'], is_X=True, descrip='shop_item_id' )

    # Add the category id as a feature
    if 'cat_id' in features:
        X = get_regression_vectors_from_matrix( X, obj.ts['cat_id'], is_X=True, descrip='cat_id' )

    # Add the date block number as a feature
    if 'date_block_num' in features:
        X = get_regression_vectors_from_matrix( X, obj.ts['date_block_num'], is_X=True, descrip='date_block_num' )

    # Add the month as a feature
    if 'month' in features:
        X = get_regression_vectors_from_matrix( X, obj.ts['month'], is_X=True, descrip='month' )

    # Add the year as a feature
    if 'year' in features:
        X = get_regression_vectors_from_matrix( X, obj.ts['year'], is_X=True, descrip='year' )

    return X, Y

##################################################################################################################
# Define functions to help with the analysis

def create_pivot_ts( input_table, pivot_column, val_column, agg_rule, missing_method ):
    
        # Make time series out of the monthly  sales
        tmp_table = input_table.groupby( [ 'year_month', pivot_column ] ).agg( { val_column : agg_rule } ).reset_index()
        ts = tmp_table.pivot_table( val_column, index="year_month", columns=pivot_column )
        
        # Fill missing values with 0
        if missing_method == "zero":
            ts = ts.fillna(0)
        elif missing_method == 'ffill':
            ts = ts.fillna(method=missing_method)
        else:
            ValueError( 'Unsupported value: .' + missing_method )
        
        # Set negative values to 0
        ts[ ts < 0 ] = 0
        
        # Set the index to be the dates
        dates = year_month_to_datetime(ts.index.values )
        ts = ts.set_index( pd.Index( dates ) )
    
        # Make sure the dates are sorted
        ts.sort_index(axis=0, ascending=True, inplace=True )
        
        return(ts)
    

# Define a helper function
def year_month_to_datetime( ym ):

    if isinstance(ym, float ) or isinstance(ym, int):
        m = ym % 100
        y = ym // 100

        output = datetime.date( y, m, 1 )
    else:
        if isinstance( ym, pd.Series):
            ym = list(ym)

        output = []
        for j in range(len(ym)):
            m = ym[j] % 100
            y = ym[j] // 100

            output.append( datetime.date( y, m, 1 ) )

    return output    

def rmse( x1, x2 ):
    
    res = np.sqrt( np.mean( (x1.ravel()[:,np.newaxis] - x2.ravel()[:,np.newaxis] ) ** 2 ) )
    return(res)    


def decompose_shop_item_id( shop_item_id ):
    
    item_id = shop_item_id // 100
    shop_id = shop_item_id % 100

    return shop_id, item_id


def get_regression_vectors_from_matrix( X, input_mtx, is_X, descrip=None ):

    N = input_mtx.shape[1]
    new_row = np.array( [np.NaN] * N ).reshape(1,N)    
    if is_X:
        M = np.vstack( [ new_row, input_mtx.to_numpy() ] )
    else:
        M = np.vstack( [ input_mtx.to_numpy(), new_row ] )

    X[ TRAIN ].append( M[:-2,:] )
    X[ VALID ].append( M[-2,:] )
    X[ TEST  ].append( M[-1,:] )
    
    if is_X:
        X[ DESC ].append(descrip)

    return X


def remove_nan_rows( xx_input, yy_input ):

    xx_output = xx_input
    yy_output = yy_input
    for ds in [ TRAIN, VALID, TEST ]:

        idx_x = np.any( np.isnan( xx_input[ds] ), axis=1 ).ravel()
        idx_y = np.any( np.isnan( yy_input[ds] ), axis=1 ).ravel() 
        
        # For the TEST set, all y's are NaN, so we can't remove these rows
        if ds == TEST:
            idx = idx_x
        else:
            idx = idx_x | idx_y
            
        xx_output[ds] = xx_input[ds][~idx,:]
        yy_output[ds] = yy_input[ds][~idx,:]
        
    return xx_output, yy_output

def combine_features( X, Y ):

    xx = dict()
    yy = dict()
    xx[DESC] = X[DESC]
    for ds in [ TRAIN, VALID, TEST ]:

        # First unravel the ground truth matrix into a single vector
        yy[ds] = Y[ds][0].ravel()[:,np.newaxis]
        
        # Then unravel the feature matrices
        x_tmp = []
        for _, f in enumerate(X[ds], start=0):
            x_tmp.append(f.ravel()[:,np.newaxis])
            
        # Combine all of the column vectors into a single matrix
        xx[ds] = np.hstack(x_tmp)

        # Then downcast the values to float32 (from float64)
        #yy[ds] = yy[ds].astype('float32')
        #xx[ds] = xx[ds].astype('float32')        

    # Remove all rows with NaNs in the Train and Validation sets
    # In the test set, all Y values are unknown, so we only remove rows where X has NaNs
    xx, yy = remove_nan_rows( xx, yy )

    return xx, yy

def preprocess_features(xx_input):
    
    xx_pp = dict()
    for k in [ TRAIN, VALID, TEST, DESC ]:
        xx_pp[k] = xx_input[k].copy()

    for k, desc in enumerate(xx_input[DESC]):
        # Process the sales data
        if 0 == desc.find( 'sales_' ):
            for ds in [ TRAIN, VALID, TEST ]:
                # Force the sales values to be in the interval [0,20]
                xx_pp[ds][:,k] = np.maximum(  0, xx_input[ds][:,k] )
                xx_pp[ds][:,k] = np.minimum( 20, xx_input[ds][:,k] )

    return(xx_pp)
   
    
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:,:] - cumsum[:-N,:]) / float(N)


def format_forecast( test, fcst, fill_na=True ):

    # Get the forecasts for IDs that are in the test set
    fcst_df = fcst[ test['shop_item_id' ] ].to_frame().reset_index()
    
    # Rename the columns so that we can do a join on the test set and forecasts
    fcst_df.columns = [ 'shop_item_id', 'item_cnt_month']

    # Join the test set to the forecasts, and keep the relevant columns
    output = test.merge( fcst_df, on=["shop_item_id"], how='left', suffixes=("_left","") )
    output = output[["ID","item_cnt_month"]]
    
    # Fill NaN values, if specified by the input arguments
    if fill_na:
        output = output.fillna(0)

    return(output)


def convert_to_sparse( xx_input ):

    xx_output = dict()
    xx_output[DESC] = xx_input[DESC].copy()
    
    for ds in [ TRAIN, VALID, TEST ]:
        xx_output[ds] = scipy.sparse.csr.csr_matrix( xx_input[ds], dtype='float32' )

    return(xx_output)
    

def add_one_hot_encoding( xx_input, feature_name, binarizer ):

    xx_output = dict()
    for ds in [ TRAIN, VALID, TEST, DESC ]:
        xx_output[ds] = xx_input[ds].copy()
    
    # Get the column with the target feature
    idx = xx_input[DESC].index(feature_name)
    
    # Update the descriptions of the features
    new_cols = [ feature_name + "_{:02d}".format(x) for x in binarizer.classes_ ]    
    xx_output[DESC] = xx_input[DESC][:idx] + xx_input[DESC][idx+1:] + new_cols
    
    # (1) Remove the target column from the data sets
    # (2) One-hot encode it
    # (3) Append to the right side of the data sets
    for ds in [ TRAIN, VALID, TEST ]:
        target_data = xx_input[ds][:,idx]
        one_hot_train = binarizer.transform( target_data.todense() )
        tmp = scipy.sparse.hstack( [ xx_input[ds][:,:idx], xx_input[ds][:,idx+1:], one_hot_train ] )
        xx_output[ds] = scipy.sparse.csr.csr_matrix( tmp, dtype='float32' )
        
    return xx_output

def get_recent_data( df_input, date_block_cutoff=0 ):

    # Exclude date blocks before the cutoff
    if date_block_cutoff > 0:
        df_sub = df_input[ date_block_cutoff < df_input['date_block_num']]
    else:
        # No need to make copies if we are not removing rows
        df_sub = df_input.copy()

    return df_sub

def fit_model( model_constructor_fun, model_type, df_dict ):
    """A function that performs the validation testing and writes the formatted
    predictions to a csv file. 
    
    The forecasts will be based on the combined train and validation sets.
    
    The model_constructor_fun should take no arguments, and produce a model of the sklearn class.
    The model that is created is expected to have both 'fit' and 'predict' methods.
    
    The results of the tests will be written to the output_file."""

    if model_type == TRAIN:
        X = df_dict[TRAIN].drop([TARGET_COL], axis=1 )
        Y = df_dict[TRAIN][TARGET_COL]
    elif model_type == TEST:
        # Include the both the Train and Validation sets for constructing the Test model
        X = pd.concat( [ df_dict[TRAIN], df_dict[TRAIN]  ] )
        X = X.drop([TARGET_COL], axis=1 )
        Y = pd.concat( [ df_dict[TRAIN][TARGET_COL], df_dict[TRAIN][TARGET_COL]  ] )
    else:
        raise ValueError( 'Unsupported model type: ' + model_type )

    # Fit the model
    model = model_constructor_fun()    
    model.fit(X,Y)

    return model

def predict_model( model, data_set_type, df_dict, clip_forecasts=(0,20) ):
    
    # Get the appropriate X and Y values based on the data set type
    x_input = df_dict[data_set_type].drop([TARGET_COL], axis=1 )
    y_vals = df_dict[data_set_type][TARGET_COL]

    # Run the model on the validation set and see the score
    yhat = model.predict( x_input )

    # Clip the forecast to lie in the appropriate interval
    if clip_forecasts is not None:
        yhat = np.maximum( clip_forecasts[0], yhat )
        yhat = np.minimum( clip_forecasts[1], yhat )    

    res = rmse( yhat, y_vals )
    print( data_set_type + ' RMSE {}'.format( res ) )
  
    return(yhat)

def encode_means_with_smoothing( df, target_col, group_col, alpha ):
    # Mean Encoding
    global_mean = df[target_col].mean()
    my_fun = lambda x: ( x.sum() + alpha * global_mean ) / ( len(x) + alpha )
    encoded_feature = df.groupby(group_col)[target_col].transform(my_fun)
    return(encoded_feature)


def encode_means_with_cv( df, target_col, group_col, n_splits ):

    kf = KFold(n_splits=n_splits, shuffle=False)
    split_info = [ x for x in kf.split(df) ]

    mu = np.nanmean( df[target_col] )
    encoded_feature = pd.Series( np.nan * np.ones_like(df[target_col]), index = df[group_col] )

    for splt in split_info:
        # Get the test and train indices for the current fold
        idx_train, idx_test = splt

        # Get the test and train data
        train_data = df.iloc[idx_train,:]
        test_data = df.iloc[idx_test,:]

        # Put the means into the output vector
        encoded_feature.iloc[idx_test] = encode_means_from_test_train_split( train_data, test_data, target_col, group_col)

    # Fill missing values with the global mean
    encoded_feature = encoded_feature.fillna(mu)
    
    return encoded_feature


def encode_means_from_test_train_split( train_data, test_data, target_col, group_col):
    
    # Get item IDs common to both test and train, and also those just found in the test set
    common_ids = set(test_data[group_col]).intersection( set(train_data[group_col]) )
    missing_ids = set(test_data[group_col]).difference(common_ids)

    # Construct a dictionary mapping item IDs from the test set to their means in the train set
    train_means = train_data.groupby(group_col)[target_col].mean()    
    common_means = pd.Series( [ train_means[x] for x in list(common_ids) ], index=pd.Index(common_ids, dtype='int64' ) )
    missing_means = pd.Series( np.nan * np.ones_like(missing_ids), index=pd.Index(list(missing_ids), dtype='int32' ) )
    all_means = dict(pd.concat( [ common_means, missing_means ] ) )

    encoded_features_test = [ all_means[x] for x in test_data[group_col] ]    
    return encoded_features_test    