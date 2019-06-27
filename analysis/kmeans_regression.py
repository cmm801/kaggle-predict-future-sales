
TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

class KMeansRegression():
    
    def __init__(self, kgObj, n_clusters, eval_lags, date_block_cutoff=0, \
                     preprocess=True, forecast_method='pca', forecast_params={'dof' : 2 } ):
        
        self.n_clusters = n_clusters
        self.eval_lags = eval_lags
        self.kgObj = kgObj

        self.date_block_cutoff = date_block_cutoff
        
        self.lower_forecast_limit =  0
        self.upper_forecast_limit = 20
    
        self.preprocess = preprocess
        
        self.ts = kgObj.ts['sales']
        
        self.forecast_method = forecast_method
        self.forecast_params = forecast_params

        self.xx = { TRAIN : [], VALID : [], TEST : [] }
        self.yy = { TRAIN : [], VALID : [], TEST : [] }          
        self.xx_norm = { TRAIN : [], VALID : [], TEST : [] }
        self.yy_norm = { TRAIN : [], VALID : [], TEST : [] }          
        self.yhat = { TRAIN : [], VALID : [], TEST : [] }        
        self.yhat_norm = { TRAIN : [], VALID : [], TEST : [] }   
        
        self.mu = { TRAIN : [], VALID : [], TEST : [] }
        self.sigma = { TRAIN : [], VALID : [], TEST : [] }        
     
        self.labels = { TRAIN : [], VALID : [], TEST : [] }                
        
        self.model = { TRAIN : [], TEST : [] }            
        
        # Set up the train / validation / test data sets
        self._setup_train_valid_test_data()


    def forecast_from_labels( self, xx_norm_input, model_type ):
        
        if self.forecast_method == 'pca':
            y_fcst_norm = self.forecast_from_labels_with_pca( xx_norm_input, model_type )
        elif self.forecast_method == 'prophet':
            y_fcst_norm = self.forecast_from_labels_with_prophet( xx_norm_input, model_type )
            
        return(y_fcst_norm)
    
        
    def forecast_from_labels_with_pca( self, xx_norm_input, model_type ):
        
        y_fcst_norm = np.nan * np.zeros( shape=(xx_norm_input.shape[0],1), dtype=float )

        labels = self.model[model_type].predict( xx_norm_input )
        
        # Loop through the different sets of labels
        for L in range(kmr.n_clusters):

            # Get the subset of input features corresponding to the target label
            xx_sub_input = xx_norm_input[ labels == L, : ]

            # Get the subset of data for the specific label in the model's data
            xx_sub_model = kmr.xx_norm[model_type][ kmr.labels[model_type] == L, : ]
            yy_sub_model = kmr.yy_norm[model_type][ kmr.labels[model_type] == L, : ]
           
            dof = self.forecast_params.dof
            if dof == 0:
                y_fcst_norm[labels == L] = np.mean(yy_sub_model)
            else:
                pc = PCA( n_components=dof )
                pc.fit(xx_sub_model)

                xx_model_pca = pc.transform(xx_sub_model)

                lm = LinearRegression()
                lm.fit( xx_model_pca, yy_sub_model )
                
                xx_input_pca = pc.transform(xx_sub_input)                
                y_fcst_norm[labels == L] = lm.predict( xx_input_pca )
                
        return y_fcst_norm
    
        
    def forecast_from_labels_with_prophet( self, xx_norm_input, model_type ):
        # prophet requires a pandas df at the below config 
        # ( date column named as DS and the value column as Y)

        # Check that all periods are consecutive, in order for the following methodology to work
        period_diffs = self.eval_lags[1:] - self.eval_lags[:-1]
        assert( np.all(period_diffs == 1 ) )
        
        y_fcst_norm = np.nan * np.zeros( shape=(xx_norm_input.shape[0],1), dtype=float )

        labels = self.model[model_type].predict( xx_norm_input )

        # Create some fake months for use in constructing a time series
        dates = [ datetime.date( 2000 + x // 12, 1 + (x % 12), 1 ) for x in range(len(self.eval_lags)) ]

        # Loop through the different sets of labels
        for L in range(kmr.n_clusters):

            # Get the average time series values
            xx_sub_input = xx_norm_input[ labels == L, : ]            
            ts_values = xx_sub_input.mean(axis=0 )
            prophet_ts = pd.DataFrame( { 'ds' : dates, 'y' : ts_values } )

            model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
            model.fit(prophet_ts) #fit the model with your dataframe

            # predict for five months in the furure and MS - month start is the frequency
            future = model.make_future_dataframe(periods=1, freq = 'MS')  

            # now lets make the forecasts
            fcst = model.predict(future)
            y_fcst_norm[labels == L ] = fcst.yhat.iloc[-1]

        return y_fcst_norm
    
    
    def format_data( self, data_type ):

        # Find the location of the date_block_num column
        idx_db_col = [ x == 'date_block_num' for x in self.xx_full[DESC] ].index(True)
        
        # Get the existing data sets
        date_nums = self.xx_full[data_type][:,idx_db_col]
        xx_input = np.hstack( [ self.xx_full[data_type][:,:idx_db_col], \
                                self.xx_full[data_type][:,idx_db_col+1:] ] )
        yy_input = self.yy_full[data_type].copy()
        
        # Exclude date blocks before the cutoff
        if date_block_cutoff > 0:
            # Find the rows that are greater than thte lower date cutoff
            idx = [ r > self.date_block_cutoff for r in date_nums ]
            self.xx[data_type], self.yy[data_type] = xx_input[idx,:], yy_input[idx,:]
        else:
            # No need to make copies if we are not removing rows
            self.xx[data_type], self.yy[data_type] = xx_input, yy_input    

    
    def normalize_data( self, data_type ):
       
        # Find the means and standard deviations
        self.mu[data_type] = np.nanmean( self.xx[data_type], axis=1, keepdims=True )
        self.sigma[data_type] = np.nanstd( self.xx[data_type], axis=1, keepdims=True )
        self.sigma[data_type][ self.sigma[data_type] == 0 ] = 1
        
        # Normalize the features and label vectors
        self.xx_norm[data_type] = self.normalize( self.xx[data_type], self.mu[data_type], self.sigma[data_type] )
        self.yy_norm[data_type] = self.normalize( self.yy[data_type], self.mu[data_type], self.sigma[data_type] )    
    
     
    def fit_train( self ):

        # Format the training data
        self.format_data( TRAIN )
        self.format_data( VALID )
        
        # Normalize the features and label vectors
        self.normalize_data( TRAIN )
        self.normalize_data( VALID )
        
        # Initialize and fit the model
        self.model[TRAIN] = KMeans( n_clusters=self.n_clusters )
        self.model[TRAIN].fit( self.xx_norm[TRAIN] )        
        
        # Get the fitted labels for the training set
        self.labels[TRAIN] = self.model[TRAIN].predict( self.xx_norm[TRAIN] )

        
    def predict_train(self, clip_forecasts=True):
                 
        # Calculate the mean within the different clusters            
        self.yhat_norm[TRAIN] = self.forecast_from_labels( self.xx_norm[TRAIN], model_type=TRAIN )
        
        # Save the forecasts into a new variable
        self.yhat[TRAIN] = self.unnormalize( self.yhat_norm[TRAIN], self.mu[TRAIN], self.sigma[TRAIN] )
        
        # Restrict the forecasts to lie in a target interval
        if clip_forecasts:
            self.yhat[TRAIN] = np.maximum( self.yhat[TRAIN], self.lower_forecast_limit )
            self.yhat[TRAIN] = np.minimum( self.yhat[TRAIN], self.upper_forecast_limit )

            
    def predict_validation(self, clip_forecasts=True ):
        
        # Calculate the mean within the different clusters            
        self.yhat_norm[VALID] = self.forecast_from_labels( self.xx_norm[VALID], model_type=TRAIN )
        
        # Save the forecasts into a new variable
        self.yhat[VALID] = self.unnormalize( self.yhat_norm[VALID], self.mu[VALID], self.sigma[VALID] )
        
        # Restrict the forecasts to lie in a target interval
        if clip_forecasts:
            self.yhat[VALID] = np.maximum( self.yhat[VALID], self.lower_forecast_limit )
            self.yhat[VALID] = np.minimum( self.yhat[VALID], self.upper_forecast_limit )

    
    def fit_test( self ):

        # Get the matrix of X-values and Y-values to use for testing
        self.X_test = self.ts.iloc[self.test_rows,:].to_numpy()
        
        # De-mean the training data
        self.mu_test = np.nanmean( self.X_test, axis=0 )
        self.sigma_test = np.nanstd( self.X_test, axis=0 )   
        self.sigma_test[ self.sigma_test == 0 ] = 1        
        self.X_test_norm = self.normalize( self.X_test, self.mu_test, self.sigma_test )
        
        
    def predict_test( self, clip_forecasts=True ):
                
        self.test_labels = self.model[TRAIN].predict( self.X_test_norm.T )
        self.test_labels_mean = [ np.mean( self.Y_valid_norm[ self.test_labels == p ]) for p in range(self.n_clusters) ]
        self.yhat_test_norm = [ self.test_labels_mean[x] for x in self.test_labels ]
        
        self.yhat_test = self.unnormalize( self.yhat_test_norm, self.mu_test, self.sigma_test )
        
        if clip_forecasts:
            self.yhat_test = np.maximum( self.yhat_test, self.lower_forecast_limit )
            self.yhat_test = np.minimum( self.yhat_test, self.upper_forecast_limit )

        return self.yhat_test

    
    def _get_feature_vectors( self, lags ):

        # Initialize a dictionary to store the feature vectors
        X = { TRAIN : [], VALID : [], TEST : [], DESC : [] }
        Y = { TRAIN : [], VALID : [], TEST : [] }

        # Create the test and train sets from the observation matrices
        Y = get_regression_vectors_from_matrix( Y, self.kgObj.ts['sales'], is_X=False )

        # Add the lagged sales as features
        X = get_lagged_features( self.kgObj, X, lags=lags )

        # Add the date block number as a feature
        X = get_regression_vectors_from_matrix( X, self.kgObj.ts['date_num_block'], is_X=True, descrip='date_block_num' )

        return X, Y


    def _preprocess_features(self, xx_input):

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

    def _setup_train_valid_test_data(self):

        self.X, self.Y = self._get_feature_vectors(lags=self.eval_lags)

        # Get the combined features in matrix form for the train, validation and test sets
        xx_raw, self.yy_full = combine_features( self.X, self.Y )

        if self.preprocess:
            # Preprocess the features (e.g. winsorize, clip values, etc.)
            self.xx_full = self._preprocess_features(xx_raw)
        else:
            self.xx_full = xx_raw

        
    def unnormalize( self, y_norm, mu, sigma ):

        y = y_norm * sigma + mu
        return(y)

    def normalize( self, y, mu, sigma ):

        y_norm = ( y - mu ) / sigma
        return(y_norm)    
    