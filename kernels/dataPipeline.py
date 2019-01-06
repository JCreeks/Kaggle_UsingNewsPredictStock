
# coding: utf-8

# In[ ]:

import gc
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from kaggle.competitions import twosigmanews
import lightgbm as lgb
from joblib import Parallel, delayed
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, make_scorer

pd.set_option('max_columns', 50)
pd.options.mode.chained_assignment = None


# In[ ]:

env = twosigmanews.make_env()
market_train, news_train = env.get_training_data()


# In[ ]:

market_sample = pd.read_csv('../input/marketdata_sample.csv')


# ## 2.1 Adding quant features

# In[ ]:

class AddingQuantFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.feature_cols = ['close', 'open', 'volume']
        self.moving_average_number_of_days = [5, 10]
        self.exponential_moving_average_span = [5, 10]
        self.bollinger_band_number_of_days = [7]
        self.ewma = pd.Series.ewm
        self.no_of_std = 2
        self.returns_features = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                            'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
        self.drop_features = ['time', 'assetName', 'volume', 'close', 'open',
                               'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                               'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                               'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                               'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
        
    def _compute_moving_average(self, X):
        for column in self.feature_cols:
            for window_period in self.moving_average_number_of_days:
                feature_column_name = 'moving_average_{0}_{1}_days'.format(column, window_period)
                X[feature_column_name] = X.groupby('assetCode')[column].apply(lambda x: x.rolling(window_period).mean())
        return X
    
    def _compute_exponential_moving_average(self, X):
        for column in self.feature_cols:
            for span_period in self.exponential_moving_average_span:
                feature_column_name = 'exponential_moving_average_{0}_{1}_days'.format(column, span_period)
                X[feature_column_name] = X.groupby('assetCode')[column].apply(lambda x: self.ewma(x, span = span_period).mean())
        return X
    
    def _compute_macd(self, X):
        for column in self.feature_cols[1:3]:
            feature_column_name = 'macd_{0}'.format(column)
            X[feature_column_name] = (X.groupby('assetCode')[column].apply(lambda x: self.ewma(x, span = 12).mean()) - 
                                     X.groupby('assetCode')[column].apply(lambda x: self.ewma(x, span = 26).mean()))
        return X
    
    def _compute_bollinger_band(self, X):
        
        # We will not take volume into account here
        for column in self.feature_cols[1:3]:
            for window_period in self.bollinger_band_number_of_days:
                bb_high_feature_column_name = 'bollinger_band_{0}_{1}_days_high'.format(column, window_period)
                bb_low_feature_column_name = 'bollinger_band_{0}_{1}_days_low'.format(column, window_period)
                std_feature_column_name = 'moving_average_std_{0}_{1}_days'.format(column, window_period)
                moving_average_feature_column_name = 'moving_average_{0}_{1}_days'.format(column, window_period)
                
                X[std_feature_column_name] = X.groupby('assetCode')[column].apply(lambda x: x.rolling(window_period).std())
                X[bb_high_feature_column_name] = X[moving_average_feature_column_name] + self.no_of_std * X[std_feature_column_name]
                X[bb_low_feature_column_name] = X[moving_average_feature_column_name] - self.no_of_std * X[std_feature_column_name]
        return X
    
    def _compute_rsi(self, X):
        return
    
    def _compute_miscellaneous_features(self, X):
        for column in self.returns_features:
            feature_column_name = '{0}_average'.format(column)
            X[feature_column_name] = X.groupby('date')[column].apply(lambda x: x.mean())
        
#         scaler = StandardScaler(with_mean = True, with_std = True)
#         X[['close']] = scaler.fit_transform(X[['close']])
#         X[['open']] = scaler.fit_transform(X[['open']])
        X['close_open_ratio'] = X['close']/X['open']
        X['close_open_ratio_average'] = X.groupby('date')['close_open_ratio'].apply(lambda x: x.mean())
        return X
    
    def fit(self, X, y = None):
        print("Adding Quant Features")
        self.quant_features_functions = [self._compute_moving_average, 
                                         self._compute_exponential_moving_average, 
                                         self._compute_miscellaneous_features
                                        ]
        return self
    
    def transform(self, X, y = None):
        
        # We will work on data after 2010
        if len(X) == 1:
            X[0]['date'] = X[0]['time'].dt.date
            X[0] = X[0].loc[X[0]['date'] >= date(2010, 1, 1)]
        else:
            X[0]['date'] = X[0]['time'].dt.date
            X[1]['date'] = X[1]['time'].dt.date

        # Apply each quant function to X
        for quant_function in self.quant_features_functions:
            X[0] = quant_function(X[0])
            
        if len(X) == 1:
            X = X[0]
        else:
            X[0] = X[0].drop(self.drop_features, 1)
            X = pd.merge(X[1], X[0], how = 'left', on = ['date', 'assetCode'])
        
        X = X.drop(self.feature_cols + ['close_open_ratio'], 1)
        gc.collect()
        return X


# ## 2.2 Adding Lag Features

# In[ ]:

class AddingLagFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.shift_size = 1
        self.n_lag = [3, 7, 14]
        self.return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10']
        
    def fit(self, X, y = None):
        print("Adding Lag Features")
        return self
    
    def _create_lag(self, X):
        code = X['assetCode'].unique()

        for col in self.return_features:
            for window in self.n_lag:
                rolled = X[col].shift(self.shift_size).rolling(window = window)
                lag_mean = rolled.mean()
#                 lag_max = rolled.max()
#                 lag_min = rolled.min()
                lag_std = rolled.std()
                X['%s_lag_%s_mean'%(col, window)] = lag_mean
#                 X['%s_lag_%s_max'%(col, window)] = lag_max
#                 X['%s_lag_%s_min'%(col, window)] = lag_min
                X['%s_lag_%s_std'%(col, window)] = lag_std
        del code
        gc.collect()
        
        return X
    
    def transform(self, X, y = None):
        
        assetCodes = X['assetCode'].unique()
        
        all_df = []
        df_codes = X.groupby('assetCode')
        df_codes = [df_code[1] for df_code in df_codes]

        pool = Pool(4)
        all_df = pool.map(self._create_lag, df_codes)
        new_df = pd.concat(all_df)
        X = pd.merge(X, new_df, how = 'left', on = ['date','assetCode'])
        pool.close()
        
        del all_df, new_df, df_codes, assetCodes
        gc.collect()
        
        return X


# ## 2.3 Impute Missing Values

# In[ ]:

class MissingValuesImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        return
        
    def fit(self, X, y = None):
        print("Missing Values Imputer")
        return self
    
    def transform(self, X, y = None):
        
        for i in X.columns:
            if X[i].dtype == "object":
                X[i] = X[i].fillna("other")
            elif (X[i].dtype == "int64" or X[i].dtype == "float64"):
                X[i] = X[i].fillna(X[i].mean())
            else:
                pass
            
        return X


# ## 2.4 Preprocessing News

# In[ ]:

class PreprocessingNews(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.drop_list = [
        'audiences', 'subjects', 'assetName',
        'headline', 'firstCreated', 'sourceTimestamp',
        ]
        
    def fit(self, X, y = None):
        print("Preprocessing News")
        return self
    
    def transform(self, X, y = None):
        
        X[1].drop(self.drop_list, axis = 1, inplace = True)
            
        # Factorize categorical columns
        for col in ['headlineTag', 'provider', 'sourceId']:
            X[1][col], uniques = pd.factorize(X[1][col])
            del uniques
        
        # Remove {} and '' from assetCodes column
        X[1]['assetCodes'] = X[1]['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
        gc.collect()
        
        return X


# ## 2.5 Unstacking News

# In[ ]:

class UnstackingNews(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        return
        
    def fit(self, X, y = None):
        print("Unstacking News")
        return self
    
    def transform(self, X, y = None):
        
        codes = []
        indexes = []
        for i, values in X[1]['assetCodes'].iteritems():
            values_as_list = values.split(", ")
            codes.extend(values_as_list)
            repeat_index = [int(i)]*len(values_as_list)
            indexes.extend(repeat_index)
        index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
        del codes, indexes
        gc.collect()
        
        X[1]['news_index'] = X[1].index.copy()

        # Merge news on unstacked assets
        X[1] = index_df.merge(X[1], how = 'left', on = 'news_index')

        # Drop the unnecessary columns
        X[1].drop(['news_index', 'assetCodes'], axis = 1, inplace = True)
        del index_df
        gc.collect()
        
        return X


# ## 2.6 Generate News Aggregations

# In[ ]:

class GroupingNewsWithAggregations(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.aggregations = ['mean']
        
    def fit(self, X, y = None):
        print("Grouping News with Aggregations")
        return self
    
    def transform(self, X, y = None):
        
        # Creating a data column
        X[1]['date'] = X[1].time.dt.date

        X[1] = X[1].groupby(['assetCode', 'date']).agg(self.aggregations)
        X[1].columns = pd.Index(["{}_{}".format(e[0], e[1]) for e in X[1].columns.tolist()])
        X[1].reset_index(inplace = True)

        # Set datatype to float32
        float_cols = {c: 'float32' for c in X[1].columns if c not in ['assetCode', 'date']}
        X[1] = X[1].astype(float_cols) 
        del float_cols
        gc.collect()
        
        X = X[0].merge(X[1], how = 'left', on = ['assetCode', 'date'])
        
        return X


# <h2>3. Model Building </h2>

# In[ ]:

class LGBClassifierCV(BaseEstimator, RegressorMixin):
    
    def __init__(self, 
                 axis = 0, 
                 lgb_params = None, 
                 fit_params = None, 
                 cv = 5, 
                 perform_bayes_search = False, 
                 perform_random_search = False, 
                 use_train_test_split = False,
                 use_full_data = True,
                 use_kfold_split = True):
        self.axis = axis
        self.lgb_params = lgb_params
        self.fit_params = fit_params
        self.fit_params = fit_params
        self.cv = cv
        self.perform_random_search = perform_random_search
        self.perform_bayes_search = perform_bayes_search
        self.use_train_test_split = use_train_test_split
        self.use_kfold_split = use_kfold_split
        self.use_full_data = use_full_data
    
    @property
    def feature_importances_(self):
        feature_importances = []
        for estimator in self.estimators_:
            feature_importances.append(
                estimator.feature_importances_
            )
        return np.mean(feature_importances, axis = 0)
    
    @property
    def evals_result_(self):
        evals_result = []
        for estimator in self.estimators_:
            evals_result.append(
                estimator.evals_result_
            )
        return np.array(evals_result)
    
    @property
    def best_scores_(self):
        best_scores = []
        for estimator in self.estimators_:
            best_scores.append(
                estimator.best_score_['validation']['macroF1']
            )
        return np.array(best_scores)
    
    @property
    def cv_scores_(self):
        return self.best_scores_ 
    
    @property
    def cv_score_(self):
        return np.mean(self.best_scores_)
    
    @property
    def best_iterations_(self):
        best_iterations = []
        for estimator in self.estimators_:
            best_iterations.append(
                estimator.best_iteration_
            )
        return np.array(best_iterations)
    
    @property
    def best_iteration_(self):
        return np.round(np.mean(self.best_iterations_))
    
    def _get_bayes_search_params(self, X, y):
        init_round = 10 
        opt_roun = 10
        n_folds = 6
        random_seed = 42
        n_estimators = 500
        learning_rate = 0.02
        colsample_bytree = 0.93

        # prepare data
        train_data = lgb.Dataset(data = X, label = y)

        # parameters
        def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight, colsample_bytree, min_child_samples, subsample):
            params = {'application': 'multiclass',
                      'num_iterations': n_estimators, 
                      'learning_rate': learning_rate, 
                      'early_stopping_round': 300, 
                      'metric': 'macroF1'}

            params["num_leaves"] = int(round(num_leaves))
            params["num_class"] = 5
            params['feature_fraction'] = max(min(feature_fraction, 1), 0)
            params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
            params['max_depth'] = int(round(max_depth))
            params['lambda_l1'] = max(lambda_l1, 0)
            params['lambda_l2'] = max(lambda_l2, 0)
            params['min_split_gain'] = min_split_gain
            params['min_child_weight'] = min_child_weight
            params['colsample_bytree'] = 0.93
            params['min_child_samples'] = 56,
            params['subsample'] = 0.84
            cv_result = lgb.cv(params, train_data, nfold = n_folds, seed = random_seed, stratified = True, verbose_eval = 200, metrics = ['auc'])
            return max(cv_result['auc-mean'])

        # range 
        lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (19, 45),
                                                'feature_fraction': (0.1, 0.9),
                                                'bagging_fraction': (0.8, 1),
                                                'max_depth': (5, 8.99),
                                                'lambda_l1': (0, 5),
                                                'lambda_l2': (0, 3),
                                                'min_split_gain': (0.001, 0.1),
                                                'min_child_weight': (5, 50),
                                                'colsample_bytree' : (0.7,1.0),
                                                'min_child_samples' : (40,65),
                                                'subsample' : (0.7,1.0)
                                               }, random_state = 0)
        # optimize
        lgbBO.maximize(init_points = init_round, n_iter = opt_roun)
        optimal_params = lgbBO.res['max']['max_params']
        print(optimal_params)
        return optimal_params
    
    def _get_random_search_params(self, X, y):
        # Define a search space for the parameters
        lgb_search_params = {
            'learning_rate': [0.15, 0.1, 0.05, 0.02, 0.01],
            'num_leaves': [i for i in range(12, 90, 6)],
            'n_estimators': [100, 200, 300, 400, 500, 600, 800],
            'min_child_samples': [i for i in range(10, 100, 10)],
            'colsample_bytree': [0.8, 0.9, 0.95, 1],
            'subsample': [0.8, 0.9, 0.95, 1],
            'reg_alpha': [0.1, 0.2, 0.4, 0.6, 0.8],
            'reg_lambda': [0.1, 0.2, 0.4, 0.6, 0.8],
        }

        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, shuffle = False)
        best_eval_score = 0
        for i in range(100):
            print("random search iteration - " + str(i+1))
            params = {k: np.random.choice(v) for k, v in lgb_search_params.items()}
            
            # Use 2 cores/threads
            params['n_jobs'] = cpu_count() - 1
            
            score = self._evaluate_model(x_train, x_val, y_train, y_val, params)
            if score < best_eval_score or best_eval_score == 0:
                best_eval_score = score
                best_params = params
        print("Best evaluation logloss", best_eval_score)
        print(best_params)
        
        return best_params
            
    def _custom_metric(self, dates, pred_proba, num_target, universe):
        y = 2*pred_proba[:, 1] - 1

        # get rid of outliers
        r = num_target.clip(-1,1)
        x = y * r * universe
        result = pd.DataFrame({'day' : dates, 'x' : x})
        x_t = result.groupby('day').sum().values
        return np.mean(x_t) / np.std(x_t)
    
    def _evaluate_model(self, x_train, x_val, y_train, y_val, params):
        model = LGBMClassifier(**params)
        model.fit(x_train, y_train)
        return log_loss(y_val, model.predict_proba(x_val))
    
    def find_best_params_(self, X, y):
        if self.perform_random_search:
            self.lgb_optimal_params = self._get_random_search_params(X, y)
        
        if self.perform_bayes_search:
            self.lgb_optimal_params = self._get_bayes_search_params(X, y)
    
    def fit(self, X, y = None, **fit_params):
        print("LGBClassifierCV")
        
        # Segregate the target columns
        dates = X['date']
        num_target = X['returnsOpenNextMktres10'].astype('float32')
        y = (X['returnsOpenNextMktres10'] >= 0).astype('int8')
        universe = X['universe'].astype('int8')

        # Drop columns that are not features
        X.drop(['returnsOpenNextMktres10', 'date', 'universe', 'assetCode', 'assetName', 'time'], axis = 1, inplace = True)
        print(X.columns)
        
        # Set all remaining columns to float32 datatype
        X = X.astype('float32')  
        gc.collect()
        
         # Find the best params using random search or random search
        if self.perform_bayes_search or self.perform_random_search:
            self.find_best_params_(X, y)
        else:
            self.lgb_optimal_params = {'learning_rate': 0.01, 
                                       'num_leaves': 66, 
                                       'n_estimators': 200, 
                                       'min_child_samples': 40, 
                                       'colsample_bytree': 0.9, 
                                       'subsample': 0.9, 
                                       'reg_alpha': 0.4, 
                                       'reg_lambda': 0.2, 
                                       'n_jobs': 3}
#             self.lgb_optimal_params = {'learning_rate': 0.05, 
#                                        'num_leaves': 12, 
#                                        'n_estimators': 100, 
#                                        'min_child_samples': 70, 
#                                        'colsample_bytree': 0.9, 
#                                        'subsample': 0.9, 
#                                        'reg_alpha': 0.8, 
#                                        'reg_lambda': 0.2, 
#                                        'n_jobs': 3}
            
#             self.lgb_optimal_params = {'learning_rate': 0.01, 
#                                        'num_leaves': 168, 
#                                        'n_estimators': 200, 
#                                        'min_child_samples': 34, 
#                                        'colsample_bytree': 0.8, 
#                                        'min_child_weight': 1.0, 
#                                        'subsample': 0.95, 
#                                        'reg_alpha': 0.4, 
#                                        'reg_lambda': 0.1, 
#                                        'n_jobs': 3}
            
        self.lgb_optimal_params["objective"] = "binary"
        self.lgb_optimal_params["metric"] = "binary_logloss"
        self.lgb_optimal_params["boosting_type"] = "gbdt"
        
        # Train on full dataset
        if self.use_full_data:
            train_data = lgb.Dataset(X, label = y)
            self.lgb_model = lgb.train(self.lgb_optimal_params,
                                       train_set = train_data)
        
        # Use a simple train-test split
        if self.use_train_test_split:
            x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, shuffle = False)
            
            train_data = lgb.Dataset(x_train, label = y_train)
            val_data = lgb.Dataset(x_val, label = y_val)
            
            self.lgb_model = lgb.train(self.lgb_optimal_params,
                                       train_set = train_data, 
                                       num_boost_round = 100,
                                       valid_sets = [val_data], 
                                       early_stopping_rounds = 5)
            
        # When not using random search to tune parameters, proceed with a simple Stratified Kfold CV
        if self.use_kfold_split:
            kf = StratifiedKFold(n_splits = self.cv, shuffle = False)
            for fold_index, (train_data, val_data) in enumerate(kf.split(X, y)):
                print("Train Fold Index - " + str(fold_index))

                lgb_model = lgb.LGBMClassifier(**self.lgb_optimal_params)
                lgb_model = lgb.train(self.lgb_optimal_params,
                                       train_set = train_data, 
                                       num_boost_round = 100,
                                       valid_sets = [val_data], 
                                       early_stopping_rounds = 5)
                self.estimators_.append(lgb_model)
        return self
    
    def predict(self, X):
        # Drop columns that are not features
        X.drop(['date', 'assetCode', 'assetName', 'time'], axis = 1, inplace = True)
        predictions = self.lgb_model.predict(X) * 2 - 1
        
        return predictions


# <h2>4. Create a Pipeline</h2>

# In[ ]:

data_pipeline = Pipeline([
    ('quant_features_generator', AddingQuantFeatures()),
#     ('news_preprocessor', PreprocessingNews()),
#     ('news_unstacker', UnstackingNews()),
#     ('news_grouping', GroupingNewsWithAggregations()),
    ('lgb', LGBClassifierCV(cv = 5,
                            perform_random_search = False,
                            perform_bayes_search = False,
                            use_train_test_split = False,
                            use_full_data = True,
                            use_kfold_split = False)
    )
])

data_pipeline.fit([market_train])


# In[ ]:

def write_submission(pipeline, env):
    days = env.get_prediction_days()
    n_days = 0
    total_market_obs = []
    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        n_days += 1
        print(n_days,end = ' ')
        
        total_market_obs.append(market_obs_df.copy())
        if len(total_market_obs) == 1:
            history_df = total_market_obs[0]
        else:
            history_df = pd.concat(total_market_obs[-(10+1):], ignore_index = True)
        
        preds = data_pipeline.predict([history_df, market_obs_df])
        preds = (preds - np.mean(preds))
        sub = pd.DataFrame({'assetCode': market_obs_df['assetCode'], 'confidence': preds})
        predictions_template_df = predictions_template_df.merge(sub, how = 'left').drop(
            'confidenceValue', axis = 1).fillna(0).rename(columns = {'confidence': 'confidenceValue'})
        
        env.predict(predictions_template_df)
        del predictions_template_df, preds, sub
        gc.collect()
    env.write_submission_file()
    
write_submission(data_pipeline, env)


# In[ ]:



