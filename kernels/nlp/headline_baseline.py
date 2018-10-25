
# coding: utf-8

# # Amateur Hour - Using Headlines to Predict Stocks
# ### Starter Kernel by ``Magichanics`` 
# *([Gitlab](https://gitlab.com/Magichanics) - [Kaggle](https://www.kaggle.com/magichanics))*
# 
# Stocks are unpredictable, but can sometimes follow a trend. In this notebook, we will be discovering the correlation between the stocks and the news. After a few tries, my best score not including this one is ``0.54194`` from [V29](https://www.kaggle.com/magichanics/amateur-hour-using-headlines-to-predict-stocks/code?scriptVersionId=6466412). Right now, I'm trying to find answers as to why my score keeps fluctuating from ``-0.26`` to ``0.55``. 
# 
# We also have to keep in mind that these results may dramatically change when it comes to testing the kernel on the private dataset. 
# 
# If there are any things that you would like me to add or remove, feel free to comment down below. I'm mainly doing this to learn and experiment with the data. I plan on rewriting a lot of code in the future to make it look nicer, since a lot of the stuff I have written may not be the most efficient way to approach specific problems.
# 
# **To Do List:**
# * Removing features with low importance?
# * Add more data
#     * i.e. grouping data possibly based on a moving time window.
# * [Shifting time](https://www.kaggle.com/c/two-sigma-financial-news/discussion/69235)
# * Use Neural Network due to the large data.
# 
# **Table of Contents (WIP): **
# 1. Feature Engineering
#     * Market Dataframe Features
#     * Merging Dataframe
#     * Headline Coefficients
#     * Clustering Columns
# 2. Modelling using LGBM
# 
# 

# ![title](https://upload.wikimedia.org/wikipedia/commons/8/8d/Wall_Street_sign_banner.jpg)
# 
# Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Wall_Street_sign_banner.jpg)

# In[ ]:

# main
import numpy as np
import pandas as pd
import os
from itertools import chain
import gc

# text processing
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# clustering
from sklearn.cluster import KMeans

# time
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import LabelEncoder
import datetime

# training
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# import environment for data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:

sampling = False


# In[ ]:

(market_train_df, news_train_df) = env.get_training_data()

# its best to get the data by its tail
if sampling:
    market_train_df = market_train_df.tail(40_000) # 40k to 100k
    news_train_df = news_train_df.tail(100_000)
else:
    market_train_df = market_train_df.tail(3_000_000)
    news_train_df = news_train_df.tail(6_000_000)


# In[ ]:

market_train_df.head()


# In[ ]:

news_train_df.head()


# ### Information on the Training Data
# * There are no Unknown ``assetName`` in ``news_train_df``, but there are 24 479 rows with Unknown as the ``assetName`` in ``market_train_df``. Merging by ``assetCode`` leaves out Unknown rows, which could be problematic.
# * ``Volume`` has the highest correlation in terms of ``returnsOpenNextMktres10``.
# * Merging by just ``assetCodes`` greatly increases the dataframe (with just 100k rows, it has turned into 10 million rows), although merging by ``assetCodes`` and ``time`` greatly decrease the original dataframe.

# ### Market Groupby Features
# We are going to group market data based on ``assetName`` and determine the median and mean of the volume.

# In[ ]:

def mean_volume(market_df):
    
    # groupby and return median
    vol_by_name = market_df[['volume', 'assetName']].groupby('assetName').median()['volume']
    #vol_by_name_mean = market_df[['volume', 'assetName']].groupby('assetName').mean()['volume'] # could try mean?
    market_df['vol_by_name'] = market_df['assetName'].map(vol_by_name)
    
    # get difference
    market_df['vol_by_name_diff'] = market_df['volume'] - market_df['vol_by_name']
    
    return market_df


# In[ ]:

market_train_df = mean_volume(market_train_df)


# ### Aggregations on News Data
# 
# It helped a lot during the Home Credit competition, and in the next block of code we will be merging the news dataframe with the market dataframe. Instead of having columns with a list of numbers, we will get aggregations for each grouping. The following block creates a dictionary that will be used when merging the data.

# In[ ]:

news_agg_cols = [f for f in news_train_df.columns if 'novelty' in f or
                'volume' in f or
                'sentiment' in f or
                'bodySize' in f or
                'Count' in f or
                'marketCommentary' in f or
                'relevance' in f]
news_agg_dict = {}
for col in news_agg_cols:
    news_agg_dict[col] = ['mean', 'sum', 'max', 'min']
news_agg_dict['urgency'] = ['min', 'count']
news_agg_dict['takeSequence'] = ['max']


# ### Joining Market & News Data
# 
# The grouping method that I'll be using is from [bguberfain](https://www.kaggle.com/bguberfain), but I'll also be adding in other columns like ``headline``, as well eliminating rows that are not partnered with either the market or news data. One way I would improve this is probably group by time periods rather than exact times given in ``time`` due to the small amount of data that share the same amount of data in terms of the ``time`` column, and possibly making it a bit more efficient. 
# 
# Notes: 
# * When you run the full dataset, expect it to take a while.
# * As you remove more time features from seconds to year, the resulting train data becomes larger and larger.

# In[ ]:

def generalize_time(X):
    # convert time to string and/or get rid of Hours, Minutes, and seconds
    X['time'] = X['time'].dt.strftime('%Y-%m-%d %H:%M:%S').str.slice(0,16) #(0,10) for Y-m-d, (0,13) for Y-m-d H
    # do not use (0,10) or (0, 13) on the whole dataset
    
# get dataframes within indecies
def get_indecies(df, indecies):
    
    # update market dataframe to only contain the specific rows with matching indecies.
    def check_index(index, indecies):
        if index in indecies:
            indecies.remove(index)
            return True
        else:
            return False
    
    df['del_index'] = df.index.values
    df['is_in_indecies'] = df['del_index'].apply(lambda x: check_index(x, indecies))
    df = df[df.is_in_indecies == True]
    del df['del_index'], df['is_in_indecies']
    
    return df

def add_null_indecies(indecies, valid_indecies, num_nulls):
    
    curr_nulls = 0
    iteration = 0
    null_indecies = []
    
    # print error if its empty
    if len(indecies) == 0 or len(valid_indecies) == 0:
        print('No correlation. Try sampling more data!')
    if num_nulls == 0:
        return
    
    # loop to get any nulls that are not present in the index
    while curr_nulls < num_nulls or iteration >= len(indecies):
        if indecies[iteration] not in valid_indecies: # filling in missing values i.e. [3, (4), (5), 6, 7...]
            null_indecies.append(indecies[iteration]) # you could try fitting it between the adjacent values?
            curr_nulls += 1
        iteration += 1
        
    return null_indecies

# this function checks for potential nulls after grouping by only grouping the time and assetcode dataframe
# returns valid news indecies for the next if statement.
def partial_groupby(market_df, news_df, df_assetCodes, num_nulls):
    
    # get new dataframe
    temp_news_df_expanded = pd.merge(df_assetCodes, news_df[['time', 'assetCodes']], left_on='level_0', right_index=True, suffixes=(['','_old']))

    # groupby dataframes
    temp_news_df = temp_news_df_expanded.copy()[['time', 'assetCode']]
    temp_market_df = market_df.copy()[['time', 'assetCode']]

    # get indecies on both dataframes
    temp_news_df['news_index'] = temp_news_df.index.values
    temp_market_df['market_index'] = temp_market_df.index.values
    market_indecies = temp_market_df.index.values.tolist()

    # set multiindex and join the two
    temp_news_df.set_index(['time', 'assetCode'], inplace=True)

    # join the two
    temp_market_df_2 = temp_market_df.join(temp_news_df, on=['time', 'assetCode'])
    del temp_market_df, temp_news_df

    # drop nulls in any columns
    temp_market_df_2 = temp_market_df_2.dropna()

    # get indecies
    market_valid_indecies = temp_market_df_2['market_index'].tolist()
    news_valid_indecies = temp_market_df_2['news_index'].tolist()
    del temp_market_df_2
    
    # get null indecies if stated to do so
    if num_nulls > 0:
        market_null_indecies = add_null_indecies(market_indecies, market_valid_indecies, num_nulls)

    # get index rows
    market_df_valid = get_indecies(market_df, market_valid_indecies)
    if num_nulls > 0:
        market_df_nulls = get_indecies(market_df, market_null_indecies)
    else:
        market_df_nulls = 'null'
    
    return market_df_valid, market_df_nulls, news_valid_indecies

def join_market_news(market_df, news_df, nulls=False, num_nulls=0):
    
    # convert time to string
    generalize_time(market_df)
    generalize_time(news_df)
    
    # Fix asset codes (str -> list)
    news_df['assetCodes'] = news_df['assetCodes'].str.findall(f"'([\w\./]+)'")

    # Expand assetCodes
    assetCodes_expanded = list(chain(*news_df['assetCodes']))
    assetCodes_index = news_df.index.repeat( news_df['assetCodes'].apply(len) )
    
    assert len(assetCodes_index) == len(assetCodes_expanded)
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})
    
    if not nulls:
        market_df, market_df_nulls, news_valid_indecies = partial_groupby(market_df, news_df, df_assetCodes, num_nulls)
    
    # create dataframe based on groupby
    news_col = ['time', 'assetCodes', 'headline', 'audiences', 'subjects'] + sorted(list(news_agg_dict.keys()))
    news_df_expanded = pd.merge(df_assetCodes, news_df[news_col], left_on='level_0', right_index=True, suffixes=(['','_old']))
    
    # check if the columns are in the index
    if not nulls:
        news_df_expanded = get_indecies(news_df_expanded, news_valid_indecies)

    def news_df_feats(x):
        if x.name == 'headline':
            return list(x)
        elif x.name == 'subjects' or x.name == 'audiences':
            output = []
            for i in x:
                # remove all special characters
                codes = i.strip('{\',}').replace('\'','').split(', ')
                for j in codes:
                    output.append(j)
            return output
    
    # groupby time and assetcode
    news_df_expanded = news_df_expanded.reset_index()
    news_groupby = news_df_expanded.groupby(['time', 'assetCode'])
    
    # get aggregated df
    news_df_aggregated = news_groupby.agg(news_agg_dict).apply(np.float32).reset_index()
    news_df_aggregated.columns = ['_'.join(col).strip() for col in news_df_aggregated.columns.values]
    
    # get any important string dataframes
    groupby_news = news_groupby.transform(lambda x: news_df_feats(x))
    news_df_cat = pd.DataFrame({'headline':groupby_news['headline'],
                               'subjects':groupby_news['subjects'],
                               'audiences':groupby_news['audiences']})
    new_news_df = pd.concat([news_df_aggregated, news_df_cat], axis=1)
    
    # cleanup
    del news_df_aggregated
    del news_df_cat
    del news_df
    
    # rename columns
    new_news_df.rename(columns={'time_': 'time', 'assetCode_': 'assetCode'}, inplace=True)
    new_news_df.set_index(['time', 'assetCode'], inplace=True)
    
    # Join with train
    market_df = market_df.join(new_news_df, on=['time', 'assetCode'])
    if num_nulls > 0:
        market_df = pd.concat([market_df, market_df_nulls], sort=False)
    
    # replace with null string
    market_df[['audiences', 'subjects', 'headline']] = market_df[['audiences', 'subjects', 'headline']].fillna('null')

    return market_df


# if there is a joining error, it means that the dataframes have no correlation with each other (solution: increase train dataset)


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'X_train = join_market_news(market_train_df, news_train_df, nulls=False, num_nulls=10_000)')


# In[ ]:

X_train.head()


# In[ ]:

X_train.shape


# ### Text Processing with Logistic Regression
# 
# We are going to vectorize the headlines and apply logistic regression (labels being binary as to whether the stocks go up or not). In a nutshell, it splits the headlines into individual words, filters out unecessary words to prevent abnormal results, vectorizes it for modelling, and then with the target column provided, we could create a dataframe of coefficients that we could use as a feature in the dataframe! Right now I am just getting the mean of the coefficients in each list of headlines. 
# 
# Note: May be useful to apply it to ``universe``, and possibly get the sum or standard deviation of the word coefficients?

# In[ ]:

# reuse data
def round_scores(x):
    if x >= 0:
        return 1
    else:
        return 0
    
def clean_headlines(headline):
    
    # remove numerical and convert to lowercase
    headline =  re.sub('[^a-zA-Z]',' ',headline)
    headline = headline.lower()
    
    # drop stopwords
    headline_words = headline.split(' ')
    headline_words = [word for word in headline_words if not word in stopwords.words('english')]
    
    # use stemming to simplify words
    ps = PorterStemmer()
    headline_words = [ps.stem(word) for word in headline_words]
    
    # join sentence back again
    return ' '.join(headline_words)

# these functions should only go towards the training data only
def get_headline_df(X_train):
    
    headlines_lst = []
    target_lst = []
    
    # iter through every headline.
    for row in range(0,len(X_train.index)):
        if X_train['headline'].iloc[row] != 'null':
            for sentence in X_train['headline'].iloc[row]:
                headlines_lst.append(clean_headlines(sentence))
                target_lst.append(round_scores(X_train['returnsOpenNextMktres10'].iloc[row]))
            
    # return dataframe
    return pd.DataFrame({'headline':pd.Series(headlines_lst), 'returnsOpenNextMktres10':pd.Series(target_lst)})
    
def get_headline(headlines_df):
    
    # get headlines as list (use only headline_df produced by get_headline_df)
    headlines_lst = []
    for row in range(0,len(headlines_df.index)):
        if headlines_df.iloc[row] != 'null':
            headlines_lst.append(headlines_df.iloc[row])

    # split headlines to separate words
    basicvectorizer = CountVectorizer()
    headlines_vectorized = basicvectorizer.fit_transform(headlines_lst) # error found here (probably in regards to get_headline_df)
    
    print(headlines_vectorized.shape)
    return headlines_vectorized, basicvectorizer

def headline_mapping(target, headlines_vectored, headline_vectorizer):
    
    print(np.asarray(target).shape)
    headline_model = LogisticRegression()
    headline_model = headline_model.fit(headlines_vectored, target)
    
    # get coefficients
    basicwords = headline_vectorizer.get_feature_names()
    basiccoeffs = headline_model.coef_.tolist()[0]
    coeff_df = pd.DataFrame({'Word' : basicwords, 
                            'Coefficient' : basiccoeffs})
    
    # dont forget to remove null key
    
    # convert dataframe to dictionary of coefficients
    coefficient_dict = dict(zip(coeff_df.Word, coeff_df.Coefficient))

    return coefficient_dict, coeff_df['Coefficient'].mean()

# for predictions
def get_coeff_col(X, coeff_dict, coeff_default):
    
    def get_coeff(word_lst):
        
        # iter through every word
        coeff_sum = 0
        for word in word_lst:
            if word in coeff_dict:
                coeff_sum += coeff_dict[word]
            else:
                coeff_sum += coeff_default
        
        # get average coefficient
        coeff_score = coeff_sum / len(word_lst)
        return coeff_score
        
    basicvectorizer = CountVectorizer()
    
    # loop through every item
    headlines_coeff_lst = []
    for row in range(0,len(X['headline'].index)):
        coeff_score = 0
        if X['headline'].iloc[row] == 'null':
            headlines_coeff_lst.append(np.nan)
            break
        for i in range(0,len(X['headline'].iloc[row])):
            coeff_score += get_coeff(clean_headlines(str(X['headline'].iloc[row][i])).split(' '))
        headlines_coeff_lst.append(coeff_score / len(X['headline'].iloc[row]))
        
    # merge coefficient frame with main
    coeff_mean_df = pd.DataFrame({'headline_coeff_mean': pd.Series(headlines_coeff_lst)})
    X = pd.concat([X.reset_index(), coeff_mean_df], axis=1)
    
    return X


# In[ ]:

headline_df = get_headline_df(X_train)
coefficient_dict, coefficient_default = headline_mapping(headline_df['returnsOpenNextMktres10'],
                                            *get_headline(headline_df['headline']))


# In[ ]:

# will be applied to X_test as well
X_train = get_coeff_col(X_train, coefficient_dict, coefficient_default)


# ### News Groupby Features
# We are going to be looking specifically at the ``audiences`` and ``subjects`` column from the news dataframe. Here are some ideas:
# * Get the number of times a certain subject/audience occurs, and get the sum of the list of audiences/subjects.

# In[ ]:

# this is set up for list of strings columns
def get_feature_count(X_feat):
    
    # get list
    item_lst = []
    for row in range(0,len(X_feat.index)):
        if X_feat.iloc[row] != 'null':
            for i in range(0, len(X_feat.iloc[row])):
                item_lst.append(X_feat.iloc[row][i])
    
    # get unique items
    unique_feats = set(item_lst)
    
    # get frequency dictionary
    item_map = {}
    for i in unique_feats:
        item_map[i] = len([n for n in item_lst if n == i])
    
    return item_map

def get_feature_count_total(X_feat, item_map):
    
    # iter through every item and get total count
    counts = []
    for row in range(0,len(X_feat.index)):
        count = 0
        if X_feat.iloc[row] != 'null':
            for i in range(0, len(X_feat.iloc[row])): # this is what is causing the error.
                count += item_map[X_feat.iloc[row][i]]
        counts.append(count)
            
    return pd.Series(counts)
    
def news_grouping_features(X):
    
    # account for all possible nulls
    X[['audiences', 'subjects']] = X[['audiences', 'subjects']].fillna('null')
    
    # get map
    audience_map = get_feature_count(X['audiences'])
    subjects_map = get_feature_count(X['subjects'])
    
    # get count of each item
    X['audiences_count'] = get_feature_count_total(X['audiences'], get_feature_count(X['audiences']))
    X['subjects_count'] = get_feature_count_total(X['subjects'], get_feature_count(X['subjects']))
    
    return X


# In[ ]:

X_train = news_grouping_features(X_train)


# ### Clustering
# We are going to be clustering a few columns together (mainly to see how this will affect our results). 

# In[ ]:

def clustering(df):

    def cluster_modelling(features):
        df_set = df[features]
        cluster_model = KMeans(n_clusters = 8)
        cluster_model.fit(df_set)
        return cluster_model.predict(df_set)
    
    # get columns:
    vol_cols = [f for f in df.columns if f != 'volume' and 'volume' in f]
    novelty_cols = [f for f in df.columns if 'novelty' in f]
    
    # fill nulls
    cluster_cols = novelty_cols + vol_cols + ['open', 'close']
    df[cluster_cols] = df[cluster_cols].fillna(0)
    
    df['cluster_open_close'] = cluster_modelling(['open', 'close'])
    df['cluster_volume'] = cluster_modelling(vol_cols)
    df['cluster_novelty'] = cluster_modelling(novelty_cols)
    
    return df


# In[ ]:

X_train = clustering(X_train)


# ### Extra Features
# 
# Here are some basic extra features from other notebooks.

# In[ ]:

def extra_features(df):
    
    # Adding daily difference
    new_col = df["close"] - df["open"]
    df.insert(loc=6, column="daily_diff", value=new_col)
    df['close_to_open'] =  np.abs(df['close'] / df['open'])
    
    return df


# In[ ]:

X_train = extra_features(X_train)


# ### Get Time Features
# 
# This section splits the timestamp column into their own separate columns, as well as other various time features.
# 
# Possible idea: Encoding time?

# In[ ]:

# ripped from my previous kernel, NYC Taxi Fare

# first get dates
def split_time(df):
    
    # split date_time into categories
    df['time_day'] = df['time'].str.slice(8,10)
    df['time_month'] = df['time'].str.slice(5,7)
    df['time_year'] = df['time'].str.slice(0,4)
#     df['time_hour'] = df['time'].str.slice(11,13)
#     df['time_minute'] = df['time'].str.slice(14,16)
    
    # source: https://www.kaggle.com/nicapotato/taxi-rides-time-analysis-and-oof-lgbm
    df['temp_time'] = df['time'].str.replace(" UTC", "")
    df['temp_time'] = pd.to_datetime(df['temp_time'], format='%Y-%m-%d %H')
    
    df['time_day_of_year'] = df.temp_time.dt.dayofyear
    df['time_week_of_year'] = df.temp_time.dt.weekofyear
    df["time_weekday"] = df.temp_time.dt.weekday
    df["time_quarter"] = df.temp_time.dt.quarter
    
    del df['temp_time']
    gc.collect()
    
    # convert to non-object columns
    time_feats = ['time_day', 'time_month', 'time_year']
    df[time_feats] = df[time_feats].apply(pd.to_numeric)
    
    # determine whether the day is set on a holiday
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2007-01-01', end='2018-09-27').to_pydatetime()
    df['on_holiday'] = df['time'].str.slice(0,10).apply(lambda x: 1 if x in holidays else 0)
    
    return df


# In[ ]:

X_train = split_time(X_train)


# Here we remove all the excess columns and use label encoding on the assetName column.

# In[ ]:

def misc_adjustments(X):
    del_cols = ['index'] + [f for f in X.columns if X[f].dtype == 'object'] #and f != 'assetName']
    for f in del_cols:
        del X[f]
        
    # encode data
    le = LabelEncoder()
#     X = X.assign(assetCode = le.fit_transform(X.assetCode),
#                 assetName = le.fit_transform(X.assetName))
    X = X.assign(assetName = le.fit_transform(X.assetName))
    
    return X


# In[ ]:

X_train = misc_adjustments(X_train)


# ### Compile X functions into one function
# 
# This will be used when looping through different batches of X_test

# In[ ]:

def get_X(market_df, news_df):
    
    # these are all the functions applied to X_train except for a few
    market_df = mean_volume(market_df)
    X_test = join_market_news(market_df, news_df, nulls=True)
    X_test = get_coeff_col(X_test, coefficient_dict, coefficient_default)
    X_test = news_grouping_features(X_test)
    X_test = clustering(X_test)
    X_test = extra_features(X_test)
    X_test = split_time(X_test)
    X_test = misc_adjustments(X_test)
    
    return X_test


# #### Resulting Dataframe and Data Correlation to Target column
# We have went to roughly 50 columns to 135!

# In[ ]:

X_train.head(10)


# ### Using LightGBM for Modelling
# 
# We are going to use parameters from a notebook for modelling our data, as well as looping through the data until we reach a certain score.  Using the hyperparameters from (https://www.kaggle.com/kazuokiriyama/tuning-hyper-params-in-lgbm-achieve-0-66-in-lb)
# 
# Notes: 
# * Might possibly add bayesian optimization if necessary?
# * Using Neural Networks?
# * Remove unimportant features via modelling.

# In[ ]:

def set_data(X_train):

    # get X and Y
    y_train = X_train['returnsOpenNextMktres10']
    del X_train['returnsOpenNextMktres10'], X_train['universe']
    
    # split data (for cross validation)
    x1, x2, y1, y2 = train_test_split(X_train, 
                                      y_train, 
                                      test_size=0.25, 
                                      random_state=99)
    
    # get columns
    train_cols = X_train.columns.tolist()
#     categorical_cols = ['assetName']
    
    # convert to LGBM Data Structures
    dtrain = lgb.Dataset(x1.values, y1, feature_name=train_cols) #categorical_feature=categorical_cols)
    dvalid = lgb.Dataset(x2.values, y2, feature_name=train_cols) #categorical_feature=categorical_cols)
    
    return dtrain, dvalid

# https://www.kaggle.com/kazuokiriyama/tuning-hyper-params-in-lgbm-achieve-0-66-in-lb
def lgbm_training(dtrain, dvalid):
    
    # hyperparameters
    x_1 = [0.19000424246380565, 2452, 212, 328, 202]
    x_2 = [0.19016805202090095, 2583, 213, 312, 220]
    
    # Set up the LightGBM params
    params_1 = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'learning_rate': x_1[0],
        'num_leaves': x_1[1],
        'min_data_in_leaf': x_1[2],
        'num_iteration': x_1[3],
        'max_bin': x_1[4],
        'verbose': 1
    }

    params_2 = {
            'task': 'train',
            'boosting_type': 'dart',
            'objective': 'binary',
            'learning_rate': x_2[0],
            'num_leaves': x_2[1],
            'min_data_in_leaf': x_2[2],
            'num_iteration': x_2[3],
            'max_bin': x_2[4],
            'verbose': 1
        }
    
    # train data
    gbm_1 = lgb.train(params_1,
        dtrain,
        num_boost_round=100,
        valid_sets=(dvalid,),
        early_stopping_rounds=5)
        
    gbm_2 = lgb.train(params_2,
            dtrain,
            num_boost_round=1000,
            valid_sets=(dvalid,),
            early_stopping_rounds=5)
        
    return gbm_1, gbm_2


# In[ ]:

gbm_1, gbm_2 = lgbm_training(*set_data(X_train.copy()))


# In[ ]:

def predict_w_gbm(X_test):
    
    # get predictions
    pred_1 = gbm_1.predict(X_test)
    pred_2 = gbm_2.predict(X_test)
    
    # return mean
    return np.clip((pred_1 + pred_2) / 2, -1, 1)


# ### Making Predictions
# 
# Now the difference between the training and test data would be these two columns,  ``['returnsOpenNextMktres10', 'universe']``. We will be trying to predict ``returnsOpenNextMktres10`` and using that as the ``confidenceValue``.

# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u"\ndef make_predictions(market_obs_df, news_obs_df):\n    \n    # predict using given model\n    X_test = get_X(market_obs_df, news_obs_df)\n    prediction_values = predict_w_gbm(X_test)\n\n    return prediction_values\n\nfor (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days(): # Looping over days from start of 2017 to 2019-07-15\n    \n    # make predictions\n    predictions_template_df['confidenceValue'] = make_predictions(market_obs_df, news_obs_df)\n    \n    # save predictions\n    env.predict(predictions_template_df)")


# ### Export Submission

# In[ ]:

# exports csv
env.write_submission_file()
print('finished!')


# ### References:
# * [Getting Started - DJ Sterling](https://www.kaggle.com/dster/two-sigma-news-official-getting-started-kernel)
# * [a simple model - Bruno G. do Amaral](https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-data)
# * [LGBM Model - the1owl](https://www.kaggle.com/the1owl/my-two-sigma-cents-only)
# * [Headline Processing - Andrew Gelé](https://www.kaggle.com/ndrewgele/omg-nlp-with-the-djia-and-reddit)
# * [Feature engineering - Andrew Lukyanenko](https://www.kaggle.com/artgor/eda-feature-engineering-and-everything)
# * [Basic Text Processing - akatsuki06](https://www.kaggle.com/akatsuki06/basic-text-processing-cleaning-the-description)
# * [The fallacy of encoding assetCode - marketneutral](https://www.kaggle.com/marketneutral/the-fallacy-of-encoding-assetcode)
# * [Hyperparameters - kazuokiriyama](https://www.kaggle.com/kazuokiriyama/tuning-hyper-params-in-lgbm-achieve-0-66-in-lb)
