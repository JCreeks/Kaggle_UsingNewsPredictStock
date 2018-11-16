
# coding: utf-8

# # Amateur Hour - Using Headlines to Predict Stocks
# ### Starter Kernel by ``Magichanics`` 
# *([GitHub](https://github.com/Magichanics) - [Kaggle](https://www.kaggle.com/magichanics))*
# 
# Stocks are unpredictable, but can sometimes follow a trend. In this notebook, we will be discovering the correlation between the stocks and the news. After a few tries, my best score not including this one is ``0.54194`` from [V29](https://www.kaggle.com/magichanics/amateur-hour-using-headlines-to-predict-stocks/code?scriptVersionId=6466412). Right now, I'm trying to find answers as to why my score keeps fluctuating from ``-0.26`` to ``0.55``.  A lot of the scrapped code that I was going to use can be found on my GitLab account. Since the data provided for both the final submissions and training/test, we will be accounting for no new ``subjects``, ``assetCode``, ``assetName`` and ``audiences``.
# 
# If there are any things that you would like me to add or remove, feel free to comment down below. I'm mainly doing this to learn and experiment with the data. I plan on rewriting a lot of code in the future to make it look nicer, since a lot of the stuff I have written may not be the most efficient way to approach specific problems.
# 
# Also, thanks for the upvotes! I'll be making another kernel for this competition later on.
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

import warnings
warnings.filterwarnings('ignore')

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

# neural networks

# import environment for data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:

sampling = False


# In[ ]:

(market_train_df, news_train_df) = env.get_training_data()

# its best to get the data by its tail
if sampling:
    market_train_df = market_train_df.tail(400_000)
    news_train_df = news_train_df.tail(1_000_000)
else:
    market_train_df = market_train_df.tail(3_000_000) # 3m to 6m was too much?
    news_train_df = news_train_df.tail(6_000_000)


# In[ ]:

market_train_df.head()


# In[ ]:

news_train_df.head()


# ### Information on the Training Data
# * There are no Unknown ``assetName`` in ``news_train_df``, but there are 24 479 rows with Unknown as the ``assetName`` in ``market_train_df``. Merging by ``assetCode`` leaves out Unknown rows, which could be problematic.
# * ``Volume`` has the highest correlation in terms of ``returnsOpenNextMktres10``.
# * Merging by just ``assetCodes`` greatly increases the dataframe (with just 100k rows, it has turned into 10 million rows), although merging by ``assetCodes`` and ``time`` greatly decrease the original dataframe.

# ### Getting rid of "Fake" News
# There are some things to notice about the news:
# * Some headlines' length have a value of 0. (roughly 300-500 rows?)
# * One Day delayed news (calculated by difference between ``sourceTimestamp`` and ``time``

# In[ ]:

def clean_news(news_df):
    
    # get rid of blank headlines
    news_df = news_df[news_df.headline != '']
    
    # get rid of delayed news
    news_df['news_delay'] = news_df['time'] - news_df['sourceTimestamp']
    news_df = news_df[news_df.news_delay < datetime.timedelta(days=1)]
    
    # get rid of headline articles with only 1 takeSequence (experimental)
#     news_df = news_df[(news_df.takeSequence != 1) | (news_df.urgency == 1)]
    
    return news_df


# In[ ]:

news_train_df = clean_news(news_train_df)


# In[ ]:

news_train_df.shape


# ### Market Groupby Features
# We are going to group market data based on ``assetName`` and determine the median and mean of the volume.

# In[ ]:

def mean_volume(market_df):
    
    # groupby and return median
    vol_by_name_median = market_df[['volume', 'assetName']].groupby('assetName').median()['volume']
    vol_by_name_mean = market_df[['volume', 'assetName']].groupby('assetName').mean()['volume'] # could try mean?
    market_df['vol_by_name_median'] = market_df['assetName'].map(vol_by_name_median)
    market_df['vol_by_name_mean'] = market_df['assetName'].map(vol_by_name_mean)
    
    # get difference for median
    market_df['vol_by_name_median_diff'] = market_df['volume'] - market_df['vol_by_name_median']
    
    return market_df


# In[ ]:

market_train_df = mean_volume(market_train_df)


# ### [Simple Quant Features](https://www.kaggle.com/youhanlee/simple-quant-features-using-python)
# Please go check out YouHan Lee's notebook on Simple Quant Features!
# 
# In addition to this, there could be a chance that stocks that aren't associated with any sort of news to change in direction with a low standard deviation within a very small time window.

# In[ ]:

def market_quant_feats(market_df):
    
    # get moving average
    def moving_avg(X, feat1):
        X[feat1 + '_MA_7MA'] = X[feat1].rolling(window=7).mean()
        X[feat1 + '_MA_15MA'] = X[feat1].rolling(window=15).mean()
        X[feat1 + '_MA_30MA'] = X[feat1].rolling(window=30).mean()
        X[feat1 + '_MA_60MA'] = X[feat1].rolling(window=60).mean()
        return X
        
    # get moving standard deviation
    def moving_std(X, feat1):
        X[feat1 + '_MA_1std'] = X[feat1].rolling(window=1).std()
        X[feat1 + '_MA_3std'] = X[feat1].rolling(window=3).std()
        X[feat1 + '_MA_5std'] = X[feat1].rolling(window=5).std()
        return X
        
    market_df = moving_avg(market_df, 'close')
    market_df = moving_avg(market_df, 'volume')
    market_df = moving_std(market_df, 'volume')
    
    # bollinger band
    no_of_std = 2
    market_df['MA_7MA'] = market_df['close'].rolling(window=7).mean()
    market_df['MA_7MA_std'] = market_df['close'].rolling(window=7).std() 
    market_df['MA_7MA_BB_high'] = market_df['MA_7MA'] + no_of_std * market_df['MA_7MA_std']
    market_df['MA_7MA_BB_low'] = market_df['MA_7MA'] - no_of_std * market_df['MA_7MA_std']
    market_df['MA_15MA'] = market_df['close'].rolling(window=15).mean()
    market_df['MA_15MA_std'] = market_df['close'].rolling(window=15).std() 
    market_df['MA_15MA_BB_high'] = market_df['MA_15MA'] + no_of_std * market_df['MA_15MA_std']
    market_df['MA_15MA_BB_low'] = market_df['MA_15MA'] - no_of_std * market_df['MA_15MA_std']
    market_df['MA_30MA'] = market_df['close'].rolling(window=30).mean()
    market_df['MA_30MA_std'] = market_df['close'].rolling(window=30).std() 
    market_df['MA_30MA_BB_high'] = market_df['MA_30MA'] + no_of_std * market_df['MA_30MA_std']
    market_df['MA_30MA_BB_low'] = market_df['MA_30MA'] - no_of_std * market_df['MA_30MA_std']
    
    # Adding daily difference
    new_col = market_df["close"] - market_df["open"]
    market_df.insert(loc=6, column="daily_diff", value=new_col)
    market_df['close_to_open'] =  np.abs(market_df['close'] / market_df['open'])
    
    return market_df


# In[ ]:

market_train_df = market_quant_feats(market_train_df)


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
# * Add Three Day Moving Period, i.e. combine all data that happened between 3 days before the market time to the market time.

# In[ ]:

def generalize_time(X):
    # convert time to string and/or get rid of Hours, Minutes, and seconds
    X['time'] = X['time'].dt.strftime('%Y-%m-%d %H:%M:%S').str.slice(0,19) #(0,10) for Y-m-d, (0,13) for Y-m-d H
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
    while curr_nulls < num_nulls and iteration < len(indecies):
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
    
    # apply countvectorizer to get features of subject and audience (commented this out for V59)
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    for f in ['audiences', 'subjects']:
        X_codes = vectorizer.fit_transform(news_df_cat[f].tolist())
        col_names = vectorizer.get_feature_names()
        X_codes = pd.DataFrame(X_codes.toarray())
        X_codes.columns = col_names
        news_df_cat = pd.concat([news_df_cat, X_codes], axis=1)
        del news_df_cat[f]
        
    # merge
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
    market_df['headline'] = market_df['headline'].fillna('null')

    return market_df


# if there is a joining error, it means that the dataframes have no correlation with each other (solution: increase train dataset)


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'X_train = join_market_news(market_train_df, news_train_df, nulls=False, num_nulls=500_000)')


# In[ ]:

X_train.head()


# In[ ]:

X_train.shape


# ### Text Processing with Logistic Regression
# 
# We are going to vectorize the headlines and apply logistic regression (labels being binary as to whether the stocks go up or not). In a nutshell, it splits the headlines into individual words, filters out unecessary words to prevent abnormal results, vectorizes it for modelling, and then with the target column provided, we could create a dataframe of coefficients that we could use as a feature in the dataframe! Right now I am just getting the mean of the coefficients in each list of headlines. 
# 
# Note: 
# * May be useful to apply it to ``universe``, and possibly get the sum or standard deviation of the word coefficients?
# * Some key words that should appear in bottom/top?
#     * Free Trade Agreement
#     * Interest Rates
#     * Contracts
#     
# Update: We are now going to apply the same code on both ``audiences`` and ``subjects``

# In[ ]:

ps = PorterStemmer()

# reuse data
def round_scores(x):
    if x >= 0:
        return 1
    else:
        return 0

# this takes up a lot of time, so apply it when getting coefficients to filter out words.
def clean_headlines(headline):
    
    # remove numerical and convert to lowercase
    headline =  re.sub('[^a-zA-Z]',' ',headline)
    headline = headline.lower()
    
    # use stemming to simplify words
    headline_words = [ps.stem(word) for word in headline.split(' ')]
    
    # join sentence back again
    return ' '.join(headline_words)

# these functions should only go towards the training data only
def get_headline_df(X_train, col='headline'):
    
    # make sure this is a copy of the given dataframe
    # remove any headlines with nulls
    X_train = X_train[X_train[col] != 'null']
    
    headlines_lst = []
    target_lst = []
    
    # iter through every headline.
    for row in range(0,len(X_train.index)):
        if X_train[col].iloc[row] != 'null':
            for sentence in X_train[col].iloc[row]:
                headlines_lst.append(sentence)
                target_lst.append(round_scores(X_train['returnsOpenNextMktres10'].iloc[row]))
            
    # return dataframe
    return pd.DataFrame({col:pd.Series(headlines_lst), 'returnsOpenNextMktres10':pd.Series(target_lst)})
    
def get_headline(headlines_df):
    
    # get headlines as list (use only headline_df produced by get_headline_df)
    headlines_lst = []
    for row in range(0,len(headlines_df.index)):
        if headlines_df.iloc[row] != 'null':
            headlines_lst.append(clean_headlines(headlines_df.iloc[row]))

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
    
    # remove stopwords from the dataframe (wait do headlines even have stopwords to begin with?)
    #coeff_df = coeff_df[coeff_df.Word.isin(stopwords.words('english'))]
    
    # convert dataframe to dictionary of coefficients
    coefficient_dict = dict(zip(coeff_df.Word, coeff_df.Coefficient))

    return coefficient_dict, coeff_df['Coefficient'].mean()

# for predictions
def get_coeff_col(X, coeff_dict, coeff_default, col='headline'):
    
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
    for row in range(0,len(X[col].index)):
        coeff_score = 0
        if X[col].iloc[row] == 'null':
            headlines_coeff_lst.append(np.nan)
        else:
            for i in range(0,len(X[col].iloc[row])):
                curr_str_lst = list(filter(None,clean_headlines(str(X[col].iloc[row][i])).split(' ')))
                coeff_score += get_coeff(curr_str_lst) / len(curr_str_lst) # get averages here (only applies to headlines)
        headlines_coeff_lst.append(coeff_score)
        
    # merge coefficient frame with main
    X = X.reset_index()
    X[col + '_coeff_sum'] = pd.Series(headlines_coeff_lst)
    
    return X


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u"headline_df = get_headline_df(X_train.copy())\ncoefficient_dict, coefficient_default = headline_mapping(headline_df['returnsOpenNextMktres10'],\n                                            *get_headline(headline_df['headline']))")


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'# will be applied to X_test as well\nX_train = get_coeff_col(X_train, coefficient_dict, coefficient_default)')


# In[ ]:

X_train[['headline', 'headline_coeff_sum']].head()


# In[ ]:

X_train[X_train.headline_coeff_sum.isnull() == False]


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


# ### Cleaning Data
# Here we remove all the excess columns, and encode specific categorical features in preparation of training.

# In[ ]:

def misc_adjustments(X):
    
    # remove list-based columns
    del_cols = ['headline', 'assetCode']
    for f in del_cols:
        del X[f]
        
    # get categorical features
    cols_categorical = ['assetName', 'time']
        
    # encode remaining categorical features
    le = LabelEncoder()
    X[cols_categorical] = X[cols_categorical].apply(LabelEncoder().fit_transform)
    
    return X


# In[ ]:

X_train = misc_adjustments(X_train)


# ### Compile X functions into one function
# 
# This will be used when looping through different batches of X_test

# In[ ]:

def get_X(market_df, news_df):
    
    # these are all the functions applied to X_train except for a few
    news_df = clean_news(news_df)
    market_df = mean_volume(market_df)
    market_df = market_quant_feats(market_df)
    X_test = join_market_news(market_df, news_df, nulls=True)
    X_test = get_coeff_col(X_test, coefficient_dict, coefficient_default)
    X_test = clustering(X_test)
    X_test = misc_adjustments(X_test)
    
    return X_test


# #### Resulting Dataframe 
# 

# In[ ]:

X_train.head(10)


# In[ ]:

X_train.shape


# ### Using LGBM For Modelling
# What could go wrong?

# In[ ]:

def fixed_train_test_split(X, y, train_size):
    
    # round train size
    train_size = int(train_size * len(X))
    
    # split data
    X_train, y_train = X[train_size:], y[train_size:]
    X_valid, y_valid = X[:train_size], y[:train_size]
    
    return X_train, y_train, X_valid, y_valid

def set_data(X_train):
    
    # get X and Y
    y_train = X_train['returnsOpenNextMktres10']
    del X_train['returnsOpenNextMktres10'], X_train['universe']
    
    # split data (for cross validation)
    x1, y1, x2, y2 = fixed_train_test_split(X_train, 
                                              y_train, 
                                              0.8)
    
    # set lgbm dataframes
    dtrain = lgb.Dataset(x1, label=y1)
    dvalid = lgb.Dataset(x2, label=y2)
    
    return dtrain, dvalid
    
def lgbm_training(X_train):
    
    # set model and parameters
    params = {'learning_rate': 0.02, 
              'boosting': 'gbdt', 
              'objective': 'regression_l1', 
              'seed': 573,
            'sub_feature': 0.7,
            'num_leaves': 60,
            'min_data': 100,
            'verbose': -1}
    
    # get x and y values
    dtrain, dvalid = set_data(X_train)
    
    # train data
    lgb_model = lgb.train(params, 
                            dtrain, 
                            1000, 
                            valid_sets=(dvalid,), 
                            verbose_eval=25, 
                            early_stopping_rounds=100)
    
    return lgb_model


# In[ ]:

lgb_model = lgbm_training(X_train.copy())


# ### Making Predictions
# 
# Now the difference between the training and test data would be these two columns,  ``['returnsOpenNextMktres10', 'universe']``. We will be trying to predict ``returnsOpenNextMktres10`` and using that as the ``confidenceValue``.

# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u"\ndef make_predictions(market_obs_df, news_obs_df):\n    \n    # predict using given model\n    X_test = get_X(market_obs_df, news_obs_df)\n    prediction_values = np.clip(lgb_model.predict(X_test), -1, 1)\n\n    return prediction_values\n\nfor (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days(): # Looping over days from start of 2017 to 2019-07-15\n    \n    # make predictions\n    predictions_template_df['confidenceValue'] = make_predictions(market_obs_df, news_obs_df)\n    \n    # save predictions\n    env.predict(predictions_template_df)")


# ### Export Submission

# In[ ]:

# exports csv
env.write_submission_file()


# ### References:
# * [Getting Started - DJ Sterling](https://www.kaggle.com/dster/two-sigma-news-official-getting-started-kernel)
# * [a simple model - Bruno G. do Amaral](https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-data)
# * [LGBM Model - the1owl](https://www.kaggle.com/the1owl/my-two-sigma-cents-only)
# * [Headline Processing - Andrew Gelé](https://www.kaggle.com/ndrewgele/omg-nlp-with-the-djia-and-reddit)
# * [Feature engineering - Andrew Lukyanenko](https://www.kaggle.com/artgor/eda-feature-engineering-and-everything)
# * [Basic Text Processing - akatsuki06](https://www.kaggle.com/akatsuki06/basic-text-processing-cleaning-the-description)
# * [The fallacy of encoding assetCode - marketneutral](https://www.kaggle.com/marketneutral/the-fallacy-of-encoding-assetcode)
# * [Market Data NN Baseline  - dieter](https://www.kaggle.com/christofhenkel/market-data-nn-baseline)
# * [Aantonova Features - aantonova](https://www.kaggle.com/aantonova/797-lgbm-and-bayesian-optimization/notebook)
# * [Simple quant features using python - YouHan Lee](https://www.kaggle.com/youhanlee/simple-quant-features-using-python)
