# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# import the datasets
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market, news) = env.get_training_data()

# check if there are abnormal prices changes in a single day

market['price_diff'] = market['close'] - market['open']
market['close/open'] = market['close'] / market['open']
market['assetCode_mean_open'] = market.groupby('assetCode')['open'].transform('mean')
market['assetCode_mean_close'] = market.groupby('assetCode')['close'].transform('mean')

# replace abnormal data record

for i, row in market.loc[market['close/open'] >= 1.5].iterrows():
    if np.abs(row['assetCode_mean_open'] - row['open']) > np.abs(row['assetCode_mean_close'] - row['close']):
        market.iloc[i,5] = row['assetCode_mean_open']
    else:
        market.iloc[i,4] = row['assetCode_mean_close']
    

for i, row in market.loc[market['close/open'] <= 0.5].iterrows():
    if np.abs(row['assetCode_mean_open'] - row['open']) > np.abs(row['assetCode_mean_close'] - row['close']):
        market.iloc[i,5] = row['assetCode_mean_open']
    else:
        market.iloc[i,4] = row['assetCode_mean_close']
        

fill_cols = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevMktres10']
                
market = market.sort_values(by = ['assetCode','time'], ascending=[True, True])

for i in market[fill_cols]:
    market[i] = market[i].fillna('ffill')

market = market.drop(['price_diff','close/open','assetCode_mean_open','assetCode_mean_close'], axis = 1)

#market.info()

### working on news data

news_simplified = news.drop(['sourceTimestamp','provider', 'sourceId', 'headline','takeSequence','subjects', 'audiences',
       'bodySize', 'companyCount', 'headlineTag', 'marketCommentary','sentenceCount', 'wordCount',
       'firstMentionSentence','sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
       'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
       'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
       'volumeCounts7D'], axis = 1)

def data_prep(market_df, news_df):
    market_df['time'] = market_df.time.dt.date
    news_df['firstCreated'] = news_df.firstCreated.dt.date
    news_df['assetCodeLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))
    news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
    market_df = pd.merge(market_df, news_df, how='left', left_on=['time','assetCode'], right_on=['firstCreated', 'assetCodes'])
    lbl = {k: v for v, k in enumerate(market_df['assetCode'].unique())}
    market_df['assetCodeT'] = market_df['assetCode'].map(lbl)
    market_df = market_df.dropna(axis=0)

    return market_df
    
combined = data_prep(market, news_simplified)
#combined.isna().sum()


#combined.info()

kcols = [c for c in combined.columns if c not in['assetCode', 'assetCodes', 'assetCodesLen', 'assetName_x','assetName_y', 'assetCodeT','firstCreated','time_x','time_y','universe','assetCodeLen']]

X = combined[kcols]

# X.info()

'''
def to_numeric(dataset):
    for i in dataset.columns:
        if dataset[i].dtype == 'object':
            dataset[i] = dataset[i].apply(pd.to_numeric, errors='coerce')
        else:
            pass
    return dataset

X = to_numeric(X)

'''

# normalize data

'''
scaler = Normalizer()
X_norm = scaler.fit_transform(X)
'''
