
# coding: utf-8

# # Market Data Only Baseline
# 
# Using a lot of ideas from NN Baseline Kernel.
# see. https://www.kaggle.com/christofhenkel/market-data-nn-baseline

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews
import gc


# In[ ]:

env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()
gc.enable()


# In[ ]:

#10:00之后的算成下一天，似乎有不好的影响
# index = news_train['time'][news_train['time'].dt.hour > 22].index
# news_train.loc[index,'time']  = news_train.loc[index,'time'].dt.ceil('d')
news_train['time'] = news_train['time'].dt.floor('d')
cols = ['sentimentNegative','sentimentNeutral','sentimentPositive','relevance','companyCount','bodySize','sentenceCount','wordCount','firstMentionSentence',
        'sentimentWordCount','takeSequence','sentimentClass','noveltyCount12H', 'noveltyCount24H','noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 
        'volumeCounts12H','volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D','volumeCounts7D']

news_total = news_train[['time','assetName'] + cols].copy()
del news_train
gc.collect()
news_train = news_total
print(news_train.columns)


# In[ ]:

import warnings
warnings.filterwarnings(action ='ignore',category = DeprecationWarning)

#直接相乘内存会爆掉，成之后，变成了0.66912，比最好0.66972差了一点，暂时不成
# for col in cols:
#     if col != 'relevance':
#         print(col)
#         news_train[col] = news_train[col] * news_train['relevance']
#聚合每一个日期前三天内的新闻数据，影响股价走势
#之前的版本，直接复制几份，然后和market_train进行join，代价较大
#直接进行news data的join
def get_news_train(raw_data,days = 6):
    news_last = pd.DataFrame()
    #衰减系数
    rate = 1.0
    for i in range(days):
        cur_train = raw_data[cols] * rate 
        rate *= 0.9
        cur_train['time'] = raw_data['time'] + datetime.timedelta(days = i,hours=22)
        cur_train['key'] = cur_train['time'].astype(str)+ raw_data['assetName'].astype(str)
        cur_train = cur_train[['key'] + cols].groupby('key').sum()
        cur_train['key'] = cur_train.index.values
        news_last = pd.concat([news_last, cur_train[['key'] + cols]])
        del cur_train
        gc.collect()
        print("after concat the shape is:",news_last.shape)
        news_last = news_last.groupby('key').sum()
        news_last['key'] = news_last.index.values
        print("the result shape is:",news_last.shape)
       
    del news_last['key']
    return news_last

news_last = get_news_train(news_train)
print(news_last.shape)
print(news_last.head())
print(news_last.dtypes)


# In[ ]:

market_train['key'] = market_train['time'].astype(str) + market_train['assetName'].astype(str)
market_train = market_train.join(news_last,on = 'key',how='left')
print(market_train['sentimentNeutral'].isnull().value_counts())
market_train.head()


# In[ ]:

# print(market_train['assetName'].nunique())
# print(news_train['assetName'].nunique())
# 通过assetName 判断有market 中有12万个example没在 news中出现,通过时间进行join，交集太少，目前感觉使用
# assetName比较合适
# print(market_train['assetName'].isin(news_train['assetName']).value_counts())
# print(market_train['time'].nunique())
# print(news_train['time'].nunique())
# print(news_train['time'].describe())


# In[ ]:

cat_cols = ['assetCode','assetName']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10','sentimentNegative','sentimentNeutral','sentimentPositive','relevance','companyCount','bodySize',
            'sentenceCount','wordCount','firstMentionSentence']


# In[ ]:

from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.25, random_state=23)


# # Handling categorical variables

# In[ ]:

def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id
encoders = [{} for i in range(len(cat_cols))]
for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market_train.loc[train_indices, cat].unique())}
    market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets


# # Handling numerical variables

# In[ ]:

from sklearn.preprocessing import StandardScaler 
import matplotlib
# market_train[num_cols] = market_train[num_cols].fillna(0)
#异常点过滤
# print(market_train['close'][market_train['close'] > 1000].count())
# print(market_train['open'][market_train['open'] > 1000].count())
# print(market_train['volume'][market_train['volume'] > 1e+08].count())

market_train['close'].clip(upper = 1000, inplace = True)
market_train['open'].clip(upper = 1000, inplace = True)
market_train['volume'].clip(upper = 1e+08, inplace = True)

# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# prices = pd.DataFrame({"close":market_train["close"], "log(close + 1)":np.log1p(market_train["close"])})
# prices.hist(bins = 10)
# 一定同时进行待预测数据集的对数转换
# 开盘价，收盘价不太符合正态分布，进行一个对数转换
# market_train['close'] = np.log1p(market_train['close'])
# market_train['open'] = np.log1p(market_train['open'])

print('scaling numerical columns')
scaler = StandardScaler()
col_mean = market_train[num_cols].mean()
market_train[num_cols]=market_train[num_cols].fillna(col_mean)

scaler = StandardScaler()
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])
# market_train.describe()
# market_train[num_cols].isna()
# market_train['returnsClosePrevMktres1'].isnull().value_counts()


# # Prepare data

# In[ ]:

def get_input(market_train, indices):
    X = market_train.loc[indices, num_cols]
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train, train_indices)

X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train, val_indices)
X_train.shape
print(X_valid.shape)


# # Train  model using hyperopt to auto hyper_parameters turing

# In[ ]:


from xgboost import XGBClassifier
import lightgbm as lgb
from functools import partial
from hyperopt import hp, fmin, tpe
from sklearn.metrics import mean_squared_error
algo = partial(tpe.suggest, n_startup_jobs=10)
def auto_turing(args):
    #model = XGBClassifier(n_jobs = 4, n_estimators = args['n_estimators'],max_depth=6)
    model = lgb.LGBMClassifier(n_estimators=args['n_estimators'])
    model.fit(X_train,y_train.astype(int))
    confidence_valid = model.predict(X_valid)*2 -1
    score = accuracy_score(confidence_valid>0,y_valid)
    print(args,score)
    return -score
# space = {"n_estimators":hp.choice("n_estimators",range(20,200))}
# print(fmin)
# best = fmin(auto_turing, space, algo=algo,max_evals=30)
# print(best)

# 单机xgb程序
model = XGBClassifier(n_jobs = 4, n_estimators = 50, max_depth=6)
model.fit(X_train,y_train.astype(int))
confidence_valid = model.predict(X_valid)*2 -1
score = accuracy_score(confidence_valid>0,y_valid)
print(score)
print("MSE")
print(mean_squared_error(confidence_valid > 0, y_valid.astype(float)))
# 单机lgb程序
# import lightgbm as lgb
# model = lgb.LGBMClassifier(n_estimators=70)
# model.fit(X_train,y_train.astype(int))
# confidence_valid = model.predict(X_valid)*2 -1
# score = accuracy_score(confidence_valid>0,y_valid)
# print(score)

# from sklearn.ensemble import RandomForestClassifier
# distribution of confidence that will be used as submission
# plt.hist(confidence_valid, bins='auto')
# plt.title("predicted confidence")
# plt.show()
# these are tuned params I found
 


# Result validation

# In[ ]:

# calculation of actual metric that is used to calculate final score
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)
market_train.describe()


# # Prediction

# In[ ]:

days = env.get_prediction_days()


# In[ ]:

n_days = 0
predicted_confidences = np.array([])
from collections import deque
news_pre = deque()
news_all = pd.DataFrame()
BaseMod = 50
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    news_all = pd.concat([news_all,news_obs_df])
    if n_days >= BaseMod and n_days % BaseMod >= 0 and n_days % BaseMod < 8:
        news_pre.append(news_obs_df)
    elif n_days >= BaseMod and n_days % BaseMod == 8:
        del news_all
        gc.collect()
        news_all = pd.DataFrame()
        for item in news_pre:
            news_all = pd.concat([news_all,item])
        news_pre.clear()
    
#     index = news_all['time'][news_all['time'].dt.hour > 22].index
#     news_all.loc[index,'time']  = news_all.loc[index,'time'].dt.ceil('d')
    news_all['time'] = news_all['time'].dt.floor('d')
    news_last = pd.DataFrame()
    
#     for col in cols:
#         if col != 'relevance':
#             print(col)
#             news_all[col] = news_all[col] * news_all['relevance']
    #聚合每一个日期前三天内的新闻数据，影响股价走势
    news_last = get_news_train(news_all)

    market_obs_df['key'] = market_obs_df['time'].astype(str) + market_obs_df['assetName'].astype(str)
    market_obs_df = market_obs_df.join(news_last,on = 'key',how='left')
    
    #异常点过滤
    market_obs_df['close'].clip(upper = 1000, inplace = True)
    market_obs_df['open'].clip(upper = 1000, inplace = True)
    market_obs_df['volume'].clip(upper = 1e+08, inplace = True)
    
    # 对数转换
#     market_obs_df['close'] = np.log1p(market_obs_df['close'])
#     market_obs_df['open'] = np.log1p(market_obs_df['open'])
    
#     col_mean = [num_cols].mean()
    #归一化
    market_obs_df[num_cols]=market_obs_df[num_cols].fillna(col_mean)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    X_test = market_obs_df[num_cols]
    X_test['assetCode'] = market_obs_df['assetCode'].apply(lambda x: encode(encoders[0], x)).values
    X_test['assetName'] = market_obs_df['assetName'].apply(lambda x: encode(encoders[1], x)).values

    
    market_prediction = model.predict(X_test)*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))

    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    del news_last
    gc.collect()

env.write_submission_file()


# In[ ]:

# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()

