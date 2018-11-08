
# coding: utf-8

# # Cleaning up market data
# ---

# In[ ]:

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:

(market_train_df, news_train_df) = env.get_training_data()
market_train_df['date'] = market_train_df['time'].dt.strftime('%Y-%m-%d')


# Starting out by looking at summary statistics for each of the numeric columns.  There are certainly some outliers and extreme values to check out.

# In[ ]:

market_train_df.describe().round(3)


# For example, the `open` price column has a maximum value of nearly 10,000 while the highest `close` price is only about 1578, so that's a clear red flag.  Looking at the `returnsClosePrevRaw1` column, there are assets that supposedly increased by around 4600% or lost nearly 100% their value in a single day.  Even more extreme day-over-day returns are seen in the `returnsOpenPrevRaw1` column.

# First I'm taking a look at assets that are showing very large drop in day-over-day close prices.  Below are all the observations showing `close` prices dropping by more than 70% day-over-day...

# In[ ]:

market_train_df[market_train_df['returnsClosePrevRaw1'] < -.7]


# There are 12 records showing a > 70% drop in daily `close` price.  The first record above from 2008 shows Bear Sterns stock dropping by 84%.  This jumps out as probably being accurate.  If you recall, Bear Stearns was an investment bank that failed due to losses largely associated with subprime mortgage backed securities.  Their failure occurred shortly before the broader global financial crisis during 2008 and 2009.

# ### Errors on 2016-07-06
# ---
# After a brief review of the data surrounding the 12 records above, it appears only four of these high negative `returnsClosePrevRaw1` values are due to data errors.  Each of the errors is for a record on 2016-07-06.  The assets `FLEX.O`, `MAT.O`, `SHLD.O`, and `ZNGA.O` are affected.  Below is a closer look at the errors...

# In[ ]:

someAssetsWithBadData = ['FLEX.O','MAT.O','SHLD.O','ZNGA.O']
someMarketData = market_train_df[(market_train_df['assetCode'].isin(someAssetsWithBadData)) 
                & (market_train_df['time'] >= '2016-07-05')
                & (market_train_df['time'] < '2016-07-08')].sort_values('assetCode')
someMarketData


# It appears that each of the columns `volume`, `close`, and `open` are inaccurate for the four records above on 2016-07-06.  I looked up what the correct historical volume and open/close prices were to confirm the errors.
# 
# Below is a graph showing the `close` price for each of these four stocks over the last couple years in the `market_train_df` dataset.  It's a little tough to see because the graphs overlap each other, but if you hover your cursor over the spike, you can see all four asset's `close` price shoots up to around 123.5 on 2016-07-06 (and then returns back to normal).

# In[ ]:

selectedAssets = market_train_df[(market_train_df['assetCode'].isin(someAssetsWithBadData))
                                 & (market_train_df['time'] >= '2015-01-01')]
selectedAssetsPivot = selectedAssets.pivot(index='date',columns='assetCode',values='close')
selectedAssetsPivot.iplot()


# Because the `close` and `open` prices are wrong, this has also affected many of the return columns for these stocks around the same time frame.  Both `raw` and `mktres` columns have been affected.
# 
# Some of the `mktres` return data that was skewed because of the innacurate close/open prices I wouldn't have expected.  For example, the `returnsClosePrevMktres1` column shows odd values for up to 20 days after the bad data row on 2016-07-06.  Looking at this return metric for Zynga, it has values of 249% on 2016-08-02, and -124% on 2016-08-03 which are obviously incorrect (see graph below).

# In[ ]:

sampleZynga = market_train_df[(market_train_df['assetCode'] == 'ZNGA.O')
                              & (market_train_df['time'] >= '2016-06-01')
                              & (market_train_df['time'] < '2016-09-01')]
sampleZynga.iplot(kind='line',x='date',y='returnsClosePrevMktres1')


# After reviewing the data around these four errors, each instance appeared to have the same pattern of bad data and affected rows and columns.  In each of the four errors, I found that 32 rows of return data had at least one column that needed to be fixed.  So I created a few functions to help fix the bad data for these errors.
# 
# It isn't entirely clear to me how to properly 'fix' the `mktres` data, since this metric is calculated behind the scenes based on the overall market and an individual stock's performance.  While I haven't had much of a chance to review it, this kernel looks into what the `mktres` return actual is and suggests it is based on a stock's beta coefficient... https://www.kaggle.com/marketneutral/eda-what-does-mktres-mean.
# 
# For now, I've opted to look at a individual stock's `raw` vs. `mktres` values for a particular return metric across a sample of days, and use a linear best fit line to approximate the `mktres` value I want to update.  Certainly far from ideal, but it's a relatively easy fix and should be good enough since the 'bad' `mktres` data I'm fixing isn't a big part of the market training dataset anyways.  Just to show an example, here's a scatter plot showing the roughly linear relationship between `returnsClosePrevRaw1` and `returnsClosePrevMktres1` for a sample of Zynga data unaffected by the data error...

# In[ ]:

quickZyngaSample = market_train_df[(market_train_df['assetCode'] == 'ZNGA.O')
                                   & ((market_train_df['time'] >= '2016-06-01') & (market_train_df['time'] < '2016-07-06')
                                      | (market_train_df['time'] >= '2016-08-04') & (market_train_df['time'] < '2016-09-01'))]
quickZyngaSample.iplot(kind='scatter',x='returnsClosePrevRaw1',y='returnsClosePrevMktres1',mode='markers')


# Finally, after the affected `PrevMktres` values were estimated, the bad `returnsOpenNextMktres10` values were updated based on the newly estimated `returnsOpenPrevMktres10` values.

# ### Fixing the 2016-07-06 Errors
# ---
# The helper functions and the fixes for the four 2016-07-06 errors are below.

# In[ ]:

def sampleAssetData(assetCode, date, numDays):
    d = datetime.strptime(date,'%Y-%m-%d')
    start = d - timedelta(days=numDays)
    end = d + timedelta(days=numDays)
    return market_train_df[(market_train_df['assetCode'] == assetCode)
                             & (market_train_df['time'] >= start.strftime('%Y-%m-%d'))
                             & (market_train_df['time'] <= end.strftime('%Y-%m-%d'))].copy()


# In[ ]:

def updateRawReturns(assetData, indices):
    for i in indices[:2]:
        market_train_df.loc[[i],['returnsClosePrevRaw1']] = assetData['close'].pct_change()
        market_train_df.loc[[i],['returnsOpenPrevRaw1']] = assetData['open'].pct_change()
    for j in [indices[0],indices[2]]:
        market_train_df.loc[[j],['returnsClosePrevRaw10']] = assetData['close'].pct_change(periods=10)
        market_train_df.loc[[j],['returnsOpenPrevRaw10']] = assetData['open'].pct_change(periods=10)


# In[ ]:

def estimateMktresReturn(sampleData, mktresCol, index):
    sampleData['ones'] = 1
    rawCol = mktresCol.replace('Mktres','Raw')
    A = sampleData[[rawCol,'ones']]
    y = sampleData[mktresCol]
    m, c = np.linalg.lstsq(A,y,rcond=-1)[0]
    return c + m * market_train_df.loc[index,rawCol]


# In[ ]:

def updateMktresReturns(assetCode, assetData, indices):
    # update range of values for returnsClosePrevMktres1, returnsOpenPrevMktres1, returnsClosePrevMktres10
    sample1 = assetData[(assetData.index < indices[0]) | (assetData.index > indices[3])]
    rowsToUpdate1 = assetData[(assetData.index >= indices[0]) & (assetData.index <= indices[3])]
    for index, row in rowsToUpdate1.iterrows():
        market_train_df.loc[[index],['returnsClosePrevMktres1']] = estimateMktresReturn(sample1,'returnsClosePrevMktres1',index)
        market_train_df.loc[[index],['returnsOpenPrevMktres1']] = estimateMktresReturn(sample1,'returnsOpenPrevMktres1',index)
        market_train_df.loc[[index],['returnsClosePrevMktres10']] = estimateMktresReturn(sample1,'returnsClosePrevMktres10',index)
    # update range of values for returnsOpenPrevMktres10
    sample2 = assetData[(assetData.index < indices[0]) | (assetData.index > indices[2])]
    rowsToUpdate2 = assetData[(assetData.index >= indices[0]) & (assetData.index <= indices[2])]
    l = []
    for index, row in rowsToUpdate2.iterrows():
        est = estimateMktresReturn(sample2,'returnsOpenPrevMktres10',index)
        l.append(est)
        market_train_df.loc[[index],['returnsOpenPrevMktres10']] = est
    # update range of values for returnsOpenNextMktres10
    rowsToUpdate3 = assetData[(assetData.index >= indices[4]) & (assetData.index <= indices[5])]
    i = 0
    for index, row in rowsToUpdate3.iterrows():
        market_train_df.loc[[index],['returnsOpenNextMktres10']] = l[i]
        i += 1


# In[ ]:

def fixBadReturnData(assetCode, badDate, badIndex):
    # store copy of bad data window
    dayWindow = 45
    badDataWindow = sampleAssetData(assetCode,badDate,dayWindow)
    badDataWindow.reset_index(inplace=True)
    # store indices needed to update raw and mktres return data
    newIdx = badDataWindow[badDataWindow['index'] == badIndex].index[0]
    indices = [badIndex,badDataWindow.loc[newIdx+1,'index'],badDataWindow.loc[newIdx+10,'index'],
               badDataWindow.loc[newIdx+20,'index'],badDataWindow.loc[newIdx-11,'index'],badDataWindow.loc[newIdx-1,'index']]
    badDataWindow.set_index('index',inplace=True)
    # correct bad raw return data
    updateRawReturns(badDataWindow,indices)
    # estimate affected mktres return data
    updateMktresReturns(assetCode,badDataWindow,indices)


# In[ ]:

# bad volume, open, and close for ZNGA.O on 2016-07-06
assetCode = 'ZNGA.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 19213100
market_train_df.loc[[badIndex],['open']] = 2.64
market_train_df.loc[[badIndex],['close']] = 2.75
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex)


# In[ ]:

# bad volume, open, and close for FLEX.O on 2016-07-06
assetCode = 'FLEX.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 5406600
market_train_df.loc[[badIndex],['open']] = 11.580
market_train_df.loc[[badIndex],['close']] = 11.750
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex)


# In[ ]:

# bad volume, open, and close for SHLD.O on 2016-07-06
assetCode = 'SHLD.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 279300
market_train_df.loc[[badIndex],['open']] = 12.8900
market_train_df.loc[[badIndex],['close']] = 13.1400
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex)


# In[ ]:

# bad volume, open, and close for MAT.O on 2016-07-06
assetCode = 'MAT.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 3242100
market_train_df.loc[[badIndex],['open']] = 32.13
market_train_df.loc[[badIndex],['close']] = 31.52
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex)


# After running the code above (which fixed the errors, corrected the `raw` returns, and estimated the affected `mktres` return values), the data around these errors looks much more reasonable. Revisiting the summary statistics, we've removed some of the extreme values previously seen in the `returnsClosePrevRaw1` column.

# In[ ]:

market_train_df.describe().round(3)


# ### Towers Watson open price error
# ---
# Now turning to the very high `open` price of 9998.99...

# In[ ]:

market_train_df[market_train_df['open'] > 2000]


# The record above for Towers Watson & Co appears to be an obvious error with an `open` price of 9998 and a `close` price of 50.  There's a similar error for Bank of New York Mellon Corp, however, it's likely I won't be using market data from 2007 and 2008 when modeling, so for now I'm going to ignore this error.
# 
# After reviewing the first couple years of data available on Towers Watson (`TW.N`), we can see there are a few months where the available market data is pretty spotty.  The record with the outlier ~10k `open` price is from Jan 2010.  From the histogram below, we can see that those four observations from Jan 2010 are the first data available for Towers Watson, and come several months before the next observations are available.

# In[ ]:

twSample = market_train_df[(market_train_df['assetCode'] == 'TW.N') & (market_train_df['time'] < '2012-01-01')]
twSample['month'] = twSample['time'].dt.strftime('%Y') + '-' + twSample['time'].dt.strftime('%m')
twSample['month'].iplot(kind='hist',bins=24)


# Given those Jan 2010 records are a bit isolated from the rest of the data for this particular asset, I opted to just remove them from the `market_train_df`.

# In[ ]:

# simply dropping Towers Watson data from Jan 2010
# as these observations are isolated from the rest of Towers Watson's data which doesn't start until June 2010
towersWatsonDataToDrop = market_train_df[(market_train_df['assetCode'] == 'TW.N') 
                                         & (market_train_df['time'] < '2010-02-01')]
towersWatsonIndicesToDrop = list(towersWatsonDataToDrop.index)
market_train_df.drop(towersWatsonIndicesToDrop,inplace=True)


# ### Bad open prices of 0.01 and 999.99
# ---
# Returning to the summary statistics table, the `returnsOpenPrevRaw1` column has a ridiculously high maximum value; implying a 920,900% day-over-day increase.  This is undoubtedly an error.

# In[ ]:

market_train_df.describe().round(3)


# Looking at records with a greater than 1000% `returnsOpenPrevRaw1` metric...

# In[ ]:

market_train_df[market_train_df['returnsOpenPrevRaw1'] > 10]


# There are 22 records with a `returnsOpenPrevRaw1` value over 1000%.  All of these records are from 2007 or 2008, and we can see a large batch of them from 2007-03-23.  Some of the records have obvious issues.  For example, three of them have an `open` price of 999.99.  After looking into the data around these records, several have a inaccurate `open` price of 0.01 the day prior.  Below is a quick look at a few examples of those bad 0.01 prices...

# In[ ]:

market_train_df[((market_train_df['assetCode'] == 'PBRa.N') & (market_train_df['time'] >= '2007-05-02') & (market_train_df['time'] < '2007-05-05'))
                | ((market_train_df['assetCode'] == 'EXH.N') & (market_train_df['time'] >= '2007-08-22') & (market_train_df['time'] < '2007-08-25'))
                | ((market_train_df['assetCode'] == 'ATPG.O') & (market_train_df['time'] >= '2007-10-29') & (market_train_df['time'] < '2007-11-01'))
                | ((market_train_df['assetCode'] == 'TEO.N') & (market_train_df['time'] >= '2007-02-26') & (market_train_df['time'] < '2007-03-01'))
               ].sort_values('assetCode')


# Again, at the moment I'm probably not going to leverage the market data from 2007 and 2008, so I'm not going to worry about fixing errors during that time frame.  But if you are going to use market data from these years, you should probably look into fixing errors like those seen above.
# 
# Looking again at the summary statistics, but this time filtering for years 2009 and after, we can see a much nicer looking set of statistics with fewer extreme/outlier values across the board...

# In[ ]:

market_train_df[market_train_df['time'].dt.year >= 2009].describe().round(3)


# ### Stock Splits
# ---
# While not data errors, if you want to use the `open` and `close` columns to generate new features, you should be aware of potential issue arising from stock splits.
# 
# Using Apple's stock as an example, below is a graph of the `open` price for the `assetCode`='AAPL.O' in the `market_train_df`...

# In[ ]:

apple = market_train_df[market_train_df['assetCode'] == 'AAPL.O']
apple.iplot(kind='line',x='date',y='open')


# From the graph it looks like Apple's stock plummeted in June 2014, but acutally the stock just split.  There is commentary about this event in the news data...

# In[ ]:

appleNews = news_train_df[news_train_df['assetName'] == 'Apple Inc']
list(appleNews[(appleNews['headline'].str.contains('stock split')) & (appleNews['relevance'] >= 0.6)].head()['headline'])


# Apple's 7-to-1 stock split occurred on 2014-06-09.  It's worth noting that while the `open` and `close` columns don't take into account the stock split, the returns columns do (see excerpt below).

# In[ ]:

apple[(apple['time'] > '2014-06-01') & (apple['time'] < '2014-06-16')]


# Since it appears there's no issues with the return columns, if you're not planning on calculating any new features using the `open` and `close` columns, then you shouldn't need to worry about stock splits.
# 
# However, I am interested in trying to create some new features from these columns (e.g. moving averages), so I'm looking at adjusting historical stock prices like Apple's to account for splits.  Below is a graph showing an adjusted view of Apple's `open` price along with some handy moving averages.

# In[ ]:

apple['adjOpen'] = np.where(apple['time'] < '2014-06-09',apple['open']/7.0,apple['open'])
apple['MA10'] = apple['adjOpen'].rolling(window=10).mean()
apple['MA50'] = apple['adjOpen'].rolling(window=50).mean()
apple['MA200'] = apple['adjOpen'].rolling(window=200).mean()
apple.iplot(kind='line',x='date',y=['adjOpen','MA10','MA50','MA200'])


# In[ ]:



