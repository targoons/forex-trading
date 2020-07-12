
# coding: utf-8

# ## Import Libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pickle
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split


# ## Event Detection Model

# In[167]:


class EventDetection_DataPreprocessing:
    def __init__(self, start_datetime,
                    end_datetime):
        self.START_DT = start_datetime
        self.END_DT = end_datetime
        self.hourly_currency_data = None
        self.news_data = None
        self.aggregated_news_data = None
        
    def assign_class(self,x, pct):
        if x > pct:
            return 1
        elif x < -1 * pct:
            return -1
        else:
            return 0
        
    def rounder(self, t):
        return t.replace(second=0, microsecond=0, minute=0) + datetime.timedelta(hours=1)
    
    def get_keywords(self, keyword_path,keyword_filename):
        fo = open(os.path.join(keyword_path, keyword_filename), "r+")
        keywords = fo.readlines()
        keywords = [word.replace('\n', '') for word in keywords]
        keywords = [word.lower() for word in keywords]
        
        return keywords
    
    def Currency_Preprocessing(self, currency_data_path, currency_data_filename, pct):
        #Read Currency Data
        raw_currency_data = pd.read_csv(os.path.join(currency_data_path, currency_data_filename), header='infer', 
                                usecols = ["<TICKER>", "<DTYYYYMMDD>", "<TIME>", "<CLOSE>"])
        raw_currency_data.rename(columns={'<TICKER>':'TICKER','<DTYYYYMMDD>':'DATE', '<TIME>':'TIME', '<CLOSE>':'CLOSE'}, 
                         inplace=True)
        raw_currency_data["DATE_TIME"] = pd.to_datetime(raw_currency_data['DATE'].astype(str) +
                                                        " "+ raw_currency_data['TIME'].apply(lambda x: '{0:0>6}'.format(x)), format='%Y%m%d %H%M%S')
        
        #Filter cata starting from START_DT
        raw_currency_data_filtered = raw_currency_data[raw_currency_data["DATE_TIME"] >= self.START_DT]
        raw_currency_data_filtered.index = raw_currency_data_filtered["DATE_TIME"]
        
        #Resample currency data at 60-minute grain
        time_index = pd.date_range(start=self.START_DT, end=self.END_DT, freq='60min')
        df_temp = pd.DataFrame(index=time_index)
        hourly_df_indexed = df_temp.join(raw_currency_data_filtered, how = 'left')
        hourly_currency_data = hourly_df_indexed.interpolate()
        
        #Create data label (-1, 0, 1)
        hourly_currency_data['RETURN'] = hourly_currency_data['CLOSE'].shift(-1) - hourly_currency_data['CLOSE']
        hourly_currency_data['RETURN_PCT'] = hourly_currency_data['RETURN'] / hourly_currency_data['CLOSE'].shift(-1) * 100
        return_range = hourly_currency_data['RETURN'].max() - hourly_currency_data['RETURN'].min()
        hourly_currency_data['RETURN'] = hourly_currency_data['RETURN'] / return_range
        hourly_currency_data['LABEL'] = hourly_currency_data['RETURN']
        hourly_currency_data['LABEL'] = hourly_currency_data['LABEL'].apply(lambda x: self.assign_class(x, pct))

        self.hourly_currency_data = hourly_currency_data
    
    def Text_Preprocessing(self, text_data_path, keyword_path, keyword_filename):
        #Get the list of files
        files = os.listdir(text_data_path)
        files = [file for file in files if (file.endswith('.json')) & (file.startswith('merged'))]

        for i, file in enumerate(files):
            with open(text_data_path + file) as f:
                data = json.load(f)
            if not data:
                    continue
            if i==0:
                news_df = pd.DataFrame(data)
            else:
                news_df_temp = pd.DataFrame(data)
                news_df = pd.concat([news_df, news_df_temp])
        
        print("News file readed.")
        
        #Update news index
        news_df.reset_index(inplace=True)
        
        #Update timestamp
        from dateutil import parser
        from dateutil.parser import isoparse
        from dateutil.tz import UTC
        
        news_df['crawled'] = [isoparse(crawled_date).astimezone(UTC) for crawled_date in news_df['crawled']]
        news_df['published'] = [isoparse(crawled_date).astimezone(UTC) for crawled_date in news_df['published']]
        news_df['crawled_round'] = [self.rounder(t.tz_localize(None)) for t in news_df['crawled']]
        news_df['published_round'] = [self.rounder(t.tz_localize(None)) for t in news_df['published']]
        
        print("Timestamp update completed.")
        
        #Lowercase titles
        news_df['title'] = news_df['title'].apply(lambda x: x.lower())
        #news_df['text'] = news_df['text'].apply(lambda x: x.lower())
        
        keywords = self.get_keywords(keyword_path, keyword_filename)
        
        #Create columns of keywords
        for word in keywords:
            news_df[word] = 0
        self.news_data = news_df
        total_news_count = self.news_data.shape[0]
        for index, row in self.news_data.iterrows():
            for word in keywords:
                if word in row['title']:
                    news_df.at[index, word] = 1
            if index % 1000 == 0:
                print("Read " + str(index) + '/' + str(total_news_count) + " completed.")
        self.aggregated_news_data = self.news_data.groupby(['crawled_round']).sum()
        
class EventDetection_DataAggregation:
    def __init__(self, currency_data, Google_Trends_data=None, news_data=None, volume_data=None):
        self.currency_data = currency_data
        self.Google_Trends_data = Google_Trends_data
        self.news_data = news_data
        self.volume_data = volume_data
        self.event_model_data = None
    
    def get_keywords(self, keyword_path,keyword_filename):
        fo = open(os.path.join(keyword_path, keyword_filename), "r+")
        keywords = fo.readlines()
        keywords = [word.replace('\n', '') for word in keywords]
        keywords = [word.lower() for word in keywords]
        
        return keywords
    
    def Get_EventModelData(self, keyword_path, keyword_filename):
        keywords = self.get_keywords(keyword_path, keyword_filename)
        hourly_currency_news = pd.merge(self.currency_data, self.news_data, left_index=True, right_index=True, how='left')
        cols_keyword = hourly_currency_news[keywords].sum() < 0
        cols_keyword = list(cols_keyword[cols_keyword==False].index)
        cols = cols_keyword + ['TICKER',
         'DATE',
         'TIME',
         'CLOSE',
         'DATE_TIME',
         'RETURN',
         'LABEL']
        hourly_data = hourly_currency_news[cols]
        hourly_data = hourly_data.loc[:,~hourly_data.columns.duplicated()]
        hourly_data.fillna(0, inplace=True)
        col_name = []
        for i in range(1, 25, 1):
            for col in cols_keyword:
                hourly_data[col + '_' + str(i)] = hourly_data[col].shift(i)
                col_name.append(col + '_' + str(i))
        hourly_data.drop(columns=cols_keyword + ['RETURN', 'TICKER', 'DATE', 'TIME', 'DATE_TIME', 'CLOSE'], inplace=True)    
        hourly_data.fillna(0, inplace=True)
        self.event_model_data = hourly_data
        
class EventDetection_ModelTraining:
    def __init__(self, event_model_data):
        self.event_model_data = event_model_data
        self.X = self.event_model_data.drop(columns=['LABEL'])
        self.Y = self.event_model_data["LABEL"]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        
    def train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, 
                                                            test_size=0.15, shuffle=False)
        #convert -1 label to 1
        self.y_train[self.y_train==-1] = 1
        #convert -1 label to 1
        self.y_test[self.y_test==-1] = 1
    
    def training(self, param, model_file_name='../Model/EventDetection.model'):
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        d = xgb.DMatrix(self.X, label=self.Y)
        
        num_round = 500
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=10)
        bst.save_model(model_file_name)
        self.model = bst
        
class EventDetection_Prediction:
    def __init__(self, model, X):
        self.model = model
        dtest = xgb.DMatrix(X)
        self.y_predict = model.predict(dtest, ntree_limit=50)
        self.predicted_dataframe = X
        self.predicted_dataframe['Text_Label'] = self.y_predict


# In[170]:

if __name__ == "__main__": 
    #Arguments
    ## Parameters
    START_DT = '2017-09-01 00:00:00'
    END_DT = '2019-03-31 23:00:00'
    pct = 0.01

    ## DATA PATH
    ### Currency
    currency_data_path = "/project/msca/projects/ForexPrediction/data/Currencies/"
    currency_data_filename = "EURUSD.txt"

    ### Text
    text_data_path = '/project/msca/projects/ForexPrediction/data/WebHose/'

    ### Keywords
    keyword_path = '/home/targoons/'
    keyword_filename = 'keyword.txt'

    ## Model Training
    param = {'alpha': 0.01,
     'colsample_bytree': 0.5,
     'eval_metric': ['auc'],
     'feature_selector': 'thrifty',
     'gamma': 5,
     'lambda': 1,
     'learning_rate': 0.05,
     'max_depth': 3,
     'nthread': 8,
     'objective': 'binary:logistic',
     'subsample': 0.5}


    # In[181]:


    ## Data Preprocessing
    data = EventDetection_DataPreprocessing(START_DT,END_DT)
    data.Currency_Preprocessing(currency_data_path, currency_data_filename, pct)
    print(data.hourly_currency_data.head())
    #data.Text_Preprocessing(text_data_path, keyword_path, keyword_filename)
    data.news_data = pd.read_pickle('agg_news_score.pkl')
    print(data.news_data.head())

    ## Data Aggregation
    data_agg = EventDetection_DataAggregation(data.hourly_currency_data, news_data=data.news_data)
    data_agg.Get_EventModelData(keyword_path, keyword_filename)
    print(data_agg.event_model_data.head())

    ## Model Training
    model_train = EventDetection_ModelTraining(data_agg.event_model_data)
    model_train.train_test()
    model_train.training(param)

    ## Model Prediction
    prediction = EventDetection_Prediction(model_train.model, model_train.X)
    print(prediction.y_predict)
    print(prediction.predicted_dataframe)

