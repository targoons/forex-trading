
# coding: utf-8

# ## Import Libraries

# In[57]:


import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pickle
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split


# ## PriceChangeMovement Model

# In[80]:


class PriceChangeMovement_DataPreprocessing:
    def __init__(self, start_datetime,
                    end_datetime):
        self.START_DT = start_datetime
        self.END_DT = end_datetime
        self.hourly_currency_data = None
        self.Google_Trends_data = None
        self.volume_data = None
        self.price_model_data = None
    
    def assign_class(self,x, pct):
        if x > pct:
            return 1
        elif x < -1 * pct:
            return -1
        else:
            return 0
    
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
    
    def GoogleTrends_Preprocesssing(self, googletrends_data_path, googletrends_filename):
        google_trends = pd.read_csv(os.path.join(googletrends_data_path, googletrends_filename), index_col = 0)
        google_trends.index = pd.to_datetime(google_trends.index)
        
        self.Google_Trends_data = google_trends
        
    def BuySellVolume_Preprocessing(self, buysell_vol_data_path, buysell_vol_filename):
        buysell_vol = pd.read_csv(os.path.join(buysell_vol_data_path, buysell_vol_filename), index_col = 0)
        buysell_vol.index = pd.to_datetime(buysell_vol.index)
        
        self.volume_data = buysell_vol
    
class PriceChangeMovement_DataAggregation:
    def __init__(self, currency_data, Google_Trends_data=None, event_model_data=None, volume_data=None):
        self.currency_data = currency_data
        self.Google_Trends_data = Google_Trends_data
        self.event_model_data = event_model_data
        self.volume_data = volume_data
        self.price_model_data = None
    
    def Get_PriceModelData(self):
        hourly_data = self.currency_data
        cols = ['RETURN'] 

        if self.Google_Trends_data is not None:
            hourly_data = pd.merge(hourly_data, self.Google_Trends_data, left_index=True, right_index=True, how='left')
            cols = cols + list(self.Google_Trends_data)
            
        if self.event_model_data is not None:
            hourly_data = pd.merge(hourly_data, self.event_model_data, left_index=True, right_index=True, how='left')
            col_to_remove = list(self.event_model_data)
            col_to_remove.remove('Text_Label')
            cols = cols + ['Text_Label']
        if self.volume_data is not None:
            hourly_data = pd.merge(hourly_data, self.volume_data, left_index=True, right_index=True, how='left')
            cols = cols + ['ActualVolume_Buy', 'ActualVolume_Sell']

        hourly_data = hourly_data.loc[:,~hourly_data.columns.duplicated()]
        hourly_data.fillna(0, inplace=True)
        
        for i in range(1, 25, 1):
            for col in cols:
                hourly_data[col + '_' + str(i)] = hourly_data[col].shift(i)
                #col_name.append(col + '_' + str(i))

        hourly_data.fillna(0, inplace=True)
        hourly_data.drop(columns=cols + ['TICKER','DATE', 'TIME', 'CLOSE', 'DATE_TIME', 'RETURN_PCT'] + list(self.event_model_data), inplace=True)
        self.price_model_data = hourly_data
        
class PriceChangeMovement_ModelTraining:
    def __init__(self, price_model_data):
        self.price_model_data = price_model_data
        self.X = self.price_model_data.drop(columns=['LABEL'])
        self.Y = self.price_model_data["LABEL"]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predicted_dataframe = None
        
    def train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, 
                                                            test_size=0.15, shuffle=False)
        #convert -1 label to 2
        self.Y[self.Y==-1] = 2 
        self.y_train[self.y_train==-1] = 2
        self.y_test[self.y_test==-1] = 2
        
    def training(self, param, model_file_name='../Model/PriceChangeMovement.model'):
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        d = xgb.DMatrix(self.X, label=self.Y)
        
        num_round = 2000
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=20)
        bst.save_model(model_file_name)
        self.model = bst
        
class PriceChangeMovement_Prediction:
    def __init__(self, model, X):
        self.model = model
        dtest = xgb.DMatrix(X)
        self.y_predict = model.predict(dtest, ntree_limit=50)
        y_train_pred = np.asarray([np.argmax(line) for line in self.y_predict])
        X['Predicted_Label'] = y_train_pred
        self.predicted_dataframe = X



if __name__ == "__main__": 

    START_DT = '2018-10-01 00:00:00'
    END_DT = '2019-03-31 23:00:00'
    pct = 0.01

    ## DATA PATH
    ### Currency
    currency_data_path = "/project/msca/projects/ForexPrediction/data/Currencies/"
    currency_data_filename = "EURUSD.txt"

    ### Google Trends
    google_trends_data_path = "~"
    google_trends_filename = "google_trends_aggregated.csv"

    ### Currency Buy/Sell Volumes
    buysell_vol_data_path = '~'
    buysell_vol_filename = 'EURUSD_aggregated_orders_with_features.csv'

    ### Event Detection Model Result
    event_model_data_filename = 'text_label_prediction.pkl'

    ### Keywords
    keyword_path = '/home/targoons/'
    keyword_filename = 'keyword.txt'

    ## Model Training Parameters
    best_param = {'alpha': 0.75,
     'colsample_bytree': 0.2,
     'eval_metric': ['merror'],
     'feature_selector': 'thrifty',
     'gamma': 1,
     'lambda': 5,
     'learning_rate': 0.01,
     'max_depth': 3,
     'nthread': 4,
     'objective': 'multi:softprob',
     'num_class' : 3,
     'scale_pos_weight': 1,
     'subsample': 0.2}


    # In[81]:


    ## Data Preprocessing
    data = PriceChangeMovement_DataPreprocessing(START_DT,END_DT)
    data.Currency_Preprocessing(currency_data_path, currency_data_filename, pct)
    print(data.hourly_currency_data.head(2))
    data.GoogleTrends_Preprocesssing(google_trends_data_path, google_trends_filename)
    print(data.Google_Trends_data.head(2))
    data.BuySellVolume_Preprocessing(buysell_vol_data_path, buysell_vol_filename)
    print(data.volume_data.head(2))
    event_model_data = pd.read_pickle(event_model_data_filename)
    print(event_model_data.head(2))
    ## Data Aggregation
    data_agg = PriceChangeMovement_DataAggregation(data.hourly_currency_data,
                                                     Google_Trends_data=data.Google_Trends_data, 
                                                     event_model_data=event_model_data, 
                                                     volume_data=data.volume_data
                                                      )
    data_agg.Get_PriceModelData()
    print(data_agg.price_model_data.head())

    ## Model Training
    model_train = PriceChangeMovement_ModelTraining(data_agg.price_model_data)
    model_train.train_test()
    model_train.training(param)

    ## Model Prediction
    prediction = PriceChangeMovement_Prediction(model_train.model, model_train.X)
    print(prediction.y_predict)


    from sklearn.metrics import accuracy_score
    accuracy_score(model_train.Y, prediction.predicted_dataframe['Predicted_Label'])

