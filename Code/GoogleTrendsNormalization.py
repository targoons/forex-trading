import pandas as pd
from os import listdir
from os.path import isfile, join

class GoogleTrendsNormalization:
    def __init__(self, hourly_path, daily_path, normalized_file_name='google_trends_aggregated.csv'):
        self.hourly_path = hourly_path
        self.daily_path = daily_path
        self.normalized_file_name = normalized_file_name
        self.normalized_score = None
        
    def round_less_than_one(self, d):
        if d == '<1':
            return 0
        else:
            return(int(d))
    
    def Normalize(self):
        scores = []
        keywords = listdir(self.daily_path)
        file_name_hourly = [self.hourly_path + f for f in listdir(self.hourly_path) if isfile(join(self.hourly_path, f))]
        file_name_daily = [self.daily_path + f + '/multitimeline.csv' for f in listdir(self.daily_path)]

        
        for i, kw in enumerate(keywords):
            file_name_hourly = self.hourly_path + kw + '.csv'
            file_name_daily = self.daily_path + kw + '/multiTimeline.csv'

            hourly_df = pd.read_csv(file_name_hourly, index_col='date')
            hourly_df.drop(columns=['isPartial'], inplace=True)
            daily_df = pd.read_csv(file_name_daily, header=1, index_col='Day')

            hourly_df.index = pd.to_datetime(hourly_df.index)
            daily_df.index = pd.to_datetime(daily_df.index)

            daily_df = daily_df.resample('1H').pad()
            hourly_df[kw] = hourly_df[kw].apply(lambda x : self.round_less_than_one(x))
            daily_df.iloc[:,0] = daily_df.iloc[:,0].map(lambda x : self.round_less_than_one(x))
            combined_df = daily_df.join(hourly_df, how='left')
            combined_df.fillna(0, inplace=True)
            Nan_Col = combined_df[combined_df.isna().any(axis=1)]        
            combined_df[kw + '_score'] = combined_df[combined_df.columns[0]] * combined_df[combined_df.columns[1]]
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            scores.append(combined_df[kw + '_score'])
        score_df = pd.concat(scores, axis=1)
        self.normalized_score = score_df
        score_df.to_csv(self.normalized_file_name)
        print('File successfully saved at ' + self.normalized_file_name)

if __name__ == '__main__':
    hourly_path = 'GoogleTrends/Hourly Data/'
    daily_path = 'GoogleTrends/Daily Data/'
    GoogleTrends = GoogleTrendsNormalization(hourly_path, daily_path)
    GoogleTrends.Normalize()



