import pandas as pd
import datetime

pd.options.mode.chained_assignment = None

class TradeVolumeAggregation:
    def __init__(self, file_path, filename, ST_DT, END_DT,
                 saved_file_path = '../Data/',
                 saved_filename='_aggregated_orders_with_features.csv'):
        self.trade_df = pd.read_csv(file_path + filename, sep=';')
        self.start_date = ST_DT
        self.end_date = END_DT
        self.aggregated_trade_vol = None
        self.saved_filename = saved_file_path + filename[:6] + saved_filename

    def rounder(self, t):
        return t.replace(second=0, microsecond=0, minute=0) + datetime.timedelta(hours=1)
    
    def Aggregate(self):
        dat_filtered = self.trade_df[~(self.trade_df['CloseTime'].str.contains('1/1/0001 12:00:00 AM'))]
        dat_filtered['OpenTime'] = pd.to_datetime(dat_filtered['OpenTime'])
        dat_filtered['CloseTime'] = pd.to_datetime(dat_filtered['CloseTime'])
        dat_filtered['OpenTime'] = dat_filtered.OpenTime.apply(lambda x: x + datetime.timedelta(hours=-8))
        dat_filtered['CloseTime'] = dat_filtered.CloseTime.apply(lambda x: x + datetime.timedelta(hours=-8))

        dat_filtered['ActualVolume'] = dat_filtered.Volume * 100000
        dat_filtered.set_index(dat_filtered.Order, inplace=True)
        dat_filtered.sort_index(inplace=True)
        
        dat_filtered[['buyingEUR', 'sellingEUR']] = pd.get_dummies(dat_filtered.Command)
        time_index = pd.date_range(start=self.start_date, end=self.end_date, freq='60min')
        hourly_df = pd.DataFrame(index=time_index)
        
        dat_filtered['OpenTime'] = dat_filtered.OpenTime.apply(self.rounder)
        dat_filtered['CloseTime'] = dat_filtered.CloseTime.apply(self.rounder)
        
        buying_eur_series = dat_filtered[dat_filtered['buyingEUR'] == 1].groupby('OpenTime').sum()['ActualVolume']
        selling_eur_series = dat_filtered[dat_filtered['sellingEUR'] == 1].groupby('OpenTime').sum()['ActualVolume']
        self.aggregated_trade_vol = selling_eur_series.to_frame().join(buying_eur_series, how='outer', rsuffix='_Buy', lsuffix='_Sell')
        self.aggregated_trade_vol.to_csv(self.saved_filename)
    

if __name__ == "__main__": 
    ST_DT = '2018-07-27 00:00:00'
    END_DT = '2019-08-02 12:00:00'
    
    Trade = TradeVolumeAggregation('../Data/TradeOrder/', 'EURUSD Orders.csv', ST_DT, END_DT)
    Trade.Aggregate()


