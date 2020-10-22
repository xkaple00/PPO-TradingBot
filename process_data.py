import pandas as pd
# import talib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime



class FeatureExtractor:


      ## SECOND DATA + VOLUME 
      def __init__(self, df):
         self.df = df
         self.timestamp = df["Timestamp"]
         self.close = df["Close"].astype('float')  #!!!
         self.volume = df["Volume"].astype('float')  #!!!

      def add_bar_features(self):
         self.df["Timestamp"] = self.unix_timestamp(self.timestamp)
         self.df["Close"] = self.close
         self.df["Volume"] = self.volume / np.max(self.volume)
         self.df["Close_stationary"] = self.make_stationary(self.close, 12)
         # self.df["Ask_volume_stationary"] = self.make_stationary(self.ask_volume, 12)
         # self.df["Bid_volume_stationary"] = self.make_stationary(self.bid_volume, 12)

         # self.df["BidAskSpread"] = self.bid_ask_spread

         return self.df


      def make_stationary(self, df, rol_window):
         df_log = np.log(df)
         moving_avg = df_log.rolling(rol_window).mean()
         df_moving_avg_diff = df - moving_avg
         df_moving_avg_diff = df_moving_avg_diff.fillna(df_moving_avg_diff[rol_window-1])
         df_moving_avg_diff.dropna()
         df_log_diff = df_moving_avg_diff - df_moving_avg_diff.shift(1) 
         return df_log_diff

      def unix_timestamp(self, timestamp_column):
         linux_time = np.zeros((len(timestamp_column), 1))
         for i in range(len(timestamp_column)):
            date_time_obj = datetime.strptime(timestamp_column[i], '%Y%m%d %H:%M:%S')
            sec = date_time_obj.timestamp() 
            linux_time[i] = sec

         return linux_time
