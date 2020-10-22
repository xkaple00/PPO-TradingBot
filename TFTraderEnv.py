import process_data
import pandas as pd
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import deque, OrderedDict
import tensorflow as tf
import os
import shutil
import pickle
import datetime
from stable_baselines.common.env_checker import check_env

# position constant
LONG_0 = 0
LONG_1 = 1
LONG_2 = 2

SHORT_3 = 3
SHORT_4 = 4
SHORT_5 = 5

FLAT = 6

# action constant
OPEN_LONG_0 = 0
OPEN_LONG_1 = 1
OPEN_LONG_2 = 2

OPEN_SHORT_3 = 3
OPEN_SHORT_4 = 4
OPEN_SHORT_5 = 5

CLOSE_LONG_0 = 6
CLOSE_LONG_1 = 7
CLOSE_LONG_2 = 8

CLOSE_SHORT_3 = 9
CLOSE_SHORT_4 = 10
CLOSE_SHORT_5 = 11

HOLD = 12

class OhlcvEnv(gym.Env):

    def __init__(self, window_size, path, train=True, show_trade=True):
        self.train= train
        self.show_trade = show_trade
        self.path = path
        self.positions = ["LONG_0","LONG_1","LONG_2", "SHORT_0","SHORT_1","SHORT_2", "FLAT"]
        self.position_array = np.zeros(7)
        # self.position_array = np.zeros((self.lookback_window, 7))

        self.order = ["OPEN_LONG_0","OPEN_LONG_1","OPEN_LONG_2", "OPEN_SHORT_3","OPEN_SHORT_4","OPEN_SHORT_5", \
                     "CLOSE_LONG_0","CLOSE_LONG_1","CLOSE_LONG_2","CLOSE_SHORT_3","CLOSE_SHORT_4","CLOSE_SHORT_5", "HOLD"]
        self.fee = 0.0001 # for Crypto 0.001
        # self.seed()
        self.file_list = []
        # load_csv
        self.load_from_csv()

        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, self.n_features+14) # same with and without volume

        self.info_frequency = int(1e4)

        self.current_tick = 0
        # self.balance = 0
        self.available_balance = 0
        self.equity = self.old_equity = 0


        #Lists for info

        self.available_balance_list = deque(maxlen=self.info_frequency)
        self.reward_list = deque(maxlen=self.info_frequency)
        self.temp_reward_list = deque(maxlen=self.info_frequency)
        self.closingPrice_ask_list = deque(maxlen=self.info_frequency)
        self.closingPrice_bid_list = deque(maxlen=self.info_frequency)
        self.temp_reward_sum_list = deque(maxlen=self.info_frequency)
        self.equity_list = deque(maxlen=self.info_frequency)
        self.equity_reward_list = deque(maxlen=self.info_frequency)
        self.temp_reward_coef_list = deque(maxlen=self.info_frequency)

        # defines action space
        self.action_space = spaces.Discrete(len(self.order))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        self.reward = 0
        self.episode = 0
        # self.reward_list = []

        self.lookback_window = 1

        self.done = False
        self.temp_reward = np.zeros(7)

        #start training from Monday (whole week)
        while self.linux_timestamp[self.current_tick] - self.linux_timestamp[self.current_tick-1] < 1e5: # surpass the weekend
            self.current_tick += 1

        print("Start time linux:", self.linux_timestamp[self.current_tick])
        print("Start time human:", datetime.datetime.fromtimestamp(self.linux_timestamp[self.current_tick]))
        
        self.week_start_linux = self.linux_timestamp[self.current_tick]        
        self.state_queue = deque(maxlen=self.window_size)
        self.state = self.preheat_queue()

        self.n_hold_index = 0. # initial value zero
        # self.max_episode_length = 512
        self.reward_coef = 1e-2 # for BTC 0.01 # 100 CHFUSD # 10 BRENT


        self.closed_long_0 = 0
        self.closed_long_1 = 0
        self.closed_long_2 = 0
        self.closed_short_3 = 0
        self.closed_short_4 = 0
        self.closed_short_5 = 0

        self.total_profit_long = 0
        self.total_profit_short = 0

    # # SECOND DATA + VOLUME 
    def load_from_csv(self):
        if(len(self.file_list) == 0):
            self.file_list = [x.name for x in Path(self.path).iterdir() if x.is_file()]
            self.file_list.sort()
        self.rand_episode = self.file_list.pop() # Ok if only one file in directory
        raw_df= pd.read_csv(self.path + self.rand_episode)
        extractor = process_data.FeatureExtractor(raw_df)
        self.df = extractor.add_bar_features() 


        self.df.dropna(inplace=True) # drops Nan rows
        self.closingPrices_ask = self.df["Close"].values
        self.closingPrices_bid = self.df["Close"].values
        # self.original_timestamp = self.df["Timestamp"].values
        self.linux_timestamp = self.df["Timestamp"].values

        feature_list = ["Timestamp", "Close_stationary", "Volume"]

        self.df = self.df[feature_list].values # leave only stationary data


    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_time(self, actual_timestamp, week_start_linux):
        return (actual_timestamp - week_start_linux) / 432000 #432k seconds in 5 business days

    ## Original normalization
    def normalize_frame(self, frame):
        offline_scaler = StandardScaler()

        actual_timestamp = frame[..., :1]
        timestamp = self.normalize_time(actual_timestamp, self.week_start_linux)

        agent_state = frame[..., 1:]
        
        temp = np.concatenate((timestamp,  agent_state), axis=1)

        return temp

    #Original step
    def step(self, action):
        s, r, d, i = self._step(action)
        self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue))), r, d, i

    def _step(self, action):
        
        if self.done:
            self.reset()


        self.action = HOLD
        self.reward = 0

        # up to 6 positions can be opened per trade
        # valid action sequence per position would be
        # LONG : OPEN_LONG - HOLD - HOLD - CLOSE_LONG
        # SHORT : OPEN_SHORT - HOLD - HOLD - CLOSE_SHORT
        # invalid action sequence is just considered hold
        # (e.g.) "OPEN_LONG - OPEN_LONG" would be considred "OPEN_LONG - HOLD"
        if action == OPEN_LONG_0: 
            if self.position_array[0] != 1: # if previous position was not LONG_0
                self.position_array[0] = 1 # update position to LONG_0
                self.action = OPEN_LONG_0 # record action as buy
                self.entry_price_long_0 = self.closingPrice_ask # maintain entry price
                # self.reward = 0.1
        
        elif action == OPEN_LONG_1: # vice versa for short trade
            if self.position_array[1] != 1:
                self.position_array[1] = 1
                self.action = OPEN_LONG_1
                self.entry_price_long_1 = self.closingPrice_ask
                # self.reward = 0.1

        elif action == OPEN_LONG_2: # vice versa for short trade
            if self.position_array[2] != 1:
                self.position_array[2] = 1
                self.action = OPEN_LONG_2
                self.entry_price_long_2 = self.closingPrice_ask
                # self.reward = 0.1

        elif action == OPEN_SHORT_3: # vice versa for short trade
            if self.position_array[3] != 1:
                self.position_array[3] = 1
                self.action = OPEN_SHORT_3
                self.entry_price_short_3 = self.closingPrice_bid
                # self.reward = 0.1

        elif action == OPEN_SHORT_4: # vice versa for short trade
            if self.position_array[4] != 1:
                self.position_array[4] = 1
                self.action = OPEN_SHORT_4
                self.entry_price_short_4 = self.closingPrice_bid
                # self.reward = 0.1

        elif action == OPEN_SHORT_5: # vice versa for short trade
            if self.position_array[5] != 1:
                self.position_array[5] = 1
                self.action = OPEN_SHORT_5
                self.entry_price_short_5 = self.closingPrice_bid
                # self.reward = 0.1

        elif action == CLOSE_LONG_0: 
            if self.position_array[0] == 1: # if previous position was not LONG_0
                self.position_array[0] = 0 # update position to LONG_0
                self.action = CLOSE_LONG_0 # record action as buy
                self.exit_price_long_0 = self.closingPrice_bid
                self.reward = ((self.exit_price_long_0 - self.entry_price_long_0) - self.exit_price_long_0 * self.fee) * self.reward_coef # calculate reward
                # self.temp_reward[0] = 0
                self.closed_long_0 += 1
                self.total_profit_long += self.reward

        elif action == CLOSE_LONG_1: 
            if self.position_array[1] == 1: # if previous position was not LONG_0
                self.position_array[1] = 0 # update position to LONG_0
                self.action = CLOSE_LONG_1 # record action as buy
                self.exit_price_long_1 = self.closingPrice_bid
                self.reward = ((self.exit_price_long_1 - self.entry_price_long_1) - self.exit_price_long_1 * self.fee) * self.reward_coef # calculate reward
                # self.temp_reward[1] = 0
                self.closed_long_1 += 1
                self.total_profit_long += self.reward

        elif action == CLOSE_LONG_2: 
            if self.position_array[2] == 1: # if previous position was not LONG_0
                self.position_array[2] = 0 # update position to LONG_0
                self.action = CLOSE_LONG_2 # record action as buy
                self.exit_price_long_2 = self.closingPrice_bid
                self.reward = ((self.exit_price_long_2 - self.entry_price_long_2) - self.exit_price_long_2 * self.fee) * self.reward_coef # calculate reward
                # self.temp_reward[2] = 0
                self.closed_long_2 += 1
                self.total_profit_long += self.reward

        elif action == CLOSE_SHORT_3: 
            if self.position_array[3] == 1: # if previous position was not LONG_0
                self.position_array[3] = 0 # update position to LONG_0
                self.action = CLOSE_SHORT_3 # record action as buy
                self.exit_price_short_3 = self.closingPrice_ask
                self.reward = ((self.entry_price_short_3 - self.exit_price_short_3) - self.exit_price_short_3 * self.fee) * self.reward_coef # calculate reward
                # self.temp_reward[3] = 0
                self.closed_short_3 += 1
                self.total_profit_short += self.reward

        elif action == CLOSE_SHORT_4: 
            if self.position_array[4] == 1: # if previous position was not LONG_0
                self.position_array[4] = 0 # update position to LONG_0
                self.action = CLOSE_SHORT_4 # record action as buy
                self.exit_price_short_4 = self.closingPrice_ask
                self.reward = ((self.entry_price_short_4 - self.exit_price_short_4) - self.exit_price_short_4 * self.fee) * self.reward_coef # calculate reward
                # self.temp_reward[4] = 0
                self.closed_short_4 += 1
                self.total_profit_short += self.reward

        elif action == CLOSE_SHORT_5: 
            if self.position_array[5] == 1: # if previous position was not LONG_0
                self.position_array[5] = 0 # update position to LONG_0
                self.action = CLOSE_SHORT_5 # record action as buy
                self.exit_price_short_5 = self.closingPrice_ask
                self.reward = ((self.entry_price_short_5 - self.exit_price_short_5) - self.exit_price_short_5 * self.fee) * self.reward_coef # calculate reward
                # self.temp_reward[5] = 0
                self.closed_short_5 += 1
                self.total_profit_short += self.reward


        
        self.reward_list.append(self.reward)
        self.available_balance += self.reward

        if(self.position_array[0] == 1.):
            self.temp_reward[0] = ((self.closingPrice_bid - self.entry_price_long_0) - self.closingPrice_bid * self.fee) * self.reward_coef
        else:
            self.temp_reward[0] = 0
        
        if(self.position_array[1] == 1.):
            self.temp_reward[1] = ((self.closingPrice_bid - self.entry_price_long_1) - self.closingPrice_bid * self.fee) * self.reward_coef
        else:
            self.temp_reward[1] = 0

        if(self.position_array[2] == 1.):
            self.temp_reward[2] = ((self.closingPrice_bid - self.entry_price_long_2) - self.closingPrice_bid * self.fee) * self.reward_coef
        else:
            self.temp_reward[2] = 0

        if(self.position_array[3] == 1.):
            self.temp_reward[3] = ((self.entry_price_short_3 - self.closingPrice_ask) - self.closingPrice_ask * self.fee) * self.reward_coef
        else:
            self.temp_reward[3] = 0

        if(self.position_array[4] == 1.):
            self.temp_reward[4]= ((self.entry_price_short_4 - self.closingPrice_ask) - self.closingPrice_ask * self.fee) * self.reward_coef
        else:
            self.temp_reward[4] = 0

        if(self.position_array[5] == 1.):
            self.temp_reward[5] = ((self.entry_price_short_5 - self.closingPrice_ask) - self.closingPrice_ask * self.fee) * self.reward_coef
        else:
            self.temp_reward[5] = 0
        
        self.equity = self.available_balance + np.sum(self.temp_reward)

        self.time_from_start = self.linux_timestamp[self.current_tick] - self.week_start_linux

        self.equity_reward = self.equity 

        self.old_equity = self.equity


        if self.time_from_start >= 432000 - 512: # stop episode 1024 sec before end of week

            self.reward += np.sum(self.temp_reward)
            
            self.done = True
            self.position_array = np.zeros(7)
            self.temp_reward = np.zeros(7)

            print("Episode {} finished".format(self.episode))
            print("Week from {} to {}".format(datetime.datetime.fromtimestamp(int(self.week_start_linux)).strftime('%Y-%m-%d %H:%M:%S'),
            datetime.datetime.fromtimestamp(int(self.linux_timestamp[self.current_tick])).strftime('%Y-%m-%d %H:%M:%S')))
            print("Equity: {}".format(self.equity))


        self.available_balance_list.append(self.available_balance)
        self.reward_list.append(self.reward)
        self.temp_reward_list.append(self.temp_reward)
        self.closingPrice_ask_list.append(self.closingPrices_ask[self.current_tick])
        self.closingPrice_bid_list.append(self.closingPrices_bid[self.current_tick])
        self.temp_reward_sum_list.append(np.sum(self.temp_reward))
        self.equity_list.append(self.equity)
        self.equity_reward_list.append(self.equity_reward)
        # self.temp_reward_coef_list.append(self.temp_reward_coef)

        info = {'available_balance':self.available_balance_list,
            "reward":self.reward_list,
            "temp_reward":self.temp_reward_list,
            "temp_reward_sum": self.temp_reward_sum_list,

            "price_ask":self.closingPrice_ask_list,
            "price_bid":self.closingPrice_bid_list,
            "equity":self.equity_list,
            "equity_reward":self.equity_reward_list,
            }

        # print('print_info',  info)
        
        if(self.show_trade and self.current_tick % self.info_frequency == 0) or self.done:
        # if self.done:
            print("Tick: {0}/ Available balance : {1}".format(self.current_tick, self.available_balance))

            print("Action {0}".format(self.order[action-1]))
            print("temp_reward: {}".format(self.temp_reward))
            print("temp_reward_sum: {}".format(np.sum(self.temp_reward)))
            print('reward', self.reward)
            print("equity", self.equity)
            print("equity_reward", self.equity_reward)
            print('position_array', self.position_array)
            print("linux_timestamp_end", self.normalize_frame(np.concatenate(tuple(self.state_queue))))
            print(
            "closed_long_0:", self.closed_long_0,
            "closed_long_1:", self.closed_long_1,
            "closed_long_2:", self.closed_long_2,
            "closed_short_3:", self.closed_short_3,
            "closed_short_4:", self.closed_short_4,
            "closed_short_5:", self.closed_short_5)

            print("total_profit_long", self.total_profit_long,
            "total_profit_short", self.total_profit_short)
            print("\n\n")

            with open('./info/ppo_{}.npy'.format(self.current_tick), 'wb') as f:
                pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
        
        self.current_tick += 1
        self.state = self.updateState()

        return self.state, self.equity_reward, self.done, info

    def reset(self):

        self.done = False
        
        # self.ticks_in_force_long = 0
        # self.ticks_in_force_short = 0
        self.temp_reward = np.zeros(7)

        self.closed_long_0 = 0
        self.closed_long_1 = 0
        self.closed_long_2 = 0
        self.closed_short_3 = 0
        self.closed_short_4 = 0
        self.closed_short_5 = 0

        self.total_profit_long = 0
        self.total_profit_short = 0

        if self.episode >= 8:
            while self.linux_timestamp[self.current_tick] - self.linux_timestamp[self.current_tick-1] < 1e5: # surpass the weekend
                self.current_tick += 1
            self.state = self.preheat_queue()
            self.week_start_linux = self.linux_timestamp[self.current_tick]    

        # self.reward_list = []

        # clear internal variables
        self.available_balance = 0. # initial balance, u can change it to whatever u like 
        self.equity = self.old_equity = 0
        # self.portfolio = float(self.balance) # (coin * current_price + current_balance) == portfolio
        self.closingPrice_bid = self.closingPrices_bid[self.current_tick]
        self.closingPrice_ask = self.closingPrices_ask[self.current_tick]

        self.position_array = np.zeros(7)

        self.episode += 1

        print("start episode ... {0} at {1} tick" .format(self.episode, self.current_tick))

        return self.state


    def preheat_queue(self):
        while(len(self.state_queue) < self.window_size):

            rand_action = 13
            s, r, d, i= self._step(rand_action)

            self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue)))


    def one_hot_encode(self, x, n_classes):
        return np.eye(n_classes)[x]

    def updateState(self):
        self.closingPrice_bid = float(self.closingPrices_bid[self.current_tick])
        self.closingPrice_ask = float(self.closingPrices_ask[self.current_tick])
        
        profit = self.temp_reward 

        state = np.concatenate((self.df[self.current_tick], self.position_array, profit))

        return state.reshape(1,-1)

    def close(self):
        pass


# Environment check
# PATH_TRAIN = "../data/XAGUSD/"
PATH_TRAIN = "../data/BRENT_linux_timestamp/"

# PATH_TRAIN = "./data/train/"
# PATH_TEST = "./data/test/"
TIMESTEP = 512 # window size 30


if __name__ == "__main__":
    env = OhlcvEnv(window_size=TIMESTEP, path=PATH_TRAIN, train=True)
    print('Check environment', check_env(env))
    obs = env.reset()
    env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    OPEN_LONG_0 = 0
    # Hardcoded best agent: always OPEN_LONG_0 
    n_steps = 2
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(OPEN_LONG_0)
        print('obs=', obs, 'obs_shape=', obs.shape, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break