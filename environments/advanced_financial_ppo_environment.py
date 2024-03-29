import numpy as np
import pandas as pd
import random
import string
import matplotlib
matplotlib.use('Agg')  # Verwenden Sie den nicht-interaktiven Backend 'Agg'
import matplotlib.pyplot as plt

import tf_agents

from queue import Queue
from enum import Enum
import os

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.trajectories import time_step as ts
import datetime

class Action(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2
    INITIAL = 3

maximum_steps = 700
image_data_queue = Queue()   

class FinancialEnvironment(py_environment.PyEnvironment):
    def __init__(self, data, window_size, capture_mode = False):
        super(FinancialEnvironment, self).__init__()
        self.render_buf = []
        self.capture_mode = capture_mode

        self.df = data
        self.window_size = window_size

        self.current_step = 0
        self._start_offset = np.random.randint(window_size+1, len(data)-1-maximum_steps)
        self._render_offset = self._start_offset
        self.step_counter = 0
        self.current_step = self._start_offset
        self.steps_beyond_done = None

        self.data_window = []

        # Normalize values
        min_open = np.min(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'open'])
        max_open = np.max(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'open'])
        min_close = np.min(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'close'])
        max_close = np.max(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'close'])
        min_high = np.min(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'high'])
        max_high = np.max(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'high'])
        min_low = np.min(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'low'])
        max_low = np.max(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'low'])
        min_volume = np.min(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'volume'])
        max_volume = np.max(self.df.loc[self.current_step - 1 - self.window_size : self.current_step - 1, 'volume'])

        self.data_window = np.empty((self.window_size, 9), dtype = np.float32)
        for i in range(self.window_size):
            cur_idx = self.current_step - self.window_size + i
            candle = np.array([
                (self.df.loc[cur_idx, 'open'] - min_open) / (max_open - min_open),
                (self.df.loc[cur_idx, 'close'] - min_close) / (max_close - min_close),
                (self.df.loc[cur_idx, 'high'] - min_high) / (max_high - min_high),
                (self.df.loc[cur_idx, 'low'] - min_low) / (max_low - min_low),
                (self.df.loc[cur_idx, 'volume'] - min_volume) / (max_volume - min_volume) if self.df.loc[cur_idx, 'volume'] else 0,
                self.df.loc[cur_idx, 'time'],
                0,
                0,
                1
            ], dtype=np.float32)
            self.data_window[i,:] = candle

        # initialize variables to keep track of trades
        self.buy_price = 0
        self.sell_price = 0
        self.fee = 0.2/100
        self.last_action = Action.INITIAL
        self.steps_on_hold = 0
        self.cum_profit = 0

        self.buy_actions = 0
        self.sell_actions = 0
        self.hold_actions = 0

        # visualizsation
        self.buy_idxs = []
        self.sell_idxs = []
        self._render_buy_idxs = []
        self._render_sell_idxs = []

    def _next_observation(self):
        obs = np.empty(9, dtype=np.float32)
        cur_idx = self.current_step

        # Hot one encoding for self.last_action
        buy_flag = 1 if self.last_action == Action.BUY else 0
        sell_flag = 1 if self.last_action == Action.SELL else 0
        hold_flag = 1 if self.last_action == Action.HOLD else 0

        # Normalize values
        min_open = np.min(self.df.loc[cur_idx - 1 - self.window_size : cur_idx, 'open'])
        max_open = np.max(self.df.loc[cur_idx - 1 - self.window_size : cur_idx - 1, 'open'])
        min_close = np.min(self.df.loc[cur_idx - 1 - self.window_size : cur_idx - 1, 'close'])
        max_close = np.max(self.df.loc[cur_idx - 1 - self.window_size : cur_idx - 1, 'close'])
        min_high = np.min(self.df.loc[cur_idx - 1 - self.window_size : cur_idx - 1, 'high'])
        max_high = np.max(self.df.loc[cur_idx - 1 - self.window_size : cur_idx - 1, 'high'])
        min_low = np.min(self.df.loc[cur_idx - 1 - self.window_size : cur_idx - 1, 'low'])
        max_low = np.max(self.df.loc[cur_idx - 1 - self.window_size : cur_idx - 1, 'low'])
        min_volume = np.min(self.df.loc[cur_idx - 1 - self.window_size : cur_idx - 1, 'volume'])
        max_volume = np.max(self.df.loc[cur_idx - 1 - self.window_size : cur_idx - 1, 'volume'])

        obs = np.array([
            (self.df.loc[cur_idx, 'open'] - min_open) / (max_open - min_open),
            (self.df.loc[cur_idx, 'close'] - min_close) / (max_close - min_close),
            (self.df.loc[cur_idx, 'high'] - min_high) / (max_high - min_high),
            (self.df.loc[cur_idx, 'low'] - min_low) / (max_low - min_low),
            (self.df.loc[cur_idx, 'volume'] - min_volume) / (max_volume - min_volume) if self.df.loc[cur_idx, 'volume'] else 0,
            self.df.loc[cur_idx, 'time'],
            buy_flag,
            sell_flag,
            hold_flag
        ], dtype=np.float32)

        self.data_window = np.roll(self.data_window, -1, axis=0)
        self.data_window[-1] = obs
        return self.data_window

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "close"]

        if action == Action.BUY:
            self.buy_actions += 1
            self.buy_price = current_price
            self.last_action = Action.BUY
            self.steps_on_hold = 0
            self.buy_idxs.append(self.current_step)

        elif action == Action.SELL:
            self.sell_actions += 1
            self.sell_price = current_price
            self.last_action = Action.SELL
            self.steps_on_hold = 0
            self.sell_idxs.append(self.current_step)

        else: # action == Action.HOLD
            self.hold_actions += 1
            self.steps_on_hold += 1

    def _calculate_reward(self, action, price):
        current_price = self.df.loc[self.current_step, "close"]
        # print("action:", action, "last action:", self.last_action, "current price:", self.buy_price, "sell price:", self.sell_price)

        # Normal sell after buy
        if self.last_action == Action.BUY and action == Action.SELL:
            reward = ((current_price - self.buy_price) / self.buy_price) - self.fee
            self.cum_profit += ((current_price - self.buy_price) / self.buy_price) - self.fee

        # Normal buy after sell
        elif self.last_action == Action.SELL and action == Action.BUY:
            reward = ((self.sell_price - current_price) / self.sell_price) - self.fee

        # Hold after buy
        elif action == Action.HOLD and self.last_action == Action.BUY:
            reward = 0 #((current_price - self.buy_price) / self.buy_price) * 0.2
        
        # Hold after sell
        elif action == Action.HOLD and self.last_action == Action.SELL:
            reward = 0 #((self.sell_price - current_price) / self.sell_price) * 0.2

        # Initial buy
        elif action == Action.BUY and self.last_action == Action.INITIAL:
            reward = 1

        elif action == Action.SELL and self.last_action == Action.INITIAL:
            reward = -1

        elif action == Action.BUY and self.last_action == Action.BUY:
            reward = -1

        elif action == Action.SELL and self.last_action == Action.SELL:
            reward = -1

        else:
            reward = 0

        # Apply trade penalty
        # num_trades = self.buy_actions + self.sell_actions
        # trade_penalty = (num_trades ** 2) / 1000
        # reward -= trade_penalty

        # Apply holding penatly
        reward -= (self.steps_on_hold/maximum_steps) * 5.0

        return reward

    def _step(self, action):
        action = Action(action)

        current_price = self.df.loc[self.current_step, "close"]
        reward = self._calculate_reward(action, current_price)
        money_loss_condition = self.cum_profit <= -0.5
        #invalid_sequence_condition = (action == Action.BUY and self.last_action == Action.BUY) or (action == Action.SELL and self.last_action == Action.SELL) or (action == Action.SELL and self.last_action == Action.INITIAL)
        early_termination = money_loss_condition #or invalid_sequence_condition
        self.current_step += 1
        self.step_counter += 1
        
        self._take_action(action)

        done = (self.current_step >= len(self.df)-1) or (self.step_counter >= maximum_steps)
        
        obs = self._next_observation()

        if not done and not early_termination:
            return ts.transition(obs, reward = reward, discount = 1.0)
        
        elif done and not early_termination:
            print('END profit: {0:.2f}% actions: buy = {1}; sell = {2}; hold = {3}'.format(self.cum_profit * 100, self.buy_actions, self.sell_actions, self.hold_actions))
            self._soft_reset()
            return ts.termination(obs, reward = reward)
        
        else:
            print(str(early_termination), 'TERM profit: {0:.2f}% actions: buy = {1}; sell = {2}; hold = {3}'.format(self.cum_profit * 100, self.buy_actions, self.sell_actions, self.hold_actions))
            self._soft_reset()
            return ts.termination(obs, reward = reward)

    def _soft_reset(self):
        # print("internally resetting environment")
        self._render_offset = self._start_offset
        if self.capture_mode:
            self.prepare_render_data()

        self._start_offset = np.random.randint(1, len(self.df)-1-maximum_steps)
        self.step_counter = 0
        self.current_step = self._start_offset
        self.steps_beyond_done = None

         # initialize variables to keep track of trades
        self.buy_price = 0
        self.sell_price = 0
        self.fee = 0.2/100
        self.last_action = Action.INITIAL
        self.steps_on_hold = 0
        self.cum_profit = 0

        self.buy_actions = 0
        self.sell_actions = 0
        self.hold_actions = 0

        # visualizsation
        self._render_buy_idxs = self.buy_idxs
        self._render_sell_idxs = self.sell_idxs
        self.buy_idxs = []
        self.sell_idxs = []

    def _reset(self):
        # print("externally resetting environment")
        self._render_offset = self._start_offset
        self._start_offset = np.random.randint(1, len(self.df)-1-maximum_steps)
        self.step_counter = 0
        self.current_step = self._start_offset
        self.steps_beyond_done = None

         # initialize variables to keep track of trades
        self.buy_price = 0
        self.sell_price = 0
        self.fee = 0.2/100
        self.last_action = Action.INITIAL
        self.steps_on_hold = 0
        self.cum_profit = 0

        self.buy_actions = 0
        self.sell_actions = 0
        self.hold_actions = 0

        # visualizsation
        self._render_buy_idxs = self.buy_idxs
        self._render_sell_idxs = self.sell_idxs
        self.buy_idxs = []
        self.sell_idxs = []

        return ts.restart(self._next_observation())

    def observation_spec(self):
        return tf_agents.specs.BoundedArraySpec(shape=(self.window_size, 9), dtype=np.float32, minimum=0, maximum=np.inf, name='observation')

    def action_spec(self):
        return tf_agents.specs.BoundedTensorSpec(shape=(), dtype=np.int32, minimum=0, maximum=len(Action)-2, name='action')
    
    def prepare_render_data(self):
        data_buf = {
            'buy_idxs': self.buy_idxs,
            'sell_idxs': self.sell_idxs,
            'prices': self.df['close'][self._render_offset:self._render_offset+maximum_steps].tolist(),
            'close' : self.df['close'].tolist(),
            'offset': self._render_offset
        }
        image_data_queue.put(data_buf)

    def render(self, mode='human'):
        while not image_data_queue.empty():
            data_buf = image_data_queue.get()

            plt.ioff()
            plt.figure(figsize=(32,18))

            plt.plot(data_buf["prices"], label='Close Price', color='blue', alpha=0.5)

            for idx in data_buf["buy_idxs"]:
                plt.scatter(idx-data_buf["offset"], data_buf["close"][idx], color='green')

            for idx in data_buf["sell_idxs"]:
                plt.scatter(idx-data_buf["offset"], data_buf["close"][idx], color='red')

            plt.title('Stock Price with Buy and Sell Actions')
            plt.legend()

            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            id = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
            filename = f"results/plots/{current_datetime}-{str(data_buf['offset'])}-{id}-actions.png"
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            plt.close()

    def evaluate_reward_function(self, action):
        pass

def create_environment(data, num_parallel_environments):
    eval_env = tf_py_environment.TFPyEnvironment(FinancialEnvironment(data, window_size = 20, capture_mode=True))
    train_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [lambda: FinancialEnvironment(data, window_size = 20, capture_mode = False) for _ in range(num_parallel_environments)]
        )
    )

    return (train_env, eval_env)
