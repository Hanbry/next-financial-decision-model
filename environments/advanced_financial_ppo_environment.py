import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tf_agents

import random
import string

from enum import Enum
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts

class Action(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

maximum_steps = 3000
initial_capital = 400

class FinancialEnvironment(py_environment.PyEnvironment):
    def __init__(self, data):
        super(FinancialEnvironment, self).__init__()
        self.df = data
        self.current_step = 0
        self._start_offset = np.random.randint(1, len(data)-1-maximum_steps)
        self._render_offset = self._start_offset
        self.step_counter = 0
        self.current_step = self._start_offset
        self.steps_beyond_done = None

        # initialize variables to keep track of trades
        self.buy_price = 0
        self.sell_price = 0
        self.fee = 0.2/100
        self.last_action = -1
        self.steps_on_hold = 0
        self.capital = initial_capital
        self.shares = 0

        self.buy_actions = 0
        self.sell_actions = 0
        self.hold_actions = 0

        # visualizsation
        self.buy_idxs = []
        self.sell_idxs = []
        self._render_buy_idxs = []
        self._render_sell_idxs = []
    
    def _next_observation(self):
        obs = np.empty(6, dtype=np.float32)
        cur_idx = self.current_step
        obs = np.array([
            (self.df.loc[cur_idx, 'open']-self.df.loc[cur_idx, 'open'])/self.df.loc[cur_idx, 'open'],
            (self.df.loc[cur_idx, 'close']-self.df.loc[cur_idx, 'close'])/self.df.loc[cur_idx, 'close'],
            (self.df.loc[cur_idx, 'high']-self.df.loc[cur_idx, 'high'])/self.df.loc[cur_idx, 'high'],
            (self.df.loc[cur_idx, 'low']-self.df.loc[cur_idx, 'low'])/self.df.loc[cur_idx, 'low'],
            (self.df.loc[cur_idx, 'volume']-self.df.loc[cur_idx, 'volume'])/self.df.loc[cur_idx, 'volume'] if self.df.loc[cur_idx, 'volume'] else 0,
            self.df.loc[cur_idx, 'time']
        ], dtype = np.float32)

        return obs

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "close"]

        if action == Action.BUY.value:
            self.buy_actions += 1
            self.buy_price = current_price
            self.last_action = 0
            self.steps_on_hold = 0
            self.shares = (self.capital/current_price)*(1-self.fee)
            self.capital = 0
            self.buy_idxs.append(self.current_step)

        elif action == Action.SELL.value:
            self.sell_actions += 1
            self.sell_price = current_price
            self.last_action = 1
            self.steps_on_hold = 0
            self.capital = (self.shares*current_price)*(1-self.fee)
            self.shares = 0
            self.sell_idxs.append(self.current_step)

        else: # action == Action.HOLD.value
            self.hold_actions += 1
            self.steps_on_hold += 1

    def _calculate_reward(self, action, price):
        current_price = self.df.loc[self.current_step, "close"]

        # Normal sell after buy
        if self.last_action == Action.BUY.value and action == Action.SELL.value:
            reward = (self.capital - initial_capital) / initial_capital

        # Normal buy after sell
        elif self.last_action == Action.SELL.value and action == Action.BUY.value:
            reward = (current_price - self.sell_price) / self.sell_price

        # Hold after buy
        elif action == Action.HOLD.value and self.shares > 0:
            reward = ((current_price - self.buy_price) / self.buy_price) * 0.2
        
        # Hold after sell
        elif action == Action.HOLD.value and self.last_action == Action.SELL.value:
            reward = 0 #((self.sell_price - current_price) / self.sell_price) * 0.2

        # Initial buy
        elif action == Action.BUY.value and self.last_action == -1:
            reward = 1

        else:
            reward = 0

        # Apply trade penalty
        num_trades = self.buy_actions + self.sell_actions
        trade_penalty = (num_trades ** 2) / 1000
        reward -= trade_penalty

        # Apply holding penatly
        reward -= self.steps_on_hold * 0.01

        return reward

    def _step(self, action):
        current_price = self.df.loc[self.current_step, "close"]
        reward = self._calculate_reward(action, current_price)
        self._take_action(action)
        self.current_step += 1
        self.step_counter += 1

        money_loss_condition = (self.capital <= (initial_capital * 0.5) and self.shares <= 0) or ((self.shares * current_price) <= (initial_capital * 0.5) and self.capital <= 0)
        invalid_sequence_condition = (action == Action.BUY.value and self.last_action == Action.BUY.value) or (action == Action.SELL.value and self.last_action == Action.SELL.value)
        early_termination = money_loss_condition or invalid_sequence_condition
        done = (self.current_step >= len(self.df)-1) or (self.step_counter >= maximum_steps)
        
        obs = self._next_observation()

        if not done and not early_termination:
            return ts.transition(obs, reward = reward, discount = 1.0)
        
        elif self.steps_beyond_done is None and done and not early_termination:
            print('actions: buy = {0}; sell = {1}; hold = {2}'.format(self.buy_actions, self.sell_actions, self.hold_actions))
            self.steps_beyond_done = 0
            return ts.transition(obs, reward = reward, discount = 1.0)
        
        elif self.steps_beyond_done is not None and done and not early_termination:
            # print("send termination step ", self.steps_beyond_done)
            self.steps_beyond_done += 1
            self._soft_reset()
            return ts.termination(obs, reward=reward)
        
        else:
            # print("early termination: ", str(early_termination))
            # print('actions: buy = {0}; sell = {1}; hold = {2}'.format(self.buy_actions, self.sell_actions, self.hold_actions))
            self._soft_reset()
            return ts.termination(obs, reward=reward)

    def _soft_reset(self):
        print("internally resetting environment")
        self._render_offset = self._start_offset
        self._start_offset = np.random.randint(1, len(self.df)-1-maximum_steps)
        self.step_counter = 0
        self.current_step = self._start_offset
        self.steps_beyond_done = None

         # initialize variables to keep track of trades
        self.buy_price = 0
        self.sell_price = 0
        self.fee = 0.2/100
        self.last_action = Action.SELL
        self.steps_on_hold = 0
        self.capital = initial_capital
        self.shares = 0

        self.buy_actions = 0
        self.sell_actions = 0
        self.hold_actions = 0

        # visualizsation
        self._render_buy_idxs = self.buy_idxs
        self._render_sell_idxs = self.sell_idxs
        self.buy_idxs = []
        self.sell_idxs = []

    def _reset(self):
        print("externally resetting environment")
        self._render_offset = self._start_offset
        self._start_offset = np.random.randint(1, len(self.df)-1-maximum_steps)
        self.step_counter = 0
        self.current_step = self._start_offset
        self.steps_beyond_done = None

         # initialize variables to keep track of trades
        self.buy_price = 0
        self.sell_price = 0
        self.fee = 0.2/100
        self.last_action = Action.SELL
        self.steps_on_hold = 0
        self.capital = initial_capital
        self.shares = 0

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
        return tf_agents.specs.BoundedArraySpec(shape=(6,), dtype=np.float32, minimum=0, maximum=np.inf, name='observation')

    def action_spec(self):
        return tf_agents.specs.BoundedTensorSpec(shape=(), dtype=np.int32, minimum=0, maximum=len(Action)-1, name='action')
    
    def render(self, mode='human'):
        if mode == 'human':
            plt.ioff()  # Turn off interactive mode to prevent displaying the plot
            plt.figure(figsize=(32,18))
            
            plt.plot(self.df['close'][self._render_offset:self._render_offset+maximum_steps], label='Close Price', color='blue', alpha=0.5)

            for idx in self._render_buy_idxs:
                plt.scatter(idx, self.df.loc[idx, 'close'], color='green')
                
            for idx in self._render_sell_idxs:
                plt.scatter(idx, self.df.loc[idx, 'close'], color='red')
            
            plt.title('Stock Price with Buy and Sell Actions')
            plt.legend()

            id = ''.join(random.choices(string.ascii_letters + string.digits, k=4))

            plt.savefig(f"results/{str(self._render_offset)}-{id}-actions.png")
            plt.close()

    def evaluate_reward_function(self, action):
        pass


def create_environment(data):
    train_py_env = FinancialEnvironment(data)
    eval_py_env = FinancialEnvironment(data)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    return (train_py_env, eval_py_env, train_env, eval_env)
