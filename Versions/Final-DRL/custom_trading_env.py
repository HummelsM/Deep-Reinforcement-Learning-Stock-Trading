import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO,DQN,A2C
from stable_baselines3.common.env_checker import check_env
from gym import spaces
import random
import os
import gymnasium as gym  # Replace `gym` with `gymnasium`
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from fast_ml.model_development import train_valid_test_split
from sklearn.preprocessing import MinMaxScaler
from fake_data import generate_sine_wave_data
from fake_data import generate_mean_reverting_random_walk
from fake_data import generate_brownian_motion
from fake_data import generate_synthetic_data
from preprocess import preprocess_data,compute_rsi,compute_macd


class StockTradingEnv(gym.Env):  # Inherit from gymnasium.Env
    def __init__(self, data, initial_cash=10000,early_stop_threshold=0.2):
        self.data = data
        self.initial_cash = initial_cash
        self.early_stop_threshold = early_stop_threshold
        self.reset()
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32
        )

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)  # Call the parent class's reset to handle seeding
        np.random.seed(seed)  #
        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.buy_and_hold_value = self.initial_cash
        self.random_action_value = self.initial_cash
        self.shares_held = 0
        self.initial_price = self.data.iloc[0]["Close"]
        return self._get_observation(), {}  # Gymnasium reset requires returning (obs, info)

    def _get_observation(self):
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        return obs

    def step(self, action):
        current_price = self.data.iloc[self.current_step]["Close"]

        # Execute action
        if action == 0:  # Hold
            pass
        elif action == 1:  # Buy
            if self.cash > current_price:
                self.shares_held += self.cash // current_price
                self.cash %= current_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.cash += self.shares_held * current_price
                self.shares_held = 0

        # Update portfolio value
        self.portfolio_value = self.cash + self.shares_held * current_price

        # Update benchmarks
        self.buy_and_hold_value = self.initial_cash + (current_price - self.initial_price) * (self.initial_cash // self.initial_price)
        self.random_action_value += random.choice([-current_price, current_price, 0])

        # Calculate reward
        reward = self._calculate_reward()

        # Check for portfolio loss termination
        early_stop = self.portfolio_value < self.early_stop_threshold * self.initial_cash

        # Advance step
        self.current_step += 1
        terminated = early_stop or self.current_step >= len(self.data) - 1
        truncated = False  # Add logic if you have time limits or truncation conditions

        return self._get_observation(), reward, terminated, truncated, {} # Gymnasium requires returning (obs, reward, done, info)

    def _calculate_reward(self):
        roi = (self.portfolio_value - self.initial_cash) / self.initial_cash
        if roi > 0.1:
            return 10
        elif roi > 0:
            return 5
        elif roi == 0:
            return -1
        else:
            return -10



    def render(self):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}, Cash: {self.cash:.2f}, Shares Held: {self.shares_held}")





def calculate_metrics(returns):
    #cumulative_return = np.exp(np.log1p(returns).cumsum())[-1] - 1
    cumulative_return = np.exp(np.log1p(returns).cumsum().iloc[-1]) - 1

    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    calmar_ratio = (returns.mean() * 252) / abs(max_drawdown)
    downside_std = returns[returns < 0].std()
    sortino_ratio = (returns.mean() * np.sqrt(252)) / downside_std if downside_std > 0 else np.nan

    return {
        "Cumulative Return": cumulative_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Sortino Ratio": sortino_ratio
    }



