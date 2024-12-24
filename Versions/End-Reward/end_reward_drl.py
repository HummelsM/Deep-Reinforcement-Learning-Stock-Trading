import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Custom Stock Trading Environment
#This algorithm utilizes the stable-baselines3 rl algorithms
#to train the environment as to what action should be taken



class StockTradingEnv(gym.Env):
    def __init__(self, data, initial_cash=1000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.returns=0
        self.cash=initial_cash
        self.total_cash=initial_cash
        self.initial_investment = initial_cash
        self.final_investment = initial_cash
        self.current_idx = 5  # Start after the first 5 days
        self.shares = 0
        self.trades = []
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_idx = 5
        self.final_investment=self.initial_investment
        self.shares = 0
        self.trades = []
        return self._get_state()

    def step(self, action):
        if self.current_idx >= len(self.data) - 5:
            return self._get_state(), 0, True, {}

        state = self._get_state()

        self._update_investment(action)
        self.trades.append((self.current_idx, action))
        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 5
        next_state = self._get_state()

        reward = 0  # Intermediate reward is 0, final reward will be given at the end of the episode

        return next_state, reward, done, {}

    def _get_state(self):
        window_size = 5
        state = self.data['Close'].iloc[self.current_idx - window_size:self.current_idx].values
        state = (state - np.mean(state))  # Normalizing the state
        return state

    def _update_investment(self, action):
        current_price = self.data['Close'].iloc[self.current_idx]
        if action == 1:  # Buy
            shares_to_buy = self.cash / current_price  # Calculate how many shares can be bought
            self.shares += shares_to_buy  # Add shares to the portfolio
            self.cash -= shares_to_buy * current_price  # Deduct the equivalent cash
            
        elif action == 2:  # Sell
            cash_from_sale = self.shares * current_price  # Calculate cash from selling all shares
            self.cash += cash_from_sale  # Add cash to the available cash
            self.shares = 0  # Set shares to 0 (if partial selling is required, modify this)
            
        #self.final_investment = self.final_investment + self.shares * current_price
        self.total_cash=self.cash
        self.final_investment = self.cash + self.shares * current_price

    def _get_final_reward(self):
        #roi = (self.final_investment - self.initial_investment) / self.initial_investment
        #roi = (self.cash - self.initial_investment) / self.initial_investment
        self.returns= self.initial_investment + (0.05 * self.initial_investment)
        if self.total_cash > self.returns:
            return 1
        elif self.total_cash <= self.total_cash <= self.returns and self.total_cash >= self.initial_investment:
            return 0
        elif self.total_cash < self.initial_investment:
            return -1

    def render(self, mode="human", close=False, episode_num=None):
        #roi = (self.final_investment - self.initial_investment) / self.initial_investment
        #roi = (self.cash - self.initial_investment) / self.initial_investment
        reward = self._get_final_reward()
        print(f'Episode: {episode_num}, Initial Investment: {self.initial_investment}, '
              f'Total Portfolio Value: {self.final_investment}, Total Cash: {self.total_cash}, Returns: {self.returns}, Reward: {reward}')





# Train and Test with RL Model
if __name__ == '__main__':

    #####Loading datasets#####
    
    # Load the training dataset
    train_df = pd.read_csv('GOOG.csv')
    
    start_date = '2020-01-03'
    end_date = '2021-12-28'

    train_data = train_df[(train_df['Date'] >= start_date) & (train_df['Date'] <= end_date)]
    train_data = train_data.set_index('Date')
    

    # Test the model on a different dataset
    test_df = pd.read_csv('TSLA.csv')
    start_date = '2022-01-03'
    end_date = '2022-12-28'

    test_data = test_df[(test_df['Date'] >= start_date) & (test_df['Date'] <= end_date)]
    test_data = test_data.set_index('Date')

    ###################

    # Create and train the RL model
    env = DummyVecEnv([lambda: StockTradingEnv(train_data)])
    env = StockTradingEnv(train_data, initial_cash=10000)

    
    #model = A2C("MlpPolicy", env, verbose=1,gamma=0.95,learning_rate=0.1,ent_coef=1)
    #model = PPO("MlpPolicy", env, verbose=1,gamma=0.95,learning_rate=0.1,ent_coef=1)
    model = DQN("MlpPolicy", env, verbose=1,gamma=0.95,learning_rate=0.01)
    model.learn(total_timesteps=25000)
    model.save("ppo_cartpole")
    del model # remove to demonstrate saving and loading
    model = DQN.load("ppo_cartpole")
    #model = A2C.load("ppo_cartpole")
    #model = PPO.load("ppo_cartpole")

    env = StockTradingEnv(test_data, initial_cash=1000)

    num_test_episodes = 200  # Define the number of test episodes
    cumulative_reward = 0

    for episode in range(num_test_episodes):
        state = env.reset()
        done = False

        while not done:
            state = state.reshape(1, -1)
            action, _states = model.predict(state)  # Use the trained model to predict actions
            next_state, _, done, _ = env.step(action)
            state = next_state
        
        reward = env._get_final_reward()
        cumulative_reward += reward
        env.render(episode_num=episode + 1)

    print(f'Cumulative Reward after {num_test_episodes} episodes: {cumulative_reward}')

