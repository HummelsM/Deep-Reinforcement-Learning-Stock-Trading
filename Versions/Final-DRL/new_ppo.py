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
from Datasets.Fake-Data.fake_data import generate_sine_wave_data,generate_mean_reverting_random_walk,generate_brownian_motion,generate_synthetic_data
from Preprocess.preprocess import preprocess_data,compute_rsi,compute_macd

sine_wave_data = generate_sine_wave_data(length=500)
mean_reverting_data = generate_mean_reverting_random_walk(length=500)
brownian_data = generate_brownian_motion(length=500)




# Generate datasets
bullish_data = generate_synthetic_data(start_price=100, days=200, trend='bullish')
bearish_data = generate_synthetic_data(start_price=100, days=200, trend='bearish')
volatile_data = generate_synthetic_data(start_price=100, days=200, trend='volatile')


# Plotting
plt.figure(figsize=(12, 6))
plt.plot(bullish_data['Day'], bullish_data['Close'], label="Bullish")
plt.plot(bearish_data['Day'], bearish_data['Close'], label="Bearish")
plt.plot(volatile_data['Day'], volatile_data['Close'], label="Volatile")
plt.title("Synthetic Stock Price Data")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()



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

# Generate synthetic dataset

# Load data
#data = bearish_data
df=volatile_data
data=preprocess_data(df)
#data=pd.read_csv('TSLA.csv')
#data.drop('Date',axis=1,inplace=True)

# Split data into training and testing
#train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
def split_data(data, train_ratio=0.7, valid_ratio=0.15):
    """
    Splits the dataset into training, validation, and testing sets.

    Parameters:
        data (pd.DataFrame): The complete dataset
        train_ratio (float): Proportion of data for training
        valid_ratio (float): Proportion of data for validation

    Returns:
        train_data, valid_data, test_data (pd.DataFrame): The splits
    """
    n = len(data)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    
    train_data = data.iloc[:train_end]
    valid_data = data.iloc[train_end:valid_end]
    test_data = data.iloc[valid_end:]
    
    return train_data, valid_data, test_data

train_data, valid_data,test_data = split_data(data)

# Create environments
train_env = StockTradingEnv(data=train_data)
valid_env = StockTradingEnv(data=valid_data)
test_env = StockTradingEnv(data=test_data)

# Check environments
check_env(train_env)

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)

# Create the evaluation callback
eval_callback = EvalCallback(
    valid_env,  # Evaluation environment
    best_model_save_path='./logs/',
    log_path='./logs/',
    eval_freq=500,  # Evaluate every 5000 steps
    deterministic=True,
    render=False,
    callback_on_new_best=stop_callback
)

'''
# Train model
#model = A2C("MlpPolicy", train_env, verbose=1,gamma=0.6,ent_coef=0.04,vf_coef=0.5,learning_rate=0.007)
#model = PPO("MlpPolicy", train_env, verbose=1,ent_coef=0.0001)
model = PPO("MlpPolicy", train_env, verbose=1)
#model.learn(total_timesteps=32000,callback=eval_callback)
model.learn(total_timesteps=9000,callback=eval_callback)
'''

train_rewards, test_rewards = [], []
train_roi, test_roi = [], []

# Train and evaluate the model
num_episodes = 200
# Training Loop
for episode in range(num_episodes):
    '''   
    obs, info = train_env.reset()  # Unpack the tuple
    episode_reward = 0
    while True:
        action, _ = model.predict(obs)  # Pass only the `obs`
        obs, reward, terminated, truncated, _ = train_env.step(action)  # Unpack 5 return values
        test_env.render()
        
        episode_reward += reward
        if terminated or truncated:
            break
    train_rewards.append(episode_reward)
    train_roi.append((train_env.portfolio_value - train_env.initial_cash) / train_env.initial_cash)

    
    if os.path.exists("ppo_stock_trading"):
        os.remove("ppo_stock_trading")

    model.save("ppo_stock_trading")
    #del model
    '''

    model = PPO.load("ppo_stock_trading.zip")

    # Evaluation Loop
    obs, info = test_env.reset()  # Unpack the tuple
    episode_reward = 0
    while True:
        action, _ = model.predict(obs)  # Pass only the `obs`
        obs, reward, terminated, truncated, _ = test_env.step(action)  # Unpack 5 return values
        episode_reward += reward
        if terminated or truncated:
            break
    test_rewards.append(episode_reward)
    test_roi.append((test_env.portfolio_value - test_env.initial_cash) / test_env.initial_cash)

    try:
        loaded_model = PPO.load("ppo_stock_trading.zip")
    except Exception as e:
        print(f"Error loading model: {e}")


# Visualize results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(train_rewards, label="Train Rewards")
plt.plot(test_rewards, label="Test Rewards")
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(train_roi, label="Train ROI")
plt.plot(test_roi, label="Test ROI")
plt.xlabel("Episode")
plt.ylabel("ROI")
plt.legend()

plt.tight_layout()
plt.show()





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

# Generate cumulative returns and Sharpe ratio for train and test rewards

if len(train_roi) > 1:
    train_returns = pd.Series(train_roi).pct_change().fillna(0)
else:
    train_returns = pd.Series([0])  # Default to zero returns for insufficient data

if len(test_roi) > 1:
    test_returns = pd.Series(test_roi).pct_change().fillna(0)
else:
    test_returns = pd.Series([0])  # Default to zero returns for insufficient data


#train_returns = pd.Series(train_rewards).pct_change().fillna(0)
#test_returns = pd.Series(test_rewards).pct_change().fillna(0)


train_metrics = calculate_metrics(train_returns)
test_metrics = calculate_metrics(test_returns)

# Append metrics to a DataFrame for comparison
metrics_df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"])

# Plotting Results
plt.figure(figsize=(15, 12))

# Plot cumulative rewards
plt.subplot(3, 1, 1)
plt.plot(train_rewards, label="Train Rewards")
plt.plot(test_rewards, label="Test Rewards")
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.legend()
plt.title("Cumulative Rewards Over Episodes")

# Plot ROI
plt.subplot(3, 1, 2)
plt.plot(train_roi, label="Train ROI")
plt.plot(test_roi, label="Test ROI")
plt.xlabel("Episode")
plt.ylabel("ROI")
plt.legend()
plt.title("ROI Over Episodes")

# Bar plot for performance metrics
plt.subplot(3, 1, 3)
metrics_df[["Cumulative Return", "Sharpe Ratio", "Calmar Ratio"]].plot.bar(figsize=(10, 6), ax=plt.gca())
plt.title("Performance Metrics Comparison")
plt.ylabel("Value")
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Plot cumulative returns curve
def plot_cumulative_returns(series, title):
    plt.figure(figsize=(12, 6))
    cumulative_returns = np.exp(np.log1p(series).cumsum())
    plt.plot(cumulative_returns, label="Cumulative Returns")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_cumulative_returns(train_returns, "Train Cumulative Returns")
plot_cumulative_returns(test_returns, "Test Cumulative Returns")


# Calculate rolling average ROI
window_size = 10  # Adjust window size for smoothing
average_test_roi = pd.Series(test_roi).rolling(window=window_size).mean()

# Plot ROI with rolling average
plt.figure(figsize=(12, 6))
plt.plot(test_roi, label="Test ROI", alpha=0.5)
plt.plot(average_test_roi, label=f"Rolling Avg ROI (window={window_size})", linewidth=2, color='red')
plt.xlabel("Episode")
plt.ylabel("ROI")
plt.legend()
plt.title("Test ROI and Rolling Average")
plt.grid(True)
plt.show()




