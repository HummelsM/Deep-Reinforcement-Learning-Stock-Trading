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
from preprocess import preprocess_data,compute_rsi,compute_macd,split_data
from custom_trading_env import StockTradingEnv, calculate_metrics


#Load and Generate datasets
sine_wave_data = generate_sine_wave_data(length=500)
mean_reverting_data = generate_mean_reverting_random_walk(length=500)
brownian_data = generate_brownian_motion(length=500)


bullish_data = generate_synthetic_data(start_price=100, days=200, trend='bullish')
bearish_data = generate_synthetic_data(start_price=100, days=200, trend='bearish')
volatile_data = generate_synthetic_data(start_price=100, days=200, trend='volatile')

real_world_data=pd.read_csv('GOOG.csv')

df=sine_wave_data
data=preprocess_data(df)
#data=pd.read_csv('TSLA.csv')
#data.drop('Date',axis=1,inplace=True)


'''
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
'''



#Data Splitting
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



def train_agent(algorithm, env, save_path, total_timesteps=20000, callback=None):
    model = algorithm("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(save_path)
    return model

def evaluate_agent(model, env, num_episodes=100):
    rewards = []
    roi_list = []
    observations = []  # Store observations
    actions = []  # Store actions
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        while True:
            action, _ = model.predict(obs)
            observations.append(obs)  # Log observations
            actions.append(action)    # Log actions
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        roi_list.append((env.portfolio_value - env.initial_cash) / env.initial_cash)
    
    return rewards, roi_list, np.array(observations), np.array(actions)


# Define paths for saving models
save_paths = {
    "PPO": "ppo_stock_trading",
    "A2C": "a2c_stock_trading",
    "DQN": "dqn_stock_trading",
}

# Train and evaluate the model
models = {}
for name, algorithm in [("PPO", PPO), ("A2C", A2C), ("DQN", DQN)]:
    print(f"Training {name}...")
    models[name] = train_agent(algorithm, train_env, save_paths[name], total_timesteps=10000, callback=eval_callback)


results = {}
for name, save_path in save_paths.items():
    print(f"Evaluating {name}...")
    model = PPO.load(save_path) if name == "PPO" else \
            A2C.load(save_path) if name == "A2C" else \
            DQN.load(save_path)
    rewards, roi, observations, actions = evaluate_agent(model, test_env, num_episodes=100)
    results[name] = {"rewards": rewards, "roi": roi}


metrics = {}
for name, result in results.items():
    returns = pd.Series(result["roi"]).pct_change().fillna(0)
    metrics[name] = calculate_metrics(returns)
metrics_df = pd.DataFrame(metrics).T


# Plot cumulative rewards
plt.figure(figsize=(10, 6))
for name, result in results.items():
    plt.plot(result["rewards"], label=f"{name} Rewards")
plt.title("Cumulative Rewards Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.legend()
plt.show()

# Plot ROI
plt.figure(figsize=(10, 6))
for name, result in results.items():
    plt.plot(result["roi"], label=f"{name} ROI")
plt.title("ROI Over Episodes")
plt.xlabel("Episode")
plt.ylabel("ROI")
plt.legend()
plt.show()

# Bar plot for performance metrics
metrics_df[["Cumulative Return", "Sharpe Ratio", "Calmar Ratio"]].plot.bar(figsize=(10, 6))
plt.title("Performance Metrics Comparison")
plt.ylabel("Value")
plt.grid(True)
plt.xticks(rotation=45)
plt.show()



                                                              ######################### Market Scenarios Visualizations ####################


scenarios = {
    "Bullish": bullish_data,
    "Bearish": bearish_data,
    "Volatile": volatile_data
}

scenario_results = {}
for scenario, data in scenarios.items():
    print(f"Evaluating agents on {scenario} scenario...")
    env = StockTradingEnv(data=preprocess_data(data))
    for name, model in models.items():
        rewards, roi, observations, actions = evaluate_agent(model, test_env, num_episodes=100)
        scenario_results.setdefault(name, {})[scenario] = {"rewards": rewards, "roi": roi}


market_metrics = {}
for agent, scenarios in scenario_results.items():
    market_metrics[agent] = {scenario: calculate_metrics(pd.Series(data["roi"]).pct_change().fillna(0)) for scenario, data in scenarios.items()}
market_metrics_df = pd.DataFrame.from_dict(market_metrics, orient="index")


# Plot ROI for all agents grouped by market scenario
plt.figure(figsize=(15, 12))
for i, scenario in enumerate(scenarios.keys()):
    plt.subplot(3, 1, i + 1)
    for agent, results in scenario_results.items():
        plt.plot(results[scenario]["roi"], label=f"{agent}")
    plt.title(f"ROI for {scenario} Market")
    plt.xlabel("Episode")
    plt.ylabel("ROI")
    plt.legend()
plt.tight_layout()
plt.show()

# Plot rewards for all agents grouped by market scenario
plt.figure(figsize=(15, 12))
for i, scenario in enumerate(scenarios.keys()):
    plt.subplot(3, 1, i + 1)
    for agent, results in scenario_results.items():
        plt.plot(results[scenario]["rewards"], label=f"{agent}")
    plt.title(f"Rewards for {scenario} Market")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
plt.tight_layout()
plt.show()


# Bar chart for scenario-specific performance metrics
metrics_to_plot = ["Cumulative Return", "Sharpe Ratio", "Calmar Ratio"]

for metric in metrics_to_plot:
    scenario_metric_data = market_metrics_df.applymap(lambda x: x.get(metric) if isinstance(x, dict) else None)
    scenario_metric_data.plot.bar(figsize=(12, 6))
    plt.title(f"{metric} Across Agents and Scenarios")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()



                                                        ######################### Functions Visualizations ####################



functions = {
    #"Sine Wave": sine_wave_data,
    "Brownian Motion": brownian_data,
    "Real World Data": real_world_data,
    "Random Walk": mean_reverting_data
}

functions_results = {}
for function, data in functions.items():
    print(f"Evaluating agents on {function} function...")
    env = StockTradingEnv(data=preprocess_data(data))
    for name, model in models.items():
        rewards, roi, obeservations,actions = evaluate_agent(model, test_env, num_episodes=100)
        functions_results.setdefault(name, {})[function] = {"rewards": rewards, "roi": roi}


func_metrics = {}
for agent, functions in functions_results.items():
    func_metrics[agent] = {function: calculate_metrics(pd.Series(data["roi"]).pct_change().fillna(0)) for function, data in functions.items()}
func_metrics_df = pd.DataFrame.from_dict(func_metrics, orient="index")


# Plot ROI for all agents grouped by function
plt.figure(figsize=(15, 12))
for i, function in enumerate(functions.keys()):
    plt.subplot(3, 1, i + 1)
    for agent, results in functions_results.items():
        plt.plot(results[function]["roi"], label=f"{agent}")
    plt.title(f"ROI for {function} Market")
    plt.xlabel("Episode")
    plt.ylabel("ROI")
    plt.legend()
plt.tight_layout()
plt.show()

# Plot rewards for all agents grouped by function
plt.figure(figsize=(15, 12))
for i, function in enumerate(functions.keys()):
    plt.subplot(3, 1, i + 1)
    for agent, results in functions_results.items():
        plt.plot(results[function]["rewards"], label=f"{agent}")
    plt.title(f"Rewards for {function} Market")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
plt.tight_layout()
plt.show()


# Bar chart for scenario-specific performance metrics
metrics_to_plot = ["Cumulative Return", "Sharpe Ratio", "Calmar Ratio"]

for metric in metrics_to_plot:
    func_metric_data = func_metrics_df.applymap(lambda x: x.get(metric) if isinstance(x, dict) else None)
    func_metric_data.plot.bar(figsize=(12, 6))
    plt.title(f"{metric} Across Agents and Functions")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()




                                                              ###########################  For PPO ########################


import shap
# Evaluate the model and collect observations
rewards, roi, observations, actions = evaluate_agent(models["PPO"], test_env, num_episodes=100)

# SHAP Analysis
# Define a wrapper function for PPO model predictions
def predict_with_ppo(observations):
    actions, _ = models["PPO"].policy.predict(observations, deterministic=True)
    return actions  # SHAP requires outputs, e.g., probabilities or logits

# Create the SHAP explainer
explainer = shap.Explainer(predict_with_ppo, observations)
shap_values = explainer(observations)

# SHAP Summary Plot
shap.summary_plot(shap_values, features=observations, feature_names=[f"Feature_{i}" for i in range(observations.shape[1])])

# SHAP Dependence Plot for Specific Feature
shap.dependence_plot(0, shap_values.values, observations)  # Replace 0 with the index of a feature


                                     ########################### For DQN ###################################
# Evaluate the model and collect observations
rewards, roi, observations, actions = evaluate_agent(models["PPO"], test_env, num_episodes=100)

# SHAP Analysis
# Define a wrapper function for DQN model predictions
def predict_with_ppo(observations):
    actions, _ = models["DQN"].policy.predict(observations, deterministic=True)
    return actions  # SHAP requires outputs, e.g., probabilities or logits

# Create the SHAP explainer
explainer = shap.Explainer(predict_with_ppo, observations)
shap_values = explainer(observations)

# SHAP Summary Plot
shap.summary_plot(shap_values, features=observations, feature_names=[f"Feature_{i}" for i in range(observations.shape[1])])

# SHAP Dependence Plot for Specific Feature
shap.dependence_plot(0, shap_values.values, observations)  # Replace 0 with the index of a feature


                                     ########################### For A2C ##################################

# Evaluate the model and collect observations
rewards, roi, observations, actions = evaluate_agent(models["PPO"], test_env, num_episodes=100)

# SHAP Analysis
# Define a wrapper function for A2C model predictions
def predict_with_ppo(observations):
    actions, _ = models["A2C"].policy.predict(observations, deterministic=True)
    return actions  # SHAP requires outputs, e.g., probabilities or logits

# Create the SHAP explainer
explainer = shap.Explainer(predict_with_ppo, observations)
shap_values = explainer(observations)

# SHAP Summary Plot
shap.summary_plot(shap_values, features=observations, feature_names=[f"Feature_{i}" for i in range(observations.shape[1])])

# SHAP Dependence Plot for Specific Feature
shap.dependence_plot(0, shap_values.values, observations)  # Replace 0 with the index of a feature

