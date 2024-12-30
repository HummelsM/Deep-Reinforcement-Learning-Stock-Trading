# deep-reinforcement-learning-stock-trading
This stock trading deep reinforcement learning repository is part of a thesis project. It houses multiple stock trading environments, each with a somewhat different workflow.
This project was tested on and is compatible with Python 3.10 and 3.11 versions, however other versions may also be compatible. Additional libraries which were used are included in the requirements.txt file.

The code has been tested on both synthetic as well as real-world data from the S&P500 companies such as Google,Microsoft, UBER,Apple, Tesla,Amazon,etc.This data was obtained from the YFinance library. However data from other sources can be used as well.
Synthetic data includes stock market trends like bear, bull and volatile markets, and data from functions such as Sine Wave, Random Walk and Brownian Motion. These functions and trends can be created using methods from the fake_data.py file. A preprocess file called preprocess.py is available which houses data preprocessing methods as well feature engineering methods like RSI and MACD integration.

This project used 3 deep reinforcement learning algorithms: PPO,A2C, DQN. The reason for this is because the environment uses a discrete action space and a continuous observation space, both of which are already embodied in the aforementioned DRL algorithms.

The Final-DRL environment is the one used in the final thesis results.In the Final-DRL folder, the programs for DQN, PPO AND A2C are written in sepearate files i order to make hyperparameteric tuning easier.
The visualize file is used for comparison of the three algorithms. For ease of use, the evironment which is used by these algorithms in their separate files is also available in the custom_trading_env.py file so that the visualize file can easily call the environment.

This thesis found that the Final-DRL model performs well for A2C and PPO and moderately well for DQN. This is because the environment somewhat encourages exploration.

The metrics used for the evaluation of these algorithms were cumulative rewards, ROI, Sharpe ratio, Calmar ratio.

