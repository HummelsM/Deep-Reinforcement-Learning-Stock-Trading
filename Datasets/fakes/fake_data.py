
import numpy as np
import pandas as pd

def generate_sine_wave_data(length, amplitude=1, frequency=0.05, noise=0.02):
    time = np.arange(length)
    # Generate sine wave and add Gaussian noise
    prices = amplitude * np.sin(2 * np.pi * frequency * time) + np.random.normal(0, noise, length)
    # Shift the prices to make sure they are positive
    prices = prices + amplitude + 1  # Shift up by (amplitude + 1) to ensure all prices are positive
    # Apply a minimum threshold to avoid extremely low values
    prices = np.clip(prices, 0.01, None)
    
    data = pd.DataFrame(prices, columns=['Close'])
    data['Date'] = pd.date_range(start='2022-01-01', periods=length, freq='D')
    return data.set_index('Date')




def generate_mean_reverting_random_walk(length, mean=100, speed=0.1, noise=1):
    prices = [mean]
    for _ in range(1, length):
        # Mean-reverting process with noise
        drift = speed * (mean - prices[-1])
        shock = np.random.normal(0, noise)
        prices.append(prices[-1] + drift + shock)
    data = pd.DataFrame(prices, columns=['Close'])
    data['Date'] = pd.date_range(start='2022-01-01', periods=length, freq='D')
    return data.set_index('Date')


def generate_brownian_motion(length, start_price=100, volatility=1):
    prices = [start_price]
    for _ in range(1, length):
        shock = np.random.normal(0, volatility)
        prices.append(prices[-1] + shock)
    data = pd.DataFrame(prices, columns=['Close'])
    data['Date'] = pd.date_range(start='2022-01-01', periods=length, freq='D')
    return data.set_index('Date')

def generate_synthetic_data(start_price, days, trend='bullish', volatility=0.02):
    """
    Generates synthetic stock price data for bullish, bearish, or volatile markets.

    Parameters:
        start_price (float): Starting price of the stock.
        days (int): Number of days to simulate.
        trend (str): Type of market ('bullish', 'bearish', 'volatile').
        volatility (float): Daily price volatility.

    Returns:
        pd.DataFrame: Synthetic stock price data with columns ['Day', 'close'].
    """
    prices = [start_price]
    for _ in range(days):
        if trend == 'bullish':
            daily_return = 0.001 + np.random.normal(0, volatility)  # Upward trend
        elif trend == 'bearish':
            daily_return = -0.001 + np.random.normal(0, volatility)  # Downward trend
        elif trend == 'volatile':
            daily_return = np.random.normal(0, volatility * 2)  # Random large moves
        else:
            raise ValueError("Trend must be 'bullish', 'bearish', or 'volatile'")

        prices.append(prices[-1] * (1 + daily_return))

    return pd.DataFrame({'Day': range(len(prices)), 'Close': prices})



# Generate datasets
bullish_data = generate_synthetic_data(start_price=100, days=200, trend='bullish')
bearish_data = generate_synthetic_data(start_price=100, days=200, trend='bearish')
volatile_data = generate_synthetic_data(start_price=100, days=200, trend='volatile')
