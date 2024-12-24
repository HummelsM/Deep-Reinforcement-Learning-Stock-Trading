from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    # Data Cleaning
    data.fillna(method='ffill', inplace=True)
    data.drop_duplicates(inplace=True)

    # Prevent zero or near-zero values in the 'Close' column
    data['Close'] = data['Close'].replace(0, 1e-6)

    # Feature Engineering
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = compute_rsi(data['Close'], 14)  # Assuming you have the compute_rsi function
    data['MACD'] = compute_macd(data['Close'])    # Assuming you have the compute_macd function
    data['Prev_Close'] = data['Close'].shift(1)

    # Noise Removal - Apply Savitzky-Golay Filter (Polynomial Smoothing)
    data['Close_Smooth'] = savgol_filter(data['Close'], window_length=11, polyorder=3)
    data['SMA_20_Smooth'] = savgol_filter(data['SMA_20'], window_length=11, polyorder=3)
    data['EMA_20_Smooth'] = savgol_filter(data['EMA_20'], window_length=11, polyorder=3)


    # Drop rows with NaN values (after rolling calculations)
    data.dropna(inplace=True)

    # Outlier Detection and Handling (using IQR method)
    for col in ['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD']:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

    # Normalize selected columns
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Prev_Close']])
    scaled_df = pd.DataFrame(scaled_features, columns=['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Prev_Close'])
    
    return scaled_df


def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    short_ema = series.ewm(span=12, adjust=False).mean()
    long_ema = series.ewm(span=26, adjust=False).mean()
    return short_ema - long_ema

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
