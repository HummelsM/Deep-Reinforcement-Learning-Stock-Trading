
import yfinance as yf
import pandas as pd

tickers=['UBER']
#tickers=['AAPL','GOOGL','TSLA']
#tickers=pd.DataFrame
#data=yf.download(tickers, start='2024-01-01',end='2024-05-31')
#tickers = yf.Tickers('msft aapl goog')

for ticker in tickers:
    data=yf.download(ticker, start='2011-01-03',end='2019-12-29',interval='1d')
    data.to_csv(ticker +'2'+'.csv')
    
