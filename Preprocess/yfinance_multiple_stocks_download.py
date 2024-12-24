import pandas as pd
import yfinance as yf
from datetime import datetime as dt
from datetime import date,timedelta
import matplotlib as plt
import numpy as np
#from psx import stocks, tickers

#tickers=tickers()

stock_list =['TSLA','AAPL','GOOGL','AMZN','MSFT']
#stock_list =['HBL','EnGro','SILK']


period="1d"
interval = "5m"

stock_list_df = pd.DataFrame()
stock_list_df={}

"""for i in stock_list:
    #item = i+ ".NS"
    stock_list_df[i] = yf.download(tickers=i,period=period,interval=interval)
    stock_list_df[i]['Date']=stock_list_df[i].index
    #stock=str(stock_list_df[i]['Date'])
    #np.datetime64 (dt.strptime(stock, '%Y%m%d %H:%M:%S').date())
    stock_list_df[i]=stock_list_df[i][["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    stock_list_df[i].reset_index(drop=True,inplace=True)
    stock_list_df[i].to_csv("GOOGL.csv",index=False)"""
    

#print(stock_list_df['TSLA','GOOGL'].round(2))


stock = yf.download(tickers='MSFT',period=period,interval=interval)
stock['Date']=stock.index
#stock=str(stock_list_df[i]['Date'])
#np.datetime64 (dt.strptime(stock, '%Y%m%d %H:%M:%S').date())
stock=stock[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
stock.reset_index(drop=True,inplace=True)
stock.to_csv("AAPL.csv",index=False)
