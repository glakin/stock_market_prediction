import pandas as pd
import keyring as k
import time
from fetch_functions import fetch_daily_adjusted, fetch_earnings, fetch_daily_technicals
from sqlalchemy import create_engine

def update_daily_prices(symbols):
    
    host = 'localhost'
    database = 'stocks'
    
    engine_string = 'mysql+pymysql://{}:{}@{}/{}'.format(k.get_password("mysql","username"), k.get_password("mysql","password"), host, database)
    engine = create_engine(engine_string)
    del(engine_string)
    
    cols = ['symbol',
            'date',
            'open',
            'high',
            'low',
            'close',
            'adjusted_close',
            'volume',
            'dividend_amount',
            'split_coefficient']
    
    dcols = ['date',
            'open',
            'high',
            'low',
            'close',
            'adjusted_close',
            'volume',
            'dividend_amount',
            'split_coefficient']  
    
    prices = pd.DataFrame(columns = cols)    
    
    for symbol in symbols:
        try:
            df = fetch_daily_adjusted(symbol)
        except:
            time.sleep(60)
            df = fetch_daily_adjusted(symbol)
        df = df.reset_index()
        
      
        df.columns = dcols        
        df['symbol']= symbol
        prices = prices.append(df)
    
    table = "prices_daily"
    
    prices.to_sql(table, con = engine, if_exists='replace')
  
def update_earnings(symbols):
    
    host = 'localhost'
    database = 'stocks'
    
    engine_string = 'mysql+pymysql://{}:{}@{}/{}'.format(k.get_password("mysql","username"), k.get_password("mysql","password"), host, database)
    engine = create_engine(engine_string)
    del(engine_string)
    
    cols = ['symbol', 
            'companyshortname', 
            'startdatetime', 
            'startdatetimetype',
            'epsestimate',
            'epsactual', 
            'epssurprisepct', 
            'timeZoneShortName',
            'gmtOffsetMilliSeconds', 
            'quoteType']
    
    earnings = pd.DataFrame(columns = cols)    
    
    for symbol in symbols:
        try:
            df = fetch_earnings(symbol)
        except:
            time.sleep(60)
            df = fetch_earnings(symbol)     
        df['symbol']= symbol
        earnings = earnings.append(df)
    
    table = "earnings"
    
    earnings.to_sql(table, con = engine, if_exists='replace')
    
def update_daily_technicals(symbols):
    
    host = 'localhost'
    database = 'stocks'
    
    engine_string = 'mysql+pymysql://{}:{}@{}/{}'.format(k.get_password("mysql","username"), k.get_password("mysql","password"), host, database)
    engine = create_engine(engine_string)
    del(engine_string)
    
    cols = ['symbol',
            'date',
            'SMA_25',
            'SMA_50',
            'EMA_25',
            'EMA_50',
            'RSI',
            'SlowK',
            'SlowD',
            'ADX',
            'MACD_Hist',
            'MACD',
            'MACD_Signal',
            'Real_Lower_Band',
            'Real_Middle_Band',
            'Real_Upper_Band',
            'Aroon_Down',
            'Aroon_Up',
            'CCI',
            'Chaikin_AD',
            'OBV']
    
    dcols = ['date',
            'SMA_25',
            'SMA_50',
            'EMA_25',
            'EMA_50',
            'RSI',
            'SlowK',
            'SlowD',
            'ADX',
            'MACD_Hist',
            'MACD',
            'MACD_Signal',
            'Real_Lower_Band',
            'Real_Middle_Band',
            'Real_Upper_Band',
            'Aroon_Down',
            'Aroon_Up',
            'CCI',
            'Chaikin_AD',
            'OBV']

    technicals = pd.DataFrame(columns = cols)
    
    for symbol in symbols:
        try:
            df = fetch_daily_technicals(symbol)
        except:
            time.sleep(60)
            df = fetch_daily_technicals(symbol)     
        df.columns = dcols
        df['symbol']= symbol
        technicals = technicals.append(df)
        
    table = "technicals_daily"
    
    technicals.to_sql(table, con = engine, if_exists='replace')
