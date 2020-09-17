from etl_functions import execute_query 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

symbol = 'LLY'

query = '''
    select p.date,
           p.symbol,
           p.open * p.adjusted_close/nullif(p.close,0) adjusted_open,
           p.high * p.adjusted_close/nullif(p.close,0) adjusted_high,
           p.low * p.adjusted_close/nullif(p.close,0) adjusted_low,
           p.adjusted_close,
           p.volume,
           t.sma_25,
           t.sma_50,
           t.ema_25,
           t.ema_50,
           t.rsi,
           t.slowK,
           t.slowD,
           t.adx,
           t.macd_hist,
           t.macd,
           t.macd_signal,
           t.real_upper_band - t.real_lower_band bband_width,
           t.aroon_up - t.aroon_down net_aroon,
           t.cci,
           t.chaikin_ad,
           t.obv,
           v.close as vix
           
    from prices_daily p
    join technicals_daily t on p.symbol = t.symbol and p.date = t.date
    left join earnings e on p.symbol = e.symbol and p.date = e.date
    left join indices_daily v on p.date = v.date and v.symbol = 'VXX'
    where e.date is null
    and p.symbol = '{}'
    order by p.symbol, p.date asc        
    '''.format(symbol)
    
cc_query = '''
    with prices as 
        (
          select symbol, 
              date, 
              adjusted_close, 
              row_number() over (partition by symbol order by date asc) rownum 
          from prices_daily  
        )
    
   	select p.date,
           p.symbol,
           p1.adjusted_close - p.adjusted_close close_change_1_day,
   		   (p1.adjusted_close - p.adjusted_close)/p.adjusted_close close_change_pct_1_day,
           p5.adjusted_close - p.adjusted_close close_change_5_day,
   		   (p5.adjusted_close - p.adjusted_close)/p.adjusted_close close_change_pct_5_day
   	from prices p
   	join prices p1 on p.symbol = p1.symbol and p1.rownum = p.rownum+1
    join prices p5 on p.symbol = p5.symbol and p5.rownum = p.rownum+5
   	left join earnings e on p.symbol = e.symbol and p.date = e.date
   	where e.date is null
    and p.symbol = '{}'
   	order by p.symbol, p.date asc      
    '''.format(symbol)

features = execute_query(query)
#features = features.set_index('date')
features = features.dropna()
#features = features.sort_index()
features = features.drop(features.tail(1).index)

min_date = min(features.date)

close_change = execute_query(cc_query)
#close_change = close_change.set_index('date')
#close_change.sort_index()   
close_change = close_change[close_change.date >= min_date]

df = pd.merge(features, close_change, on = ['symbol','date'])

df['sma_25_corrected'] = df['adjusted_close'].rolling(25).mean()
df['sma_50_corrected'] = df['adjusted_close'].rolling(50).mean()

# Define whether the stock went up or down 
df['up_down_1_day'] = np.where(df['close_change_pct_1_day'] > 0, 'up', 'down')
df['up_down_5_day'] = np.where(df['close_change_pct_5_day'] > 0, 'up', 'down')

df['close_sma25_delta'] = df['adjusted_close'] - df['sma_25_corrected']
df['close_ema25_delta'] = df['adjusted_close'] - df['ema_25']
df['sma_25_slope_1_day'] = df['sma_25_corrected'].diff()
df['sma_25_slope_5_day'] = df['sma_25_corrected'].diff(periods=5)/5
df['sma_25_slope_25_day'] = df['sma_25_corrected'].diff(periods=25)/25
df['sma_25_slope_direction_5_day'] = np.where(df['sma_25_slope_5_day'] > 0, 'up', 'down')

df.pivot_table(values = 'date',index = 'up_down_5_day', columns = 'sma_25_slope_direction_5_day', aggfunc = 'count')



plt.figure()
sns.distplot(df[feature])

plt.figure()
sns.violinplot(x = 'sma_25_slope_direction_5_day', y = 'close_change_pct_5_day', data = df)

plt.figure()
sns.jointplot(x = 'sma_25_slope_1_day', y = 'close_change_pct_5_day', data = df, kind = "reg")

plt.figure()
sns.jointplot(x = 'sma_25_slope_5_day', y = 'close_change_pct_5_day', data = df, kind = "reg")

plt.figure()
sns.jointplot(x = 'sma_25_slope_25_day', y = 'close_change_pct_5_day', data = df, kind = "reg")

plt.figure()
sns.jointplot(x = 'vix', y = 'close_change_pct_5_day', data = df, kind = "reg")