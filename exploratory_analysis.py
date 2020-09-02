from etl_functions import execute_query 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

symbol = 'KO'

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
           t.obv
           
    from prices_daily p
    join technicals_daily t on p.symbol = t.symbol and p.date = t.date
    left join earnings e on p.symbol = e.symbol and p.date = e.date
    where e.date is null
    and p.symbol = '{}'
    order by p.symbol, p.date asc        
    '''.format(symbol)
    
cc_query = '''
   	select p.date,
           p.symbol,
   		   (p2.adjusted_close - p.adjusted_close)/p.adjusted_close close_change_pct               
   	from (select symbol, date, adjusted_close, row_number() over (partition by symbol order by date asc) rownum from prices_daily) p
   	join (select symbol, date, adjusted_close, row_number() over (partition by symbol order by date asc) rownum from prices_daily) p2 on p.symbol = p2.symbol and p2.rownum = p.rownum+1
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

df['up_down'] = np.where(df['close_change_pct'] > 0, 'up', 'down')
df['close_sma25_delta'] = df['adjusted_close'] - df['sma_25']
df['close_ema25_delta'] = df['adjusted_close'] - df['ema_25']

feature = 'net_aroon'

plt.figure()
sns.distplot(df[feature])

plt.figure()
sns.jointplot(x = feature, y = 'close_change_pct', data = df, kind = "hex")

plt.figure()
sns.violinplot(x = "up_down", y = feature, data = df)

