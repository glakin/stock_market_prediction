import pandas as pd
from etl_functions import update_daily_prices, update_earnings, update_daily_technicals
from datetime import datetime
import time

symbols = pd.read_csv('./data/symbols.csv')

print('{} - Updating daily prices'.format(datetime.now().strftime("%H:%M:%S")))
update_daily_prices(symbols)
print('{} - Finished updating prices'.format(datetime.now().strftime("%H:%M:%S")))
time.sleep(60)
print('{} - Updating daily technicals'.format(datetime.now().strftime("%H:%M:%S")))
update_daily_technicals(symbols)
print('{} - Finished updating technicals'.format(datetime.now().strftime("%H:%M:%S")))
print('{} - Updating earnings'.format(datetime.now().strftime("%H:%M:%S")))
update_earnings(symbols)


