import pandas as pd
from refresh_functions import refresh_data, refresh_models

symbols = pd.read_csv('./data/symbols.csv')['symbol'].to_list()
refresh_data(symbols)
refresh_models(symbols, offset_days = 5)