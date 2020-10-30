import pandas as pd
from refresh_functions import refresh_data, refresh_models

# Read symbol list
symbols = pd.read_csv('./data/symbols.csv')['symbol'].to_list()

# Refresh the price and earnings data in the database
refresh_data(symbols)

# Refresh models with new data. Configure parameter offset_days to change
# which price we are trying to predict
refresh_models(symbols, offset_days = 5)