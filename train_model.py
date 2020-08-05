
from fetch_functions import fetch_technicals, fetch_daily
import pandas as pd
import numpy as np
from os import path
from datetime import date
from sklearn import preprocessing

current_date = date.today()
sample_days = 50

# Stock symbol to train the model with
symbol = "NFLX"

# Load technicals and price data
# Write/read from csv to reduce API calls
if path.exists("./data/{}_technicals_{}.csv".format(symbol, current_date)):
    technicals = pd.read_csv("./data/{}_technicals_{}.csv".format(symbol, current_date))
else:
    technicals = fetch_technicals(symbol, interval = "daily", save_csv = True)

if path.exists("./data/{}_daily_{}.csv".format(symbol, current_date)):
    prices = pd.read_csv("./data/{}_daily_{}.csv".format(symbol, current_date))
else:
    prices = fetch_daily(symbol, save_csv = True)

# Build dataset
df = pd.merge(prices, technicals, left_index=True, right_index=True).set_index('date_x', drop=True).drop('date_y', axis=1)
df.index.names = ['date']

# Normalize the data NEED TO MOVE THIS TO AFTER TRAINING/TEST SPLIT
scaler = preprocessing.StandardScaler()
norm_input = scaler.fit_transform(df)

# using the last {sample_days} open high low close volume data points, predict the next open value
norm_history = np.array([norm_input[i  : i + sample_days].copy() for i in range(len(norm_input) - sample_days)])
norm_target = np.array([norm_input[:,0][i + sample_days].copy() for i in range(len(norm_input) - sample_days)])
norm_target = np.expand_dims(norm_target, -1)

next_day_open_values = np.array([df[:,0][i + sample_days].copy() for i in range(len(df) - sample_days)])
next_day_open_values = np.expand_dims(norm_target, -1)

y_normaliser = preprocessing.MinMaxScaler()
y_normaliser.fit(np.expand_dims( next_day_open_values ))