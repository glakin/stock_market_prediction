
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

input_data = df.to_numpy()
target = np.array([input_data[:,3][i + sample_days].copy() for i in range(len(input_data) - sample_days)])
target = np.expand_dims(target, -1)

test_split = 0.9 
n = int(input_data.shape[0] * test_split)

input_train = input_data[:n,:]
input_test = input_data[n:,:]

target_train = target[:n]
target_test = target[n:]

scaler_input_train = preprocessing.StandardScaler()
scaler_input_test = preprocessing.StandardScaler()

input_train_norm = scaler_input_train.fit_transform(input_train)
input_test_norm = scaler_input_test.fit_transform(input_test)

# using the last {sample_days} open high low close volume data points, predict the next close value
history_train = np.array([input_train_norm[i  : i + sample_days].copy() for i in range(len(input_train_norm) - sample_days)])
history_test = np.array([input_test_norm[i  : i + sample_days].copy() for i in range(len(input_test_norm) - sample_days)])

y_train_scaler = preprocessing.StandardScaler()
y_train_scaler.fit(target_train)
y_train_norm = y_train_scaler.transform(target_train)

y_test_scaler = preprocessing.StandardScaler()
y_test_scaler.fit(target_test)
y_test_norm = y_test_scaler.transform(target_test) 

# assert history_train.shape[0] == y_train_norm.shape[0]

