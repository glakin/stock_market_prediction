
from fetch_functions import fetch_technicals, fetch_daily, fetch_intraday, fetch_daily_adjusted, fetch_earnings
import pandas as pd
import numpy as np
import time
from os import path
from datetime import date, timedelta, datetime
from sklearn import preprocessing

current_date = date.today()
sample_days = 50

# Stock symbol to train the model with
symbol = "NFLX"

# Fetch technical data
# Write/read from csv to reduce API calls
if path.exists("./data/{}_technicals_{}.csv".format(symbol, current_date)):
    technicals = pd.read_csv("./data/{}_technicals_{}.csv".format(symbol, current_date))
    technicals = technicals.set_index("date", drop = True)
else:
    technicals = fetch_technicals(symbol, interval = "daily", save_csv = True)
    time.sleep(60)

# Fetch daily price data
if path.exists("./data/{}_daily_adjusted_{}.csv".format(symbol, current_date)):
    prices = pd.read_csv("./data/{}_daily_adjusted_{}.csv".format(symbol, current_date))
    prices['date'] = prices['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    prices = prices.set_index("date", drop = True)
else:
    prices = fetch_daily_adjusted(symbol, save_csv = True)
    
# Fetch earnings data
if path.exists("./data/{}_earnings.csv".format(symbol)):
    earnings = pd.read_csv("./data/{}_earnings.csv".format(symbol))
    if datetime.strptime(max(earnings['refresh_date']), '%Y-%m-%d').date() < (current_date - timedelta(days=30)):
        earnings = fetch_earnings(symbol, save_csv = True)
else:
    earnings = fetch_earnings(symbol, save_csv = True)
    
print("Contains data through {}".format(max(prices.index)))

#Process columns
prices_cols = prices.columns
for col in prices_cols[0:4]:
    prices[col] = prices[col] * prices["5. adjusted close"]/prices["4. close"]
prices = prices.drop(["5. adjusted close", "7. dividend amount", "8. split coefficient"], axis=1)

technicals["bband_width"] = technicals["Real Upper Band"] - technicals["Real Lower Band"]
technicals = technicals.drop(["Real Upper Band", "Real Middle Band", "Real Lower Band"], axis=1)

earnings_date_strings = earnings['startdatetime'].to_list()
earnings_dates = [datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S.000Z').date()  for dt in earnings_date_strings]

# Build dataset
df = pd.merge(prices, technicals, left_index=True, right_index=True)
#df = df.drop(['MACD','MACD_Signal','Real Middle Band', 'Real Lower Band', 
#              'Real Upper Band'], axis=1)


df.index.names = ['date']
df = df.dropna()
df = df.sort_index()

earnings_idx = []
indices = df.index
for i in range(len(indices)):
    if indices[i] in earnings_dates:
        earnings_idx.append(i)

input_data = df.to_numpy()

days_history = len(input_data)
ncols = len(df.columns)

input_data = np.delete(input_data, earnings_idx, axis=0)


# Build a target dataset of the change in price between closes
#close = input_data[:,3][:]
close = df['4. close'].to_numpy()
close_change = np.ndarray(shape = (days_history-1,1), dtype = float)
for i in range(days_history-1):
    close_change[i] = close[i+1]-close[i]
#close_change[days_history-1] = 0

close = np.delete(close, earnings_idx, axis=0)
close_change = np.delete(close_change, earnings_idx, axis=0)

#Set the training/test split
test_split = 0.9 
n = int(input_data.shape[0] * test_split)

input_train = input_data[:n,:]
input_test = input_data[n:-1,:]

close_change_train = close_change[:n]
close_change_test = close_change[n:]

# Create scalers for train/test subsets of input/target data
scaler_input_train = preprocessing.StandardScaler()
scaler_input_test = preprocessing.StandardScaler()
scaler_target_train = preprocessing.StandardScaler()
scaler_target_test = preprocessing.StandardScaler()

# Normalize the data
input_train_norm = scaler_input_train.fit_transform(input_train)
input_test_norm = scaler_input_test.fit_transform(input_test)
target_train_norm = scaler_target_train.fit_transform(close_change_train)
target_test_norm = scaler_target_test.fit_transform(close_change_test)

# Split the input and target data into groupings of size sample_days
X_train = np.array([input_train_norm[i:i + sample_days].copy() for i in range(len(input_train_norm) - sample_days)])
X_test = np.array([input_test_norm[i:i + sample_days].copy() for i in range(len(input_test_norm) - sample_days)])

y_train = np.array([target_train_norm[i + sample_days].copy() for i in range(len(target_train_norm) - sample_days)])
#y_train = np.expand_dims(y_train, -1)
y_test = np.array([target_test_norm[i + sample_days].copy() for i in range(len(target_test_norm) - sample_days)])
#y_test = np.expand_dims(y_test, -1)

# Model
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(4)
#from tensorflow import set_random_seed
#set_random_seed(4)

lstm_input = Input(shape = (sample_days, ncols), name = 'lstm_input')
x = LSTM(50, name = 'lstm_0')(lstm_input)
x = Dropout(0.2, name = 'lstm_dropout_0')(x)
x = Dense(64, name = 'dense_0')(x)
x = Activation('sigmoid', name = 'sigmoid_0')(x)
x = Dense(1, name = 'dense_1')(x)
output = Activation('linear', name = 'linear_output')(x)
model = Model(inputs = lstm_input, outputs = output)

adam = optimizers.Adam(lr = 0.0005)

model.compile(optimizer = adam, loss = 'mse')

### This step isn't working. Need ot install graphviz (https://graphviz.gitlab.io/download/)
#from keras.utils import plot_model
#plot_model(model, to_file = 'model.png')

model.fit(x = X_train, y = y_train, batch_size = 32, epochs = 50, shuffle = True, validation_split = 0.1)
evaluation = model.evaluate(X_test, y_test)
print(evaluation)

y_test_predicted = model.predict(X_test)

change_test_predicted = scaler_target_test.inverse_transform(y_test_predicted)
change_test = scaler_target_test.inverse_transform(y_test)

dates_total =  [x for x in df.index.to_list() if x.date() not in earnings_dates]
dates = dates_total[n+sample_days+1:]
dates2 = dates_total[n+sample_days:-1]

#actual_close = close[n+sample_days:-1] + change_test[:-1][1]
actual_close = close[n+sample_days+1:]
pred_close = close[n+sample_days:-1] + change_test_predicted[:-1][1]
total_open = df['1. open'].to_numpy()
total_open = np.delete(total_open, earnings_idx, axis=0)
actual_open = total_open[n+sample_days+1:]

import matplotlib.pyplot as plt
plt.figure(0)
plt.plot(dates, actual_close, label = 'actual')
plt.plot(dates, pred_close, label = 'predicted')
plt.legend(['Actual','Predicted'])

plt.figure(1)
plt.plot(dates2, y_test, label = 'actual')
plt.plot(dates2, y_test_predicted, label = 'predicted')
plt.legend(['Actual','Predicted'])
print("")
print("The test period covers {} between {} and {}".format(symbol, dates[0], dates[-1]))
print("")
print("The actual change over the time period is {}".format(sum(y_test)))
print("The predicted change over the time period is {}".format(sum(y_test_predicted)))
print("")

correct_sign = np.zeros(len(y_test))
for i in range(len(y_test)):
    if (y_test[i] > 0 and y_test_predicted[i] > 0) or (y_test[i] < 0 and y_test_predicted[i] < 0):
        correct_sign[i] = 1
    else:
        correct_sign[i] = 0
        
print("The model selects the correct direction {:%} of the time".format(np.mean(correct_sign)))
print("")

pred_change = y_test_predicted[:]

b = 100
b1 = 100
b05 = 100
b02 = 100
b01 = 100
b0 = 100

n1 = 0
n05 = 0
n02 = 0
n01 = 0
n0 = 0

d1 = []
d05 = []
d02 = []
d01 = []
d0 = []

c1 = []
c05 = []
c02 = []
c01 = []
c0 = []

for i in range(len(actual_close)-1):
    if pred_change[i] >= actual_close[i]*0.01 and actual_open[i+1]-actual_close[i] <= pred_change[i]*0.5:
        b1 = b1*actual_close[i+1]/actual_open[i+1]
        n1 += 1
        d1.append(dates2[i])
        c1.append(actual_close[i])
    if pred_change[i] >= actual_close[i]*0.005 and actual_open[i+1]-actual_close[i] <= pred_change[i]*0.5:
        b05 = b05*actual_close[i+1]/actual_open[i+1]
        n05 += 1
        d05.append(dates2[i])
        c05.append(actual_close[i])
    if pred_change[i] >= actual_close[i]*0.002 and actual_open[i+1]-actual_close[i] <= pred_change[i]*0.5:
        b02 = b02*actual_close[i+1]/actual_open[i+1]
        n02 += 1
        d02.append(dates2[i])
        c02.append(actual_close[i])
    if pred_change[i] >= actual_close[i]*0.001 and actual_open[i+1]-actual_close[i] <= pred_change[i]*0.5:
        b01 = b01*actual_close[i+1]/actual_open[i+1]
        n01 += 1
        d01.append(dates2[i])
        c01.append(actual_close[i])
    if pred_change[i] >= 0 and actual_open[i+1]-actual_close[i] <= pred_change[i]*0.5:
        b0 = b0*actual_close[i+1]/actual_open[i+1]
        n0 += 1
        d0.append(dates2[i])
        c0.append(actual_close[i])
    b = b*actual_close[i+1]/actual_open[i+1]
    
        
bhold = 100 * actual_close[-1]/actual_close[0]

print("BUYING AND SELLING AFTER 1 DAY")        
print("With a threshold of 1% you traded {} times and finished with {:%} of your starting amount".format(n1,b1/100))
print("With a threshold of 0.5% you traded {} times and finished with {:%} of your starting amount".format(n05,b05/100))
print("With a threshold of 0.2% you traded {} times and finished with {:%} of your starting amount".format(n02,b02/100))
print("With a threshold of 0.1% you traded {} times and finished with {:%} of your starting amount".format(n01,b01/100))
print("With a threshold of 0% you traded {} times and finished with {:%} of your starting amount".format(n0,b0/100))
print("By buying and holding you finished with {:%} of your starting amount".format(bhold/100))
print("")

# Plot the trades
plt.figure(2)
plt.plot(dates, actual_close)
plt.scatter(d0, c0, s=1.5, c="red")
# Pickling isn't working right now. Found a potential workaround on stackoverflow
# but i'm leaving it for now 
# https://stackoverflow.com/questions/44855603/typeerror-cant-pickle-thread-lock-objects-in-seq2seq

# import pickle
# filename = "{}_model_{}".format(symbol, current_date)
# pickle.dump(model, open(filename,'wb'))
