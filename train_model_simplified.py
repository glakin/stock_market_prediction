
from fetch_functions import fetch_technicals, fetch_daily
import pandas as pd
import numpy as np
import time
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
    time.sleep(60)
if path.exists("./data/{}_daily_{}.csv".format(symbol, current_date)):
    prices = pd.read_csv("./data/{}_daily_{}.csv".format(symbol, current_date))
    prices = prices.set_index("date", drop = True)
else:
    prices = fetch_daily(symbol, save_csv = True)

# Build dataset

df = prices
df.index.names = ['date']
df = df.sort_index()

input_data = df.to_numpy()
days_history = len(input_data)
ncols = len(df.columns)

# Build a target dataset of the change in price between closes
close = input_data[:,3][:]
close_change = np.ndarray(shape = (days_history-1,1), dtype = float)
for i in range(days_history-1):
    close_change[i] = close[i+1]-close[i]
#close_change[days_history-1] = 0

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

y_test_predicted = scaler_target_test.inverse_transform(y_test_predicted)
y_test = scaler_target_test.inverse_transform(y_test)

dates = df.index[-len(y_test_predicted)+1:]
dates2 = df.index[-len(y_test_predicted):]

actual_close = close[-len(y_test_predicted)+1:] + y_test[:-1][1]
pred_close = close[-len(y_test_predicted)+1:] + y_test_predicted[:-1][1]

import matplotlib.pyplot as plt
plt.plot(dates, actual_close, label = 'actual')
plt.plot(dates, pred_close, label = 'predicted')
plt.legend(['Actual','Predicted'])

plt.plot(dates2, y_test, label = 'actual')
plt.plot(dates2, y_test_predicted, label = 'predicted')
plt.legend(['Actual','Predicted'])

print("The actual change over the time period is {}".format(sum(y_test)))
print("The predicted change over the time period is {}".format(sum(y_test_predicted)))

plt.scatter(y_test_predicted, y_test)

correct_sign = np.zeros(len(y_test))
for i in range(len(y_test)):
    if (y_test[i] > 0 and y_test_predicted[i] > 0) or (y_test[i] < 0 and y_test_predicted[i] < 0):
        correct_sign[i] = 1
    else:
        correct_sign[i] = 0
        
print("The model selects the correct direction {:%} of the time".format(np.mean(correct_sign)))

# Pickling isn't working right now. Found a potential workaround on stackoverflow
# but i'm leaving it for now 
# https://stackoverflow.com/questions/44855603/typeerror-cant-pickle-thread-lock-objects-in-seq2seq

# import pickle
# filename = "{}_model_{}".format(symbol, current_date)
# pickle.dump(model, open(filename,'wb'))
