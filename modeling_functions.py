# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:44:34 2020

@author: Jerry
"""
def train_model(symbol, analyze = True):
    from etl_functions import execute_query 
    from fetch_functions import fetch_earnings
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    import keras
    import tensorflow as tf
    from keras.models import Model
    from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
    from keras import optimizers
    
    
    sample_days = 50
    
    query = '''
        select p.date,
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
        where p.symbol = '{}'
        and e.date is null
        order by p.symbol, p.date asc        
        '''.format(symbol)
        
    cc_query = '''
       	select p.date,
       		   p2.adjusted_close - p.adjusted_close close_change               
       	from (select symbol, date, adjusted_close, row_number() over (partition by symbol order by date asc) rownum from prices_daily) p
       	join (select symbol, date, adjusted_close, row_number() over (partition by symbol order by date asc) rownum from prices_daily) p2 on p.symbol = p2.symbol and p2.rownum = p.rownum+1
       	left join earnings e on p.symbol = e.symbol and p.date = e.date
       	where p.symbol = '{}' 
       	and e.date is null
       	order by p.symbol, p.date asc      
        '''.format(symbol)
    
    df = execute_query(query)
    df = df.set_index('date')
    df = df.dropna()
    df = df.sort_index()
    df = df.drop(df.tail(1).index)
    
    min_date = min(df.index)
    ncols = len(df.columns)
    
    close_change = execute_query(cc_query)
    close_change = close_change.set_index('date')
    close_change.sort_index()   
    close_change = close_change[close_change.index >= min_date]

    X = df.to_numpy()
    y = close_change['close_change'].to_numpy()

    # days_history = X.shape[0]

    #Set the training/test split
    test_split = 0.9 
    n = int(X.shape[0] * test_split)
    
    X_train_input = X[:n,:]
    X_test_input = X[n:,:]
    
    y_train_input = y[:n].reshape(-1,1)
    y_test_input = y[n:].reshape(-1,1)
    
    # Create scalers for train/test subsets of input/target data
    scaler_X_train = preprocessing.StandardScaler()
    scaler_X_test = preprocessing.StandardScaler()
    scaler_y_train = preprocessing.StandardScaler()
    scaler_y_test = preprocessing.StandardScaler()
    
    # Normalize the data
    scaler_X_train.fit(X_train_input)
    X_train_input_norm = scaler_X_train.transform(X_train_input)
    
    # Use scaler for X_test
    X_test_input_norm = scaler_X_test.fit_transform(X_test_input)
    
    # Use X_train scaler
    #X_test_input_norm = scaler_X_train.transform(X_test_input)
    
    # Scaled y
    #y_train_input_norm = scaler_y_train.fit_transform(y_train_input)
    #y_test_input_norm = scaler_y_test.fit_transform(y_test_input)
    
    # Unscaled y
    y_train_input_norm = y_train_input
    y_test_input_norm = y_test_input
    
    # NOTE: the more I think about it the more I think that we should probably
    # wait to normalize until after the following step, where the X and y data
    # are split into sample_days-sized windows. If each of these windows
    # was normalized with itself then the stock's early days wouldn't be so much
    # lower than the more recent values. Not sure what the best practice is
    
    # Or maybe it would just be better to replace close change with % change,
    # the percent the stock moves compared to yesterday's close
    
    # Split the input and target data into groupings of size sample_days
    X_train = np.array([X_train_input_norm[i:i + sample_days].copy() for i in range(len(X_train_input_norm) - sample_days)])
    X_test = np.array([X_test_input_norm[i:i + sample_days].copy() for i in range(len(X_test_input_norm) - sample_days)])

    y_train = np.array([y_train_input_norm[i + sample_days].copy() for i in range(len(y_train_input_norm) - sample_days)])
    y_test = np.array([y_test_input_norm[i + sample_days].copy() for i in range(len(y_test_input_norm) - sample_days)])
    
    dates_train = df.index[sample_days:n].to_list()
    dates_test = df.index[n+sample_days:].to_list()
    
    # Model

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
    
    model.fit(x = X_train, y = y_train, batch_size = 32, epochs = 100, 
              shuffle = True, validation_split = 0.1, verbose = 1)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    
    if analyze == True:                
        import matplotlib.pyplot as plt
        import os
        import json
        import seaborn as sns
        
        if not os.path.exists("./analysis/{}/".format(symbol)):
            p = "./analysis/{}/".format(symbol)
            os.mkdir(p)
        
        y_test_predicted = model.predict(X_test)
        # change_test_predicted = pd.Series(data = scaler_y_test.inverse_transform(y_test_predicted)[:,0], 
        #                                   index = dates_test, 
        #                                   name = 'close_change')
        # change_test = pd.Series(data = scaler_y_test.inverse_transform(y_test)[:,0], 
        #                         index = dates_test, 
        #                         name = 'close_change')
        change_test_predicted = pd.Series(data = y_test_predicted[:,0], 
                                          index = dates_test, 
                                          name = 'close_change')
        change_test = pd.Series(data = y_test[:,0], 
                                index = dates_test, 
                                name = 'close_change')
        close_test_predicted = pd.Series(data = df.adjusted_close.to_numpy()[n+sample_days:-1] + change_test_predicted.to_list()[:-1], 
                               index = dates_test[:-1],
                               name = 'predicted_close')
        close_test = pd.Series(data = df.adjusted_close.to_numpy()[n+sample_days:], 
                                 index = dates_test[:],
                                 name = 'actual_close')

        #plt.figure(0)
        #close.plot()
        #close_predicted.plot()        
        #plt.legend(['Actual', 'Predicted'])
        #plt.savefig()
        
        plt.figure()
        change_test.plot()
        change_test_predicted.plot()
        plt.legend(['Actual', 'Predicted'])
        plt.savefig('./analysis/{}/predicted_change.png'.format(symbol))
        plt.close()
        
        fig1 = plt.figure()
        ax1 = sns.scatterplot(x = change_test, y = change_test_predicted)
        ax1.set_ylabel('Predicted Close Change')
        ax1.set_xlabel('Actual Close Change')
        plt.savefig('./analysis/{}/predicted_vs_actual_scatter.png'.format(symbol))
        plt.close()        
        
        error = change_test - change_test_predicted
        fig2 = plt.figure()
        ax2 = sns.distplot(error)
        ax2.set_xlabel('Close Change Error')
        ax2.set_ylabel('Probability Distribution')
        plt.savefig('./analysis/{}/predicted_vs_actual_scatter.png'.format(symbol))
        plt.close()
        
        from sklearn.metrics import mean_absolute_error,mean_squared_error
        mae = mean_absolute_error(change_test, change_test_predicted)
        mse = mean_squared_error(change_test, change_test_predicted)
        
        correct_sign = np.zeros(len(change_test))
        for i in range(len(change_test)):
            if (change_test[i] > 0 and change_test_predicted[i] > 0) or (change_test[i] < 0 and change_test_predicted[i] < 0):
                correct_sign[i] = 1
            else:
                correct_sign[i] = 0
                
        correct_sign_pct = np.mean(correct_sign)
        
        open_test = pd.Series(data = df.adjusted_open[n+sample_days:-1].to_numpy(), index = dates_test[:-1])
                
        b = 1
        b1 = 1
        b05 = 1
        b02 = 1
        b01 = 1
        b0 = 1
        
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
        
        for i in range(len(close_test)-2):
            if change_test_predicted[i] >= close_test[i]*0.01 and open_test[i+1]-close_test[i] <= change_test_predicted[i]*0.5:
                b1 = b1*close_test[i+1]/open_test[i+1]
                n1 += 1
                d1.append(dates_test[i])
                c1.append(close_test[i])
            if change_test_predicted[i] >= close_test[i]*0.005 and open_test[i+1]-close_test[i] <= change_test_predicted[i]*0.5:
                b05 = b05*close_test[i+1]/open_test[i+1]
                n05 += 1
                d05.append(dates_test[i])
                c05.append(close_test[i])
            if change_test_predicted[i] >= close_test[i]*0.002 and open_test[i+1]-close_test[i] <= change_test_predicted[i]*0.5:
                b02 = b02*close_test[i+1]/open_test[i+1]
                n02 += 1
                d02.append(dates_test[i])
                c02.append(close_test[i])
            if change_test_predicted[i] >= close_test[i]*0.001 and open_test[i+1]-close_test[i] <= change_test_predicted[i]*0.5:
                b01 = b01*close_test[i+1]/open_test[i+1]
                n01 += 1
                d01.append(dates_test[i])
                c01.append(close_test[i])
            if change_test_predicted[i] >= 0 and open_test[i+1]-close_test[i] <= change_test_predicted[i]*0.5:
                b0 = b0*close_test[i+1]/open_test[i+1]
                n0 += 1
                d0.append(dates_test[i])
                c0.append(close_test[i])
            b = b*close_test[i+1]/open_test[i+1]            
                
        bhold = close_test[-1]/close_test[0]
        
        # Plot the trades
        plt.figure()
        close_test.plot(title = 'Timing of Trades using Threshold of 0%')
        plt.scatter(d0, c0, s=2, c="red")
        plt.savefig('./analysis/{}/trades_threshold_0%.png'.format(symbol))
        plt.close()

        plt.figure()
        close_test.plot(title = 'Timing of Trades using Threshold of 0.1%')
        plt.scatter(d01, c01, s=2, c="red")
        plt.savefig('./analysis/{}/trades_threshold_0.1%.png'.format(symbol))
        plt.close()
        
        plt.figure()
        close_test.plot(title = 'Timing of Trades using Threshold of 0.2%')
        plt.scatter(d02, c02, s=2, c="red")
        plt.savefig('./analysis/{}/trades_threshold_0.2%.png'.format(symbol))
        plt.close()
             
        plt.figure()
        close_test.plot(title = 'Timing of Trades using Threshold of 0.5%')
        plt.scatter(d05, c05, s=2, c="red")
        plt.savefig('./analysis/{}/trades_threshold_0.5%.png'.format(symbol))    
        plt.close()    
    
        plt.figure()
        close_test.plot(title = 'Timing of Trades using Threshold of 1%')
        plt.scatter(d1, c1, s=2, c="red")
        plt.savefig('./analysis/{}/trades_threshold_1%.png'.format(symbol))
        plt.close()    
    
        analyze_dict = {
            "test_loss": test_loss,
            "train_loss": train_loss,
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "correct_sign_pct": correct_sign_pct,
            "return_buy_hold": bhold,
            "return_threshold_0_pct": b0,
            "return_threshold_0.1_pct": b01,
            "return_threshold_0.2_pct": b02,
            "return_threshold_0.5_pct": b05,
            "return_threshold_1_pct": b1,
            "trades_0_pct": n0,
            "trades_0.1_pct": n01,
            "trades_0.2_pct": n02,
            "trades_0.5_pct": n05,
            "trades_1_pct": n1            
            }
        
        with open('./analysis/{}/{}_analysis.txt'.format(symbol, symbol), 'w') as outfile:
            json.dump(analyze_dict, outfile)      
    
    # model.save('{}_daily_model.h5'.format(symbol))
    
    
    return model
        
    