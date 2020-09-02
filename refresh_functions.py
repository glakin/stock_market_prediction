import pandas as pd
from etl_functions import update_daily_prices, update_earnings, update_daily_technicals
from modeling_functions import train_model
from datetime import datetime, date
import time
import json
import seaborn as sns
from matplotlib import pyplot as plt
import os

def refresh_data(symbols):
    
    print('{} - Updating daily prices'.format(datetime.now().strftime("%H:%M:%S")))
    update_daily_prices(symbols)
    print('{} - Finished updating prices'.format(datetime.now().strftime("%H:%M:%S")))
    time.sleep(60)
    print('{} - Updating daily technicals'.format(datetime.now().strftime("%H:%M:%S")))
    update_daily_technicals(symbols)
    print('{} - Finished updating technicals'.format(datetime.now().strftime("%H:%M:%S")))
    print('{} - Updating earnings'.format(datetime.now().strftime("%H:%M:%S")))
    update_earnings(symbols)
    print('{} - Finished refreshing data'.format(datetime.now().strftime("%H:%M:%S")))

def refresh_models(symbols):

    current_date = date.today()
    
    print('{} - Training models'.format(datetime.now().strftime("%H:%M:%S")))
    for symbol in symbols:
        print('{} - Modeling {}'.format(datetime.now().strftime("%H:%M:%S"), symbol))
        train_model(symbol)
    
    loss = []
    train_loss = []
    mae = []
    sign = []
    buy_hold = []
    thresh_0 = []
    thresh_0_1 = []
    thresh_0_2 = []
    thresh_0_5 = []
    thresh_1 = []
    trades_0 = []
    trades_0_1 = []
    trades_0_2 = []
    trades_0_5 = []
    trades_1 = []
    
    for symbol in symbols:
        path = "./analysis/{}/{}".format(symbol, current_date)
        j = json.load(open('{}/{}_analysis.txt'.format(path,symbol),'r'))
        loss.append(j['test_loss'])
        train_loss.append(j['train_loss'])
        mae.append(j['mean_absolute_error'])
        sign.append(j['correct_sign_pct'])
        buy_hold.append(j['return_buy_hold'])
        thresh_0.append(j['return_threshold_0_pct'])
        thresh_0_1.append(j['return_threshold_0.1_pct'])
        thresh_0_2.append(j['return_threshold_0.2_pct'])
        thresh_0_5.append(j['return_threshold_0.5_pct'])
        thresh_1.append(j['return_threshold_1_pct'])
        trades_0.append(j['trades_0_pct'])
        trades_0_1.append(j['trades_0.1_pct'])
        trades_0_2.append(j['trades_0.2_pct'])
        trades_0_5.append(j['trades_0.5_pct'])
        trades_1.append(j['trades_1_pct'])
        
    analysis = pd.DataFrame(list(zip(loss, 
                                     train_loss, 
                                     mae, 
                                     sign, 
                                     buy_hold, 
                                     thresh_0, 
                                     thresh_0_1, 
                                     thresh_0_2, 
                                     thresh_0_5, 
                                     thresh_1, 
                                     trades_0, 
                                     trades_0_1, 
                                     trades_0_2, 
                                     trades_0_5, 
                                     trades_1)),
                            index = symbols,
                            columns = ['test_loss', 
                                       'train_loss', 
                                       'mean_absolute_error', 
                                       'correct_sign_pct',
                                       'return_buy_hold', 
                                       'return_threshold_0_pct',
                                       'return_threshold_0.1_pct',
                                       'return_threshold_0.2_pct',
                                       'return_threshold_0.5_pct',
                                       'return_threshold_1_pct',
                                       'trades_0_pct', 
                                       'trades_0.1_pct', 
                                       'trades_0.2_pct',
                                       'trades_0.5_pct', 
                                       'trades_1_pct'])
    analysis.index.rename('symbol', inplace=True)
    
    path = "./analysis/_overall/{}".format(current_date)
    if not os.path.exists(path):
            os.mkdir(path)
            
    analysis.to_csv('{}/analysis.csv'.format(path))
            
    plt.figure()
    sns.distplot(analysis['test_loss'])
    plt.savefig('{}/loss_distribution.png'.format(path))    
    plt.close()    
    
    plt.figure()
    sns.distplot(analysis['correct_sign_pct'])
    plt.savefig('{}/correct_sign_pct_distribution.png'.format(path))    
    plt.close()    