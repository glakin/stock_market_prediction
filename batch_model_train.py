# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 00:02:01 2020

@author: Jerry
"""
from modeling_functions import train_model
from datetime import datetime
import pandas as pd
import json
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date

current_date = date.today()

symbols = pd.read_csv('./data/symbols.csv')['symbol'].to_list()

print('{} - Training models'.format(datetime.now().strftime("%H:%M:%S")))
for symbol in symbols:
    print('{} - Modeling {}'.format(datetime.now().strftime("%H:%M:%S"), symbol))
    train_model(symbol)

loss = []
sign = []
buy_hold = []
thresh_0 = []
thresh_0_1 = []
thresh_0_2 = []
thresh_0_5 = []
thresh_1 = []

for symbol in symbols:
    path = "./analysis/{}/{}".format(symbol, current_date)
    j = json.load(open('{}/{}_analysis.txt'.format(path,symbol),'r'))
    loss.append(j['test_loss'])
    sign.append(j['correct_sign_pct'])
    buy_hold.append(j['return_buy_hold'])
    thresh_0.append(j['return_threshold_0_pct'])
    thresh_0_1.append(j['return_threshold_0.1_pct'])
    thresh_0_2.append(j['return_threshold_0.2_pct'])
    thresh_0_5.append(j['return_threshold_0.5_pct'])
    thresh_1.append(j['return_threshold_1_pct'])
    
analysis = pd.DataFrame(list(zip(loss, sign, buy_hold, thresh_0, thresh_0_1, thresh_0_2, thresh_0_5, thresh_1)),
                        index = symbols,
                        columns = ['test_loss','correct_sign_pct',
                                   'return_buy_hold','return_threshold_0_pct',
                                   'return_threshold_0.1_pct','return_threshold_0.2_pct',
                                   'return_threshold_0.5_pct','return_threshold_1_pct'])

plt.figure()
sns.distplot(analysis['test_loss'])

plt.figure()
sns.distplot(analysis['correct_sign_pct'])


