# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 23:37:30 2020

@author: Jerry
"""

import pandas as pd
from yahoo_earnings_calendar import YahooEarningsCalendar

symbols = pd.read_csv('./data/symbols.csv')


yec = YahooEarningsCalendar()
for symbol in symbols['symbol']:
    earn = pd.DataFrame(yec.get_earnings_of(symbol))
    path = './data/{}_earnings.csv'.format(symbol)
    earn.to_csv(path)


