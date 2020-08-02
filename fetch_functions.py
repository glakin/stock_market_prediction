# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:35:25 2020

@author: Jerry
"""
from alpha_vantage.timeseries import TimeSeries

def fetch_daily(symbol, save_csv = False):
    # Should come up with a system to handle the KPI key
    ts = TimeSeries(key='MU7WOKE2G7L93MXA', output_format='pandas')
    data, meta_data = ts.get_daily(symbol, outputsize='full')
    if save_csv == True:
        data.to_csv('./{}_daily.csv'.format(symbol))
    return data
    
def fetch_intraday(symbol, interval = '15min', save_csv = False):
    # Should come up with a system to handle the KPI key
    ts = TimeSeries(key='MU7WOKE2G7L93MXA', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol, interval = interval, outputsize='full')
    if save_csv == True:
        data.to_csv('./{}_intraday.csv'.format(symbol))
    return data