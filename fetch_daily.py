# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 18:36:58 2020

@author: Jerry
"""

from alpha_vantage.timeseries import TimeSeries

def fetch_daily(symbol):
    # Should come up with a system to handle the KPI key
    ts = TimeSeries(key='MU7WOKE2G7L93MXA', output_format='pandas')
    data, meta_data = ts.get_daily(symbol, outputsize='full')
    data.to_csv('./{}_daily.csv'.format(symbol))