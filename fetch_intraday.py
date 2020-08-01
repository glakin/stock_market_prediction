# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:33:49 2020

@author: Jerry
"""


from alpha_vantage.timeseries import TimeSeries

def fetch_intraday(symbol):
    # Should come up with a system to handle the KPI key
    ts = TimeSeries(key='MU7WOKE2G7L93MXA', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol, outputsize='full')
    data.to_csv('./{}_intraday.csv'.format(symbol, meta_data))