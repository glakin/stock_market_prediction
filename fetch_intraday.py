# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:33:49 2020

@author: Jerry
"""


from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key='MU7WOKE2G7L93MXA', output_format='pandas')
data, meta_data = ts.get_intraday('GOOGL', outputsize='full')

data.to_csv('./GOOGL_intraday.csv')