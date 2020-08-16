# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:35:25 2020

@author: Jerry
"""
# Alpha vantage documentation: https://www.alphavantage.co/documentation/

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
import pandas as pd
from datetime import date
import json

def fetch_sector_performance(save_csv = False):
    current_date = date.today()
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    sp = SectorPerformances(key=av_key, output_format='pandas')
    data, meta_data = sp.get_sector()
    if save_csv == True:
        data.to_csv('./data/sector_performance_{}.csv'.format(current_date))
    return data

def fetch_daily(symbol, save_csv = False):
    # Should come up with a system to handle the KPI key    
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TimeSeries(key=av_key, output_format = 'pandas')
    data, meta_data = ts.get_daily(symbol, outputsize = 'full')
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_daily_{}.csv'.format(symbol, current_date))
    return data

def fetch_daily_adjusted(symbol, save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TimeSeries(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_daily_adjusted(symbol, outputsize = 'full')
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_daily_adjusted_{}.csv'.format(symbol, current_date))
    return data
    
def fetch_intraday(symbol, interval = '15min', save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TimeSeries(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_intraday(symbol, interval = interval, outputsize = 'full')
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_intraday_{}.csv'.format(symbol, current_date))
    return data

# SMA = Simple Moving Average
def fetch_sma(symbol, interval = '15min', time_period = 50, series_type = 'close', save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_sma(symbol, interval = interval, time_period = time_period, series_type = series_type)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_sma_{}.csv'.format(symbol, current_date))
    return data

# EMA = Exponential Moving Average
def fetch_ema(symbol, interval = '15min', time_period = 50, series_type = 'close', save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_ema(symbol, interval = interval, time_period = time_period, series_type = series_type)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_ema_{}.csv'.format(symbol, current_date))
    return data

# VWAP = Value Weighted Average Price
def fetch_vwap(symbol, interval = '15min', save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_vwap(symbol, interval = interval)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_vwap_{}.csv'.format(symbol, current_date))
    return data

# MACD = Moving Average Convergence/Divergence
def fetch_macd(symbol, interval = '15min', series_type = 'close', save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_macd(symbol, interval = interval, series_type = series_type)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_macd_{}.csv'.format(symbol, current_date))
    return data

# STOCH = Stochastic Oscillator
def fetch_stoch(symbol, interval = '15min', save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_stoch(symbol, interval = interval)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_stoch_{}.csv'.format(symbol, current_date))
    return data

# RSI = Relative Strength Index
def fetch_rsi(symbol, interval = '15min', save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_rsi(symbol, interval = interval)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_rsi_{}.csv'.format(symbol, current_date))
    return data

# ADX = Average Directional Movement Index
def fetch_adx(symbol, interval = '15min', time_period = 50, save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_adx(symbol, interval = interval, time_period = time_period)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_adx_{}.csv'.format(symbol, current_date))
    return data

# CCI = Commodity Channel Index
def fetch_cci(symbol, interval = '15min', time_period = 50, save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_cci(symbol, interval = interval, time_period = time_period)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_cci_{}.csv'.format(symbol, current_date))
    return data

# AROON = Aroon Indicator
def fetch_aroon(symbol, interval = '15min', time_period = 50, save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_aroon(symbol, interval = interval, time_period = time_period)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_aroon_{}.csv'.format(symbol, current_date))
    return data

# BBANDS = Bollinger Bands
def fetch_bbands(symbol, interval = '15min', time_period = 50, save_csv = False, series_type = 'close'):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_bbands(symbol, interval = interval, time_period = time_period, series_type = series_type)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_bbands_{}.csv'.format(symbol, current_date))
    return data

# AD = Chaikin A/D Line
def fetch_ad(symbol, interval = '15min', save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_ad(symbol, interval = interval)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_ad_{}.csv'.format(symbol, current_date))
    return data

# OBV = On Balance Volume
def fetch_obv(symbol, interval = '15min', save_csv = False):
    credentials = json.load(open('credentials.json','r'))
    av_key = credentials['alpha_vantage']
    ts = TechIndicators(key = av_key, output_format = 'pandas')
    data, meta_data = ts.get_obv(symbol, interval = interval)
    if save_csv == True:
        current_date = date.today()
        data.to_csv('./data/{}_obv_{}.csv'.format(symbol, current_date))
    return data

# A function to pull multiple technical indicators
def fetch_technicals(symbol, save_csv = False, interval = '15min', time_period = 50, series_type = 'close'):
    from functools import reduce
    df1 = fetch_sma(symbol, save_csv = False, interval = interval, time_period = time_period, series_type = series_type)
    df2 = fetch_adx(symbol, save_csv = False, interval = interval, time_period = time_period)
    df3 = fetch_macd(symbol, save_csv = False, interval = interval, series_type = series_type)
    df4 = fetch_rsi(symbol, save_csv = False, interval = interval)
    df5 = fetch_bbands(symbol, save_csv = False, interval = interval, time_period = time_period, series_type = series_type)
    
    df_list = [df1, df2, df3, df4, df5]
    technicals = reduce(lambda  left,right: pd.merge(left,right,on=['date'], how='outer'), df_list)
    if save_csv == True:
        current_date = date.today()
        technicals.to_csv('./data/{}_technicals_{}.csv'.format(symbol, current_date))
    return technicals