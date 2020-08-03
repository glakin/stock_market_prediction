# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:38:11 2020

@author: Jerry
"""

from fetch_functions import fetch_daily, fetch_intraday, fetch_technicals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use fetch functions to pull daily and intraday data for each symbol
amzn_daily = fetch_daily("AMZN")
googl_daily = fetch_daily("GOOGL")
pton_daily = fetch_daily("PTON")
ko_daily = fetch_daily("KO")

amzn_intraday = fetch_intraday("AMZN")
googl_intraday = fetch_intraday("GOOGL")
pton_intraday = fetch_intraday("PTON")
ko_intraday = fetch_intraday("KO")

ko_intraday_1min = fetch_intraday("KO",interval="1min")

# Plot daily price and volume charts for each symbol
amzn_daily['4. close'].plot()
amzn_daily['5. volume'].plot()

googl_daily['4. close'].plot()
googl_daily['5. volume'].plot()

pton_daily['4. close'].plot()
pton_daily['5. volume'].plot()

ko_daily['4. close'].plot()
ko_daily['5. volume'].plot()

# Plot intraday charts
amzn_intraday['4. close'].plot()
googl_intraday['4. close'].plot()
pton_intraday['4. close'].plot()
ko_intraday['4. close'].plot()

# Plot Coca Cola intraday on a 1 minute interval
ko_intraday_1min['4. close'].plot()
ko_intraday_1min['5. volume'].plot()

# Some thoughts: volume tends to be high on day of IPO (see Peloton chart) so
# it probably makes sense to remove the first element from that dataset
# We should also try to account for earnings calls which can lead to big swings
# (e.g. Coca Cola had earnings on Jul 20 and opened ~2 pts up the next day)

googl_technicals = fetch_technicals('GOOGL', interval='daily')
googl_technicals['SMA'].plot()
googl_technicals['MACD'].plot()
googl_technicals['RSI'].plot()

# Run bollinger bands on the same graph
googl_technicals['Real Middle Band'].plot()
googl_technicals['Real Lower Band'].plot()
googl_technicals['Real Upper Band'].plot()