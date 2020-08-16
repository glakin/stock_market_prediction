# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 23:05:35 2020

@author: Jerry
"""
import urllib.request
import json
import pandas as pd

credentials = json.load(open('credentials.json','r'))
unibit_key = credentials['unibit']

symbol = "AAPL"

url_txt = "https://api.unibit.ai/v2/company/financialSummary?tickers={}&selectedFields=all&accessKey={}".format(symbol, unibit_key)

with urllib.request.urlopen(url_txt) as url:
    data = json.loads(url.read().decode())

df = pd.DataFrame.from_dict(data["result_data"], orient="index")
