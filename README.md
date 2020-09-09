# stock_market_prediction

The goal of this project is to predict stock market prices using deep learning.

While it would be nice to get rich off of accurate stock market predictions, the main goal here is to build a multi-faceted data science project and get some practice with keras. The repository includes code to utilize APIs to populate a database with stock data, including prices, technical indicators, and earnings data. It also contains code to train deep learning models from the data and analyze the results. There is also an exploratory analysis file that is usesd to investigate the correlation between technical indicators and price changes.

In order to run this code you will need to claim an API key from AlphaVantage (https://www.alphavantage.co/) and store the key in a .json file as within the repository called credentials.json. The API key should be given the name "alpha_vantage" (see fetch_functions.py). You will also need to be running a local MySQL instance with a database called "stocks". The credentials for this database should be stored using the keyring package in python (see etl_functions.py).
