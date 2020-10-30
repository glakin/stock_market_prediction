library(quantstrat)

stock.str <- 'EEM'

currency('USD')
stock(stock.str,currency='USD',multiplier=1)

startDate='2003-12-31'
initEq=100000
portfolio.st='macd'
account.st='macd'

initPortf(portfolio.st,symbols=stock.str)
initAcct(account.st,portfolios=portfolio.st,initEq = initEq)
initOrders(portfolio=portfolio.st)

strategy.st<-portfolio.st
# define the strategy
strategy(strategy.st, store=TRUE)

library(rjson)

cred <- fromJSON(file = 'credentials.json')
cred["tiingo"]

getSymbols(stock.str,
           from=startDate,
           adjust=TRUE,
           src='tiingo',
           api.key=cred["tiingo"])
