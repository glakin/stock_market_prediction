# This is copied mostly from 
# https://rdrr.io/github/braverock/quantstrat/f/vignettes/quantstrat-RFinance-2018.Rmd

library(quantstrat)

stock.str <- 'EEM'

currency('USD')
stock(stock.str,currency='USD',multiplier=1)

startDate='2003-12-31'
initEq=100000
portfolio.st='macd'
account.st='macd'

initPortf(portfolio.st, symbols=stock.str)
initAcct(account.st,portfolios=portfolio.st, initEq=initEq)
initOrders(portfolio=portfolio.st)

strategy.st<-portfolio.st
# define the strategy
strategy(strategy.st, store=TRUE)

library(rjson)

# read api key from credentials.json
cred <- fromJSON(file = 'credentials.json')
cred["tiingo"]

getSymbols(stock.str,
           from=startDate,
           adjust=TRUE,
           src='tiingo',
           api.key=cred["tiingo"])

#MA parameters for MACD
fastMA = 12 
slowMA = 26 
signalMA = 9
maType="EMA"

#one indicator
add.indicator(strategy.st, name = "MACD", 
              arguments = list(x=quote(Cl(mktdata)),
                               nFast=fastMA, 
                               nSlow=slowMA),
              label='_' 
)

#two signals
add.signal(strategy.st,
           name="sigThreshold",
           arguments = list(column="signal._",
                            relationship="gt",
                            threshold=0,
                            cross=TRUE),
           label="signal.gt.zero"
)

add.signal(strategy.st,
           name="sigThreshold",
           arguments = list(column="signal._",
                            relationship="lt",
                            threshold=0,
                            cross=TRUE),
           label="signal.lt.zero"
)