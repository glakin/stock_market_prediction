# Install may not work if you aren't using the most recent version of R
install.packages("installr")
library(installr)
updateR()

install.packages("devtools") # if not installed
install.packages("FinancialInstrument") #if not installed
install.packages("PerformanceAnalytics") #if not installed

# next install blotter from GitHub
devtools::install_github("braverock/blotter")
# next install quantstrat from GitHub
devtools::install_github("braverock/quantstrat")