import yfinance as yf

# Example for Nvidia (NVDA)
ticker = "NVDA"
data = yf.download(ticker, period="1wk", interval="1d")  # Fetch data for 1 week
print(data)
