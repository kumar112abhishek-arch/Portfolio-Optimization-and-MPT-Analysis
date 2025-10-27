# ==============================
# Portfolio Optimization Project (Google Colab Ready)
# Author: Abhishek kumar
# ==============================

# Step 0: Import libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Ensure plots display
%matplotlib inline

# Step 1: Download historical stock data (2 years)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(tickers, start='2023-01-01', end='2025-01-01')['Adj Close']

print("Stock Price Data:")
print(data.head())

# Step 2: Calculate Daily Returns
returns = data.pct_change().dropna()
print("\n Daily Returns:")
print(returns.head())

# Step 3: Equal-Weighted Portfolio
weights_eq = np.array([0.25, 0.25, 0.25, 0.25])
annual_returns = returns.mean() * 252
portfolio_return_eq = np.dot(weights_eq, annual_returns)
cov_matrix = returns.cov() * 252
portfolio_volatility_eq = np.sqrt(np.dot(weights_eq.T, np.dot(cov_matrix, weights_eq)))

print("\n Equal-Weighted Portfolio Performance:")
print("Expected Annual Return: {:.2f}%".format(portfolio_return_eq*100))
print("Portfolio Volatility (Risk): {:.2f}%".format(portfolio_volatility_eq*100))

# Cumulative return plot
(returns * weights_eq).cumsum().plot(figsize=(10,5))
plt.title("Equal-Weighted Portfolio Growth (Cumulative Returns)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()

# Step 4: Monte-Carlo Simulation for 10,000 Portfolios
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(4)
    weights /= np.sum(weights)
    weights_record.append(weights)
    
    port_return = np.dot(weights, annual_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = port_return / port_volatility
    
    results[0,i] = port_return
    results[1,i] = port_volatility
    results[2,i] = sharpe_ratio

results_df = pd.DataFrame(results.T, columns=["Returns", "Volatility", "Sharpe Ratio"])

# Step 5: Identify Optimal Portfolio
max_sharpe = results_df.iloc[results_df['Sharpe Ratio'].idxmax()]
max_sharpe_weights = weights_record[results_df['Sharpe Ratio'].idxmax()]

print("\n Optimal Portfolio (Max Sharpe Ratio):")
print(max_sharpe)
print("\nWeights for each stock:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {max_sharpe_weights[i]*100:.2f}%")

# Step 6: Plot Efficient Frontier
plt.figure(figsize=(10,6))
plt.scatter(results_df['Volatility'], results_df['Returns'], alpha=0.4, label="Portfolios")
plt.scatter(max_sharpe[1], max_sharpe[0], color='red', s=100, label="Optimal Portfolio")
plt.title("Efficient Frontier — 10,000 Portfolios")
plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")
plt.legend()
plt.show()
