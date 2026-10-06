# Portfolio Optimization Research — Indonesian Equity Mutual Fund

## Overview

This project is an independent quantitative analysis of the disclosed top equity holdings of an Indonesian mutual fund.

The objective is to examine whether the same investment universe could have achieved better historical **risk-adjusted returns** through alternative portfolio weighting.

Using historical market data, I applied **Monte Carlo Simulation** and **Modern Portfolio Theory (MPT)** to generate alternative portfolio allocations and evaluate their expected return, volatility, and Sharpe Ratio.

## Methodology

The analysis includes:

- Historical stock price and return analysis
- Correlation and covariance analysis
- Monte Carlo simulation of **10,000 portfolio combinations**
- Portfolio allocation constraints to maintain realistic diversification
- Expected annual return and volatility calculation
- Sharpe Ratio optimization
- Efficient Frontier visualization
- Comparison between the fund's disclosed allocation and simulated portfolios

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Historical market data

## Objective

The study evaluates whether alternative weighting among the fund's disclosed equity holdings could have produced a more efficient historical risk-return profile.

Rather than attempting to predict future performance, the project demonstrates the application of quantitative portfolio analysis to a real-world investment universe.

## Key Concepts

### Modern Portfolio Theory (MPT)
Used to analyze the relationship between portfolio diversification, expected return, and risk.

### Monte Carlo Simulation
Thousands of alternative portfolio weights are generated to identify different combinations of expected return and volatility.

### Efficient Frontier
Used to visualize portfolios that provide the highest expected return for a given level of portfolio risk.

### Sharpe Ratio
Used as the primary measure of risk-adjusted performance when comparing alternative portfolio allocations.

## Disclaimer

This project was conducted independently for educational and portfolio purposes. It is not affiliated with, endorsed by, or representative of the mutual fund or its investment manager.

The analysis is based on historical data and publicly available information. Results should not be interpreted as investment advice or as evidence that the simulated portfolios would outperform in the future.
