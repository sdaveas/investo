# Investo – Investment Simulator

A Streamlit app that simulates historical investment performance using real market data from Yahoo Finance.

## Setup

```bash
# Install dependencies
pipenv install

# Run the app
pipenv run streamlit run app.py
```

## Features

- Pick a start date, end date, and initial investment amount
- Choose from common investments: S&P 500, NASDAQ 100, bonds, gold, individual stocks, and more
- Interactive Plotly chart showing portfolio value over time
- Summary stats: total return, annualized return, and max drawdown
