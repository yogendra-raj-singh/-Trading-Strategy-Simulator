"""
Bitcoin Trading Strategy Simulation using Golden Cross Algorithm.
Simulates 60 days of Bitcoin price data, calculates 7-day and 30-day Moving Averages,
and executes trades based on Golden Cross / Death Cross crossover signals.
"""

import numpy as np
import pandas as pd


def simulate_btc_prices(
    days: int = 60,
    initial_price: float = 60000.0,
    mu: float = 0.0005,
    sigma: float = 0.03,
    seed: int = 42,
) -> pd.Series:
    """
    Simulates daily Bitcoin prices using a Geometric Brownian Motion (GBM) model.

    :param days: Number of days to simulate (default 60).
    :param initial_price: Starting BTC price in USD (default $60,000).
    :param mu: Daily drift rate (expected return).
    :param sigma: Daily volatility (~3% realistic for BTC).
    :param seed: Random seed for reproducibility.
    :return: Pandas Series of daily BTC prices indexed by day 1 to days.
    """
    if seed is not None:
        np.random.seed(seed)

    # GBM formula: S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) + sigma * Z_t)
    dt = 1
    returns = np.random.normal(
        loc=(mu - 0.5 * sigma**2) * dt,
        scale=sigma * np.sqrt(dt),
        size=days - 1,
    )

    prices = np.zeros(days)
    prices[0] = initial_price
    for t in range(1, days):
        prices[t] = prices[t - 1] * np.exp(returns[t - 1])

    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
    return pd.Series(prices, index=dates, name="Price")


def calculate_moving_averages(
    price_series: pd.Series, short_window: int = 7, long_window: int = 30
) -> pd.DataFrame:
    """
    Calculates short-term and long-term rolling moving averages using pandas.

    :param price_series: Pandas Series of prices.
    :param short_window: Window size for short-term MA (default 7 days).
    :param long_window: Window size for long-term MA (default 30 days).
    :return: DataFrame with 'Price', 'MA_Short', and 'MA_Long'.
    """
    df = pd.DataFrame(price_series)
    # Using min_periods=1 so MA is calculated from the start of available data
    df[f"MA_{short_window}"] = price_series.rolling(window=short_window, min_periods=1).mean()
    df[f"MA_{long_window}"] = price_series.rolling(window=long_window, min_periods=1).mean()
    return df


def generate_signals(
    df: pd.DataFrame, short_col: str = "MA_7", long_col: str = "MA_30"
) -> pd.DataFrame:
    """
    Generates trading signals based on Golden Cross and Death Cross strategy.

    - Golden Cross (BUY): Short MA crosses above Long MA.
    - Death Cross (SELL): Short MA crosses below Long MA.

    :param df: DataFrame containing price and moving average columns.
    :param short_col: Column name for short-term MA.
    :param long_col: Column name for long-term MA.
    :return: DataFrame with an added 'Signal' column ('BUY', 'SELL', or 'HOLD').
    """
    df = df.copy()
    df["Signal"] = "HOLD"

    prev_short = df[short_col].shift(1)
    prev_long = df[long_col].shift(1)

    # Golden Cross: Short MA crosses ABOVE Long MA
    golden_cross = (prev_short <= prev_long) & (df[short_col] > df[long_col])
    # Death Cross: Short MA crosses BELOW Long MA
    death_cross = (prev_short >= prev_long) & (df[short_col] < df[long_col])

    df.loc[golden_cross, "Signal"] = "BUY"
    df.loc[death_cross, "Signal"] = "SELL"

    return df


def simulate_portfolio_and_run_backtest(
    df: pd.DataFrame, initial_capital: float = 10000.0
) -> pd.DataFrame:
    """
    Simulates trading execution day-by-day according to generated signals.

    :param df: DataFrame with 'Price' and 'Signal'.
    :param initial_capital: Initial cash in USD (default $10,000).
    :return: DataFrame with full simulation state (Cash, BTC, Portfolio_Value, Trade_Action).
    """
    df = df.copy()
    cash = initial_capital
    btc_holding = 0.0

    cash_history = []
    btc_history = []
    portfolio_value_history = []
    trade_action_history = []

    for idx, row in df.iterrows():
        price = row["Price"]
        signal = row["Signal"]
        action = "HOLD"

        if signal == "BUY" and cash > 0:
            btc_holding = cash / price
            cash = 0.0
            action = f"BUY ({btc_holding:.4f} BTC)"
        elif signal == "SELL" and btc_holding > 0:
            cash = btc_holding * price
            action = f"SELL ({btc_holding:.4f} BTC)"
            btc_holding = 0.0

        portfolio_val = cash + (btc_holding * price)

        cash_history.append(cash)
        btc_history.append(btc_holding)
        portfolio_value_history.append(portfolio_val)
        trade_action_history.append(action)

    df["Cash_USD"] = cash_history
    df["BTC_Holding"] = btc_history
    df["Portfolio_Value"] = portfolio_value_history
    df["Trade_Action"] = trade_action_history

    return df


def print_daily_ledger_and_summary(df: pd.DataFrame, initial_capital: float = 10000.0) -> None:
    """
    Prints a clean daily ledger of trades and final portfolio performance comparison.
    """
    print("=" * 105)
    print(" BITCOIN GOLDEN CROSS TRADING SIMULATION (60-DAY LEDGER)")
    print("=" * 105)
    header = f"{'Day':<4} | {'Date':<10} | {'BTC Price ($)':<13} | {'MA 7 ($)':<11} | {'MA 30 ($)':<11} | {'Signal':<6} | {'Action':<18} | {'Portfolio Value ($)':<19}"
    print(header)
    print("-" * 105)

    for i, (date, row) in enumerate(df.iterrows(), 1):
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        price_str = f"{row['Price']:,.2f}"
        ma7_str = f"{row['MA_7']:,.2f}" if pd.notna(row['MA_7']) else "N/A"
        ma30_str = f"{row['MA_30']:,.2f}" if pd.notna(row['MA_30']) else "N/A"
        signal_str = row['Signal']
        action_str = row['Trade_Action']
        port_val_str = f"{row['Portfolio_Value']:,.2f}"

        print(f"{i:<4} | {date_str:<10} | ${price_str:>11} | ${ma7_str:>9} | ${ma30_str:>9} | {signal_str:<6} | {action_str:<18} | ${port_val_str:>17}")

    print("=" * 105)
    print("\n" + "=" * 60)
    print(" FINAL PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 60)

    start_price = df['Price'].iloc[0]
    end_price = df['Price'].iloc[-1]
    final_val = df['Portfolio_Value'].iloc[-1]
    strategy_return = ((final_val - initial_capital) / initial_capital) * 100

    buy_hold_btc = initial_capital / start_price
    buy_hold_final_val = buy_hold_btc * end_price
    buy_hold_return = ((buy_hold_final_val - initial_capital) / initial_capital) * 100

    print(f" Initial Capital:         ${initial_capital:,.2f}")
    print(f" Starting BTC Price:      ${start_price:,.2f}")
    print(f" Ending BTC Price:        ${end_price:,.2f}")
    print(f" Strategy Final Value:    ${final_val:,.2f} ({strategy_return:+.2f}%)")
    print(f" Buy & Hold Final Value:  ${buy_hold_final_val:,.2f} ({buy_hold_return:+.2f}%)")
    print("=" * 60 + "\n")


def main():
    # 1. Simulate 60 days of BTC price data
    btc_series = simulate_btc_prices(days=60, initial_price=60000.0, seed=42)

    # 2. Calculate 7-day and 30-day Moving Averages
    df_ma = calculate_moving_averages(btc_series, short_window=7, long_window=30)

    # 3. Generate Golden Cross & Death Cross signals
    df_signals = generate_signals(df_ma, short_col="MA_7", long_col="MA_30")

    # 4. Run portfolio backtest
    df_results = simulate_portfolio_and_run_backtest(df_signals, initial_capital=10000.0)

    # 5. Output daily ledger and summary statistics
    print_daily_ledger_and_summary(df_results, initial_capital=10000.0)


if __name__ == "__main__":
    main()
