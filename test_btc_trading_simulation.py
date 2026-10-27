import unittest
import numpy as np
import pandas as pd

from btc_trading_simulation import (
    simulate_btc_prices,
    calculate_moving_averages,
    generate_signals,
    simulate_portfolio_and_run_backtest,
)


class TestBTCTradingSimulation(unittest.TestCase):

    def test_simulate_btc_prices(self):
        days = 60
        initial_price = 60000.0
        prices = simulate_btc_prices(days=days, initial_price=initial_price, seed=42)

        self.assertIsInstance(prices, pd.Series)
        self.assertEqual(len(prices), days)
        self.assertEqual(prices.iloc[0], initial_price)
        self.assertTrue((prices > 0).all(), "All simulated prices should be positive")

    def test_calculate_moving_averages(self):
        prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        df_ma = calculate_moving_averages(prices, short_window=3, long_window=5)

        self.assertIn("MA_3", df_ma.columns)
        self.assertIn("MA_5", df_ma.columns)

        # MA_3 at index 2 (values 10, 20, 30) should be 20.0
        self.assertAlmostEqual(df_ma["MA_3"].iloc[2], 20.0)
        # MA_3 at index 3 (values 20, 30, 40) should be 30.0
        self.assertAlmostEqual(df_ma["MA_3"].iloc[3], 30.0)
        # MA_5 at index 4 (values 10, 20, 30, 40, 50) should be 30.0
        self.assertAlmostEqual(df_ma["MA_5"].iloc[4], 30.0)

    def test_generate_signals(self):
        data = {
            "Price": [100.0, 110.0, 120.0, 110.0],
            "MA_7": [10.0, 15.0, 25.0, 18.0],
            "MA_30": [12.0, 15.0, 20.0, 20.0],
        }
        df = pd.DataFrame(data)
        df_sig = generate_signals(df, short_col="MA_7", long_col="MA_30")

        # Row 0: short (10) < long (12) -> HOLD
        self.assertEqual(df_sig["Signal"].iloc[0], "HOLD")
        # Row 1: short (15) == long (15), prev short (10) <= long (12) -> HOLD (not crossed above yet)
        self.assertEqual(df_sig["Signal"].iloc[1], "HOLD")
        # Row 2: short (25) > long (20), prev short (15) <= long (15) -> BUY (Golden Cross)
        self.assertEqual(df_sig["Signal"].iloc[2], "BUY")
        # Row 3: short (18) < long (20), prev short (25) >= long (20) -> SELL (Death Cross)
        self.assertEqual(df_sig["Signal"].iloc[3], "SELL")

    def test_simulate_portfolio_and_run_backtest(self):
        data = {
            "Price": [100.0, 200.0, 150.0],
            "Signal": ["BUY", "HOLD", "SELL"],
        }
        df = pd.DataFrame(data)
        initial_cash = 1000.0
        res = simulate_portfolio_and_run_backtest(df, initial_capital=initial_cash)

        # Day 1: BUY at $100 -> buys 10 BTC, cash = 0, portfolio val = $1000
        self.assertEqual(res["Cash_USD"].iloc[0], 0.0)
        self.assertEqual(res["BTC_Holding"].iloc[0], 10.0)
        self.assertEqual(res["Portfolio_Value"].iloc[0], 1000.0)

        # Day 2: HOLD at $200 -> cash = 0, 10 BTC, portfolio val = $2000
        self.assertEqual(res["Cash_USD"].iloc[1], 0.0)
        self.assertEqual(res["BTC_Holding"].iloc[1], 10.0)
        self.assertEqual(res["Portfolio_Value"].iloc[1], 2000.0)

        # Day 3: SELL at $150 -> sells 10 BTC for $1500 cash, 0 BTC, portfolio val = $1500
        self.assertEqual(res["Cash_USD"].iloc[2], 1500.0)
        self.assertEqual(res["BTC_Holding"].iloc[2], 0.0)
        self.assertEqual(res["Portfolio_Value"].iloc[2], 1500.0)


if __name__ == "__main__":
    unittest.main()
