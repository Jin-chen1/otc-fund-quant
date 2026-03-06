import sys
import os
import pandas as pd
from datetime import date, datetime
from unittest.mock import MagicMock

# 1. Mock akshare to bypass import error
sys.modules['akshare'] = MagicMock()

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Now imports should work even if akshare is missing
try:
    from core.account import Account
    from core.engine import BacktestEngine
    from data.loader import DataLoader
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback if previous fix didn't work for core/engine.py import
    # But we patched akshare, so data.loader should import fine, just not work.

class MockDataLoader(DataLoader):
    def __init__(self):
        # Skip super init which might use sqlite
        self.db_path = ":memory:"

    def fetch_nav(self, fund_code, start_date, end_date):
        # Return dummy data
        # Date range: 2024-01-01 to 2024-02-01
        dates = pd.date_range(start='2024-01-01', end='2024-02-01')
        df = pd.DataFrame({
            'date': dates,
            'fund_code': fund_code,
            'nav': 1.0, # Constant NAV for easy fee calculation
            'acc_nav': 1.0
        })
        # Convert date to expected format if needed (BacktestEngine expects datetime.date in 'date' col after processing)
        # But fetch_nav usually returns string or datetime. BacktestEngine converts to date.
        # Let's return what fetch_nav usually returns.
        return df

class TestStrategy:
    def __init__(self, account, fund_code):
        self.account = account
        self.fund_code = fund_code

    def on_bar(self, current_date, history_df):
        date_str = current_date.strftime('%Y-%m-%d')

        if date_str == '2024-01-05':
            print(f"[{date_str}] Strategy Signal: BUY 10000 RMB")
            return {'buy_amount': 10000.0}

        if date_str == '2024-01-10':
            # Get current position to sell all
            # In T+1 system:
            # Buy on Jan 5 (Friday). Order Pending.
            # Next trading day... if we assume every day is trading day for this mock.
            # Jan 5: Order Placed.
            # Jan 6 (Sat): Engine loop runs? Our mock data has all days.
            # Jan 6: Order Filled (BacktestEngine processes pending orders at start of day).
            # So Shares owned on Jan 6.
            # Sell on Jan 10.
            # Holding: Jan 10 - Jan 6 = 4 days?
            # Or does it settle same day in simulation loop?

            # Let's trace BacktestEngine:
            # Loop Day T (Jan 5):
            #   Step 1: Settle Pending (None)
            #   Step 2: Strategy -> Signal BUY
            #   Step 3: Place Order (Pending, created_at=Jan 5)

            # Loop Day T+1 (Jan 6):
            #   Step 1: Settle Pending (Order from Jan 5)
            #           Executes at Jan 6 NAV.
            #           Shares added. Buy Date = Jan 6.

            # Loop ... Jan 10:
            #   Step 1: Settle...
            #   Step 2: Strategy -> Signal SELL
            #   Step 3: Place Order (Pending, created_at=Jan 10)

            # Loop Jan 11:
            #   Step 1: Settle Pending (Order from Jan 10)
            #           Executes at Jan 11 NAV.
            #           Sell Date = Jan 11.

            # Holding Period: Buy Date (Jan 6) to Sell Date (Jan 11).
            # Days = 11 - 6 = 5 days.
            # < 7 days -> 1.5% Fee.

            pos = self.account.get_position(self.fund_code)
            shares = pos.total_shares
            if shares > 0:
                print(f"[{date_str}] Strategy Signal: SELL ALL ({shares:.2f} shares)")
                return {'sell_shares': shares}

        return {}

def run_test():
    fund_code = '005658'

    # Setup
    account = Account(initial_cash=100000.0)
    loader = MockDataLoader()
    engine = BacktestEngine(account, loader)
    strategy = TestStrategy(account, fund_code)

    print(f"Initial Cash: {account.cash}")

    # Run
    engine.run(fund_code, '2024-01-01', '2024-02-01', strategy)

    print("-" * 30)
    print(f"Final Cash: {account.cash:.2f}")

    # Verify Fee
    # Expected:
    # Initial: 100000
    # Buy 10000 -> Cash 90000. Shares = 10000 (NAV=1).
    # Sell 10000 Shares -> Gross 10000.
    # Fee 1.5% = 150.
    # Net Cash = 9850.
    # Final Cash = 90000 + 9850 = 99850.

    expected_cash = 99850.0
    diff = abs(account.cash - expected_cash)

    if diff < 1.0:
        print("SUCCESS: Final cash matches expected value (1.5% fee applied).")
    else:
        print(f"FAILURE: Final cash {account.cash:.2f} != Expected {expected_cash:.2f}")
        print("Did the fee logic work?")

if __name__ == "__main__":
    run_test()
