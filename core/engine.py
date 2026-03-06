import pandas as pd
from datetime import datetime, date
from typing import List, Any
import uuid

from .account import Account
from .order import Order, OrderStatus, OrderDirection
from otc_fund_quant.data.loader import DataLoader

class BacktestEngine:
    """
    Backtest engine for simulating fund trading strategies.
    Executes orders with T+1 settlement simulation.
    """
    def __init__(self, account: Account, data_loader: DataLoader):
        self.account = account
        self.data_loader = data_loader
        self.pending_orders: List[Order] = []
        self.nav_data: pd.DataFrame = pd.DataFrame()
        self.equity_curve = []
        self.trade_logs = []

    def _get_history(self, current_date: date) -> pd.DataFrame:
        """
        Returns dataframe up to date - 1 day (T-1).

        Args:
            current_date: The current simulation date (T).

        Returns:
            pd.DataFrame: Historical data up to T-1.
        """
        if self.nav_data.empty:
            return pd.DataFrame()

        # Filter strictly less than current_date (T-1 data only)
        # nav_data['date'] is assumed to be datetime.date objects after processing in run()
        mask = self.nav_data['date'] < current_date
        return self.nav_data.loc[mask].copy()

    def run(self, fund_code: str, start_date: str, end_date: str, strategy: Any):
        """
        Run the backtest simulation.

        Args:
            fund_code: The fund identifier.
            start_date: Start date string (YYYYMMDD or YYYY-MM-DD).
            end_date: End date string.
            strategy: Strategy object with an `on_bar(date, history_df)` method.
        """
        # Fetch data
        # Using a wide range to ensure we have history for the strategy
        fetch_start = "20000101" # Default to enough history
        raw_df = self.data_loader.fetch_nav(fund_code, fetch_start, end_date.replace("-", ""))

        if raw_df.empty:
            print(f"BacktestEngine: No data found for {fund_code}")
            return

        # Prepare data: Ensure 'date' column is datetime.date
        raw_df['date'] = pd.to_datetime(raw_df['date']).dt.date
        self.nav_data = raw_df.sort_values('date').reset_index(drop=True)

        # Filter for the simulation period
        s_date = pd.to_datetime(start_date).date()
        e_date = pd.to_datetime(end_date).date()

        sim_mask = (self.nav_data['date'] >= s_date) & (self.nav_data['date'] <= e_date)
        sim_data = self.nav_data.loc[sim_mask]

        if sim_data.empty:
            print(f"BacktestEngine: No data in simulation range {start_date} - {end_date}")
            return

        print(f"Starting Backtest for {fund_code} from {start_date} to {end_date}...")

        # Loop through each day (Market Open / Day T)
        for _, row in sim_data.iterrows():
            current_date = row['date']
            current_nav = float(row['nav'])

            # --- Step 1: Settle Pending Orders (from T-1) ---
            # Orders placed on T-1 (or earlier) are executed at T's NAV

            settled_orders = []

            for order in self.pending_orders:
                if order.status == OrderStatus.PENDING:
                    try:
                        if order.direction == OrderDirection.BUY:
                            # Execute Buy: Deduct cash and add shares
                            shares = self.account.buy(fund_code, order.amount, current_nav, current_date)

                            order.mark_filled(
                                fill_date=current_date,
                                price=current_nav,
                                filled_shares=shares,
                                filled_amount=order.amount
                            )
                            self.trade_logs.append({
                                'date': current_date,
                                'action': 'BUY',
                                'shares': shares,
                                'price': current_nav,
                                'amount': order.amount,
                                'nav': current_nav
                            })
                            # print(f"[{current_date}] Filled BUY: {shares:.2f} shares @ {current_nav}")

                        elif order.direction == OrderDirection.SELL:
                            # Execute Sell: Deduct shares and add cash
                            result = self.account.sell(fund_code, order.shares, current_nav, current_date)

                            order.mark_filled(
                                fill_date=current_date,
                                price=current_nav,
                                filled_shares=order.shares,
                                filled_amount=result['cash_obtained'],
                                cost=result['fee']
                            )
                            self.trade_logs.append({
                                'date': current_date,
                                'action': 'SELL',
                                'shares': order.shares,
                                'price': current_nav,
                                'amount': result['cash_obtained'],
                                'nav': current_nav
                            })
                            # print(f"[{current_date}] Filled SELL: {result['cash_obtained']:.2f} RMB")

                    except ValueError as e:
                        print(f"[{current_date}] Order {order.id} Failed: {e}")
                        order.status = OrderStatus.REJECTED

                settled_orders.append(order)

            # Clear pending list (they are now FILLED or REJECTED)
            # In a full system, we might move them to an 'order_history' list
            self.pending_orders = []

            # --- Track Daily Equity ---
            pos = self.account.get_position(fund_code)
            market_value = pos.total_shares * current_nav
            total_assets = self.account.cash + market_value
            self.equity_curve.append({
                'date': current_date,
                'nav': current_nav,
                'cash': self.account.cash,
                'market_value': market_value,
                'total_assets': total_assets
            })

            # --- Step 2: Strategy Decision (Day T) ---
            # Pass Day T-1 data to strategy (do not reveal today's nav)
            history_df = self._get_history(current_date)

            if history_df.empty:
                continue

            # Strategy returns trading signal
            # Expected return: {'buy_amount': float} or {'sell_shares': float}
            signal = {}
            if hasattr(strategy, 'on_bar'):
                signal = strategy.on_bar(current_date, history_df)

            buy_amount = signal.get('buy_amount', 0.0)
            sell_shares = signal.get('sell_shares', 0.0)

            # --- Step 3: Place Orders ---
            # Create new Order with date=T, status=PENDING

            if buy_amount > 0:
                # Optional: Pre-check cash to avoid rejection next day (though settlement handles it)
                if self.account.cash >= buy_amount:
                    new_order = Order(
                        id=str(uuid.uuid4())[:8],
                        fund_code=fund_code,
                        direction=OrderDirection.BUY,
                        amount=buy_amount,
                        status=OrderStatus.PENDING,
                        create_date=current_date
                    )
                    self.pending_orders.append(new_order)
                else:
                    print(f"[{current_date}] Signal ignored: Insufficient cash for BUY {buy_amount}")

            elif sell_shares > 0:
                # Optional: Pre-check position
                position = self.account.get_position(fund_code)
                if position.total_shares >= sell_shares:
                    new_order = Order(
                        id=str(uuid.uuid4())[:8],
                        fund_code=fund_code,
                        direction=OrderDirection.SELL,
                        shares=sell_shares,
                        status=OrderStatus.PENDING,
                        create_date=current_date
                    )
                    self.pending_orders.append(new_order)
                else:
                    print(f"[{current_date}] Signal ignored: Insufficient shares for SELL {sell_shares}")

        print("Backtest Completed.")
