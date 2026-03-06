from typing import Dict, List
from datetime import date
from .position import Position

class Account:
    """
    Manages the overall trading account, including cash and fund positions.
    """
    def __init__(self, initial_cash: float = 0.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}

    def get_position(self, fund_code: str) -> Position:
        """
        Get position for a fund code. Creates a new one if it doesn't exist.
        """
        if fund_code not in self.positions:
            self.positions[fund_code] = Position(fund_code)
        return self.positions[fund_code]

    def buy(self, fund_code: str, amount: float, nav: float, buy_date: date) -> float:
        """
        Buy a fund with a specified cash amount.

        Args:
            fund_code: The fund identifier.
            amount: Cash amount to invest.
            nav: Current Net Asset Value.
            buy_date: Date of purchase.

        Returns:
            float: Shares bought.
        """
        if amount > self.cash:
            raise ValueError(f"Insufficient cash. Available: {self.cash}, Required: {amount}")

        # Calculate shares (assuming no entry fee for simplicity, or fee is external)
        # If entry fee needs to be handled, it should be deducted from amount here
        shares = amount / nav

        self.cash -= amount
        position = self.get_position(fund_code)
        position.add(shares, nav, buy_date)

        return shares

    def sell(self, fund_code: str, shares: float, nav: float, sell_date: date) -> dict:
        """
        Sell shares of a fund.

        Args:
            fund_code: The fund identifier.
            shares: Number of shares to sell.
            nav: Current Net Asset Value.
            sell_date: Date of sale.

        Returns:
            dict: Result of the sale {pnl, fee, cash_obtained}
        """
        if fund_code not in self.positions:
            raise ValueError(f"No position found for {fund_code}")

        position = self.positions[fund_code]
        pnl, fee, cash_obtained = position.sell(shares, nav, sell_date)

        self.cash += cash_obtained

        # Optional: Clean up empty positions?
        # Keeping it simple for now.

        return {
            "pnl": pnl,
            "fee": fee,
            "cash_obtained": cash_obtained
        }
