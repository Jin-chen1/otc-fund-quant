from enum import Enum
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    """
    Represents a fund transaction order.

    Attributes:
        id (str): Unique identifier for the order.
        fund_code (str): The code of the fund (e.g., '005658').
        direction (OrderDirection): BUY or SELL.
        amount (float): For BUY orders, the monetary amount to invest.
                        For SELL orders, the estimated amount (calculated after fill).
        shares (float): For SELL orders, the number of shares to redeem.
                        For BUY orders, the shares obtained (calculated after fill).
        status (OrderStatus): Current status of the order.
        create_date (date): The date the order was placed (Decision Day / T).
        fill_date (Optional[date]): The date the order was settled/filled (T+1).
        fill_price (float): The NAV at which the order was filled.
        transaction_cost (float): Fees associated with the transaction.
    """
    id: str
    fund_code: str
    direction: OrderDirection
    amount: float = 0.0
    shares: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    create_date: date = field(default_factory=date.today)
    fill_date: Optional[date] = None
    fill_price: float = 0.0
    transaction_cost: float = 0.0

    def mark_filled(self, fill_date: date, price: float, filled_shares: float, filled_amount: float, cost: float = 0.0):
        """
        Updates order status to FILLED with settlement details.
        """
        self.status = OrderStatus.FILLED
        self.fill_date = fill_date
        self.fill_price = price
        self.transaction_cost = cost

        # Update the unknown side of the equation
        if self.direction == OrderDirection.BUY:
            self.shares = filled_shares
            # amount was already set
        else:
            self.amount = filled_amount
            # shares was already set
