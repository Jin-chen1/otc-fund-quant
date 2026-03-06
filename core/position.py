import datetime
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Lot:
    """
    表示一笔基金买入批次。
    """
    shares: float
    cost_nav: float
    buy_date: datetime.date

class Position:
    """
    管理特定基金的持仓，支持多批次（FIFO）。
    """
    def __init__(self, fund_code: str):
        self.fund_code = fund_code
        self.lots: List[Lot] = []

    @property
    def total_shares(self) -> float:
        """返回所有批次的总持仓份额。"""
        return sum(lot.shares for lot in self.lots)

    @property
    def avg_cost(self) -> float:
        """返回加权平均持仓成本。无持仓时返回 0.0。"""
        total = self.total_shares
        if total <= 1e-9:
            return 0.0
        weighted_cost = sum(lot.shares * lot.cost_nav for lot in self.lots)
        return weighted_cost / total

    def add(self, shares: float, nav: float, date: datetime.date):
        """
        添加一笔新的买入批次。
        """
        self.lots.append(Lot(shares=shares, cost_nav=nav, buy_date=date))

    def sell(self, shares: float, nav: float, sell_date: datetime.date) -> Tuple[float, float, float]:
        """
        使用 FIFO（先进先出）方式卖出份额。

        Args:
            shares: 卖出份额数量。
            nav: 卖出时的净值。
            sell_date: 卖出日期。

        Returns:
            Tuple 包含:
            - realized_pnl: 已实现盈亏（净现金 - 成本）。
            - total_fee: 赎回费用总计。
            - cash_obtained: 扣费后获得的净现金。
        """
        if shares <= 0:
            raise ValueError("卖出份额必须为正数")

        if shares > self.total_shares + 1e-9:
            raise ValueError(f"份额不足。持有: {self.total_shares}, 请求: {shares}")

        shares_remaining = shares
        total_pnl = 0.0
        total_fee = 0.0
        total_cash = 0.0

        # FIFO: 从最早的批次开始消耗
        while shares_remaining > 1e-9 and self.lots:
            current_lot = self.lots[0]

            if current_lot.shares > shares_remaining:
                # 部分卖出当前批次
                sell_amount = shares_remaining
                current_lot.shares -= sell_amount
                shares_remaining = 0.0
            else:
                # 全部卖出当前批次
                sell_amount = current_lot.shares
                shares_remaining -= sell_amount
                self.lots.pop(0)

            if sell_date < current_lot.buy_date:
                raise ValueError(f"卖出日期 {sell_date} 不能早于买入日期 {current_lot.buy_date}")

            # 计算本次卖出的各项指标
            holding_days = (sell_date - current_lot.buy_date).days

            # 赎回费率计算
            # < 7 天: 1.5%
            # 7-365 天: 0.5%
            # > 365 天: 0.0%
            fee_rate = 0.0
            if holding_days < 7:
                fee_rate = 0.015
            elif 7 <= holding_days <= 365:
                fee_rate = 0.005
            else:
                fee_rate = 0.0

            gross_value = sell_amount * nav
            fee = gross_value * fee_rate
            net_cash = gross_value - fee

            cost = sell_amount * current_lot.cost_nav
            pnl = net_cash - cost

            total_fee += fee
            total_cash += net_cash
            total_pnl += pnl

        return total_pnl, total_fee, total_cash
