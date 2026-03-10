from __future__ import annotations

from datetime import date
from typing import Any, Dict

import pandas as pd

from ..analysis.indicators import calc_indicators
from ..core.account import Account


MAX_POSITION_RATIO = 0.60
DCA_BASE_RATIO = 0.02
DCA_INTERVAL = 10
VOLATILITY_GUARD_WINDOW = 60
PULLBACK_BUY_THRESHOLD = 0.015
DRAWDOWN_EXIT_THRESHOLD = 0.025
TREND_CONFIRM_WINDOW = 5
REENTRY_WINDOW = 3
MIN_HISTORY_DAYS = 140
TAIL_SIZE = 240
PERCENTILE_WINDOW = 120


def evaluate_bond_stability_signal(history_df: pd.DataFrame, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """纯债稳健策略信号评估。"""
    p = params or {}
    min_history_days = int(p.get("min_history_days", MIN_HISTORY_DAYS))
    if len(history_df) < min_history_days:
        return {
            "action": "HOLD",
            "reason": f"历史数据不足{min_history_days}天，无法生成信号",
            "indicators": {},
            "buy_score": 0,
            "sell_signal": False,
            "strategy": "bond_stability",
            "stable_trend_buy": False,
            "pullback_recovery_buy": False,
        }

    volatility_guard_window = int(p.get("volatility_guard_window", VOLATILITY_GUARD_WINDOW))
    pullback_buy_threshold = float(p.get("pullback_buy_threshold", PULLBACK_BUY_THRESHOLD))
    drawdown_exit_threshold = float(p.get("drawdown_exit_threshold", DRAWDOWN_EXIT_THRESHOLD))
    trend_confirm_window = int(p.get("trend_confirm_window", TREND_CONFIRM_WINDOW))
    reentry_window = int(p.get("reentry_window", REENTRY_WINDOW))
    tail_size = int(p.get("tail_size", TAIL_SIZE))
    percentile_window = int(p.get("percentile_window", PERCENTILE_WINDOW))

    lookback = max(tail_size, percentile_window, 140, volatility_guard_window + 60, 80)
    df = history_df.tail(lookback).copy()
    nav = df["nav"].astype(float)

    indicators = calc_indicators(
        history_df,
        tail_size=lookback,
        percentile_window=min(percentile_window, lookback),
        params=params,
    )

    df["ma20"] = nav.rolling(window=20).mean()
    df["ma60"] = nav.rolling(window=60).mean()
    df["ma120"] = nav.rolling(window=120).mean()
    returns = nav.pct_change()
    vol20_series = returns.rolling(window=20).std()
    vol60_series = returns.rolling(window=60).std()

    current_nav = float(nav.iloc[-1])
    ma20 = float(df["ma20"].iloc[-1]) if pd.notna(df["ma20"].iloc[-1]) else None
    ma60 = float(df["ma60"].iloc[-1]) if pd.notna(df["ma60"].iloc[-1]) else None
    ma120 = float(df["ma120"].iloc[-1]) if pd.notna(df["ma120"].iloc[-1]) else None
    vol20 = float(vol20_series.iloc[-1]) if pd.notna(vol20_series.iloc[-1]) else 0.0
    vol60_window = vol60_series.tail(volatility_guard_window)
    vol60_median = float(vol60_window.median()) if vol60_window.notna().any() else vol20

    high60 = float(nav.tail(60).max())
    drawdown_60d = (high60 - current_nav) / high60 if high60 > 0 else 0.0
    nav120 = nav.tail(120)
    low120 = float(nav120.min())
    high120 = float(nav120.max())
    range_position_120 = (current_nav - low120) / (high120 - low120) if high120 > low120 else 0.5

    trend_tail = df.tail(trend_confirm_window)
    trend_confirmed = (
        len(trend_tail) == trend_confirm_window
        and trend_tail["ma60"].notna().all()
        and bool((trend_tail["nav"] > trend_tail["ma60"]).all())
    )
    weak_breakdown = (
        len(trend_tail) == trend_confirm_window
        and trend_tail["ma60"].notna().all()
        and bool((trend_tail["nav"] < trend_tail["ma60"]).all())
        and vol20 > max(vol60_median, 0.0)
    )

    reentry_tail = df.tail(reentry_window + 1)
    recent_below_ma20 = (
        len(reentry_tail) >= 2
        and reentry_tail["ma20"].iloc[:-1].notna().all()
        and bool((reentry_tail["nav"].iloc[:-1] <= reentry_tail["ma20"].iloc[:-1]).any())
    )

    stable_trend_buy = bool(
        current_nav > (ma60 or current_nav)
        and trend_confirmed
        and (vol60_median <= 0 or vol20 <= vol60_median)
    )
    pullback_recovery_buy = bool(
        drawdown_60d >= pullback_buy_threshold
        and indicators.get("above_ma20", False)
        and recent_below_ma20
    )
    drawdown_exit = bool(drawdown_60d >= drawdown_exit_threshold and not indicators.get("above_ma20", False))
    sell_signal = bool(weak_breakdown or drawdown_exit)

    if sell_signal:
        action = "SELL"
    elif stable_trend_buy or pullback_recovery_buy:
        action = "BUY"
    else:
        action = "HOLD"

    buy_score = int(stable_trend_buy) + int(pullback_recovery_buy)
    reasons = []
    if action == "BUY":
        if stable_trend_buy:
            reasons.append("稳定上行：NAV持续站上MA60，且20日波动低于中期波动中位数")
        if pullback_recovery_buy:
            reasons.append(f"回撤修复：近60日回撤{drawdown_60d:.2%}后重新站回MA20")
    elif action == "SELL":
        if weak_breakdown:
            reasons.append("弱势破位：连续跌破MA60，且短期波动放大")
        if drawdown_exit:
            reasons.append(f"阶段回撤达到{drawdown_60d:.2%}，触发保护性减仓")
    else:
        reasons.append("走势平稳，无需主动加减仓")

    reasons.append(f"20日波动={vol20:.2%}，60日波动中位数={vol60_median:.2%}")
    reasons.append(f"120日区间位置={range_position_120:.2f}")

    indicators.update(
        {
            "ma120": ma120,
            "vol20": vol20,
            "vol60_median": vol60_median,
            "range_position_120": range_position_120,
            "drawdown_60d": drawdown_60d,
        }
    )

    return {
        "action": action,
        "reason": "；".join(reasons),
        "indicators": indicators,
        "buy_score": buy_score,
        "sell_signal": sell_signal,
        "strategy": "bond_stability",
        "stable_trend_buy": stable_trend_buy,
        "pullback_recovery_buy": pullback_recovery_buy,
    }


class BondStabilityStrategy:
    """纯债稳健策略。"""

    def __init__(self, account: Account, fund_code: str, params: Dict[str, Any] | None = None):
        self.account = account
        self.fund_code = fund_code
        self._params = params or {}
        self.bar_count = 0
        self.last_buy_bar = -999

        self.p_max_position_ratio = float(self._params.get("max_position_ratio", MAX_POSITION_RATIO))
        self.p_dca_base_ratio = float(self._params.get("dca_base_ratio", DCA_BASE_RATIO))
        self.p_dca_interval = int(self._params.get("dca_interval", DCA_INTERVAL))
        self.p_min_history_days = int(self._params.get("min_history_days", MIN_HISTORY_DAYS))

    def on_bar(self, current_date: date, history_df: pd.DataFrame) -> Dict[str, Any]:
        if len(history_df) < self.p_min_history_days:
            return {}

        self.bar_count += 1
        signal = evaluate_bond_stability_signal(history_df, params=self._params)
        position = self.account.get_position(self.fund_code)
        nav = float(signal["indicators"].get("current_nav", history_df.iloc[-1]["nav"]))

        if signal["action"] == "SELL" and position.total_shares > 1e-9:
            return {"sell_shares": position.total_shares}

        if signal["action"] != "BUY":
            return {}

        if self.bar_count - self.last_buy_bar < self.p_dca_interval:
            return {}

        multiplier = 1.25 if signal.get("pullback_recovery_buy") else 1.0
        buy_amount = self._apply_position_limit(self._calc_buy_amount(position, nav) * multiplier, position, nav)
        if buy_amount < 1.0:
            return {}

        self.last_buy_bar = self.bar_count
        return {"buy_amount": buy_amount}

    def _calc_buy_amount(self, position: Any, nav: float) -> float:
        total_assets = self.account.cash + position.total_shares * nav
        return total_assets * self.p_dca_base_ratio

    def _apply_position_limit(self, raw_amount: float, position: Any, nav: float) -> float:
        current_market_value = position.total_shares * nav
        total_assets = self.account.cash + current_market_value
        max_market_value = total_assets * self.p_max_position_ratio
        room = max_market_value - current_market_value
        capped_amount = min(raw_amount, room, self.account.cash)
        return max(capped_amount, 0.0)
