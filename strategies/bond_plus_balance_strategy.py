from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pandas as pd

from ..analysis.indicators import calc_indicators
from ..core.account import Account


MAX_POSITION_RATIO = 0.68
DCA_BASE_RATIO = 0.03
DCA_INTERVAL = 8
DRAWDOWN_GUARD_THRESHOLD = 0.05
MOMENTUM_CONFIRM_THRESHOLD = 1
MACD_NEGATIVE_DAYS = 4
BATCH_RATIOS = [0.12, 0.08]
MIN_COOLDOWN = 8
MAX_COOLDOWN = 20
MIN_HISTORY_DAYS = 140
TAIL_SIZE = 240
BATCH_INTERVAL = 3


def evaluate_bond_plus_balance_signal(history_df: pd.DataFrame, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """固收+平衡策略信号评估。"""
    p = params or {}
    min_history_days = int(p.get("min_history_days", MIN_HISTORY_DAYS))
    if len(history_df) < min_history_days:
        return {
            "action": "HOLD",
            "reason": f"历史数据不足{min_history_days}天，无法生成信号",
            "indicators": {},
            "buy_score": 0,
            "sell_signal": False,
            "strategy": "bond_plus_balance",
            "trend_buy": False,
            "pullback_buy": False,
        }

    drawdown_guard_threshold = float(p.get("drawdown_guard_threshold", DRAWDOWN_GUARD_THRESHOLD))
    momentum_confirm_threshold = int(p.get("momentum_confirm_threshold", MOMENTUM_CONFIRM_THRESHOLD))
    macd_negative_days = int(p.get("macd_negative_days", MACD_NEGATIVE_DAYS))
    tail_size = int(p.get("tail_size", TAIL_SIZE))

    lookback = max(tail_size, 160)
    df = history_df.tail(lookback).copy()
    nav = df["nav"].astype(float)
    indicators = calc_indicators(history_df, tail_size=lookback, percentile_window=min(220, lookback), params=params)

    df["ma20"] = nav.rolling(window=20).mean()
    df["ma60"] = nav.rolling(window=60).mean()
    df["ma120"] = nav.rolling(window=120).mean()
    current_nav = float(nav.iloc[-1])
    ma60 = float(df["ma60"].iloc[-1]) if pd.notna(df["ma60"].iloc[-1]) else None
    ma120 = float(df["ma120"].iloc[-1]) if pd.notna(df["ma120"].iloc[-1]) else None

    current_above_ma60 = bool(ma60 is not None and current_nav > ma60)
    medium_trend_ok = bool(ma120 is None or ma60 is None or ma60 >= ma120)
    drawdown_base = float(nav.tail(60).max())
    drawdown_60d = (drawdown_base - current_nav) / drawdown_base if drawdown_base > 0 else 0.0
    mom_10d = float(indicators.get("mom_10d", 0.0))
    rsi = float(indicators.get("rsi", 50.0))

    macd_negative = bool((df["nav"].ewm(span=12, adjust=False).mean() - df["nav"].ewm(span=26, adjust=False).mean())
                         .sub((df["nav"].ewm(span=12, adjust=False).mean() - df["nav"].ewm(span=26, adjust=False).mean()).ewm(span=9, adjust=False).mean())
                         .tail(macd_negative_days).lt(0).all())

    confirmations = 0
    if indicators.get("above_ma20", False):
        confirmations += 1
    if indicators.get("macd_turn_positive", False):
        confirmations += 1
    if mom_10d > 0:
        confirmations += 1
    if 45 <= rsi <= 65:
        confirmations += 1

    trend_buy = bool(current_above_ma60 and medium_trend_ok and confirmations >= momentum_confirm_threshold)
    pullback_buy = bool(
        drawdown_60d >= drawdown_guard_threshold * 0.5
        and indicators.get("above_ma20", False)
        and indicators.get("macd_turn_positive", False)
    )
    weak_exit = bool((not current_above_ma60) and macd_negative and float(indicators.get("mom_5d", 0.0)) < 0)
    drawdown_exit = bool(drawdown_60d >= drawdown_guard_threshold and not indicators.get("above_ma20", False))
    sell_signal = bool(weak_exit or drawdown_exit)

    if sell_signal:
        action = "SELL"
    elif trend_buy or pullback_buy:
        action = "BUY"
    else:
        action = "HOLD"

    reasons = []
    if action == "BUY":
        if trend_buy:
            reasons.append(f"趋势确认：站上MA60，确认信号 {confirmations}/4")
        if pullback_buy:
            reasons.append(f"回撤修复：近60日回撤{drawdown_60d:.2%}后重新转强")
    elif action == "SELL":
        if weak_exit:
            reasons.append("跌破MA60且MACD连续转弱，触发防守卖出")
        if drawdown_exit:
            reasons.append(f"阶段回撤达到{drawdown_60d:.2%}，触发回撤保护")
    else:
        reasons.append(f"确认信号 {confirmations}/4，未形成有效趋势")

    reasons.append(f"RSI={rsi:.1f}，10日动量={mom_10d:.2%}")

    indicators.update(
        {
            "ma120": ma120,
            "drawdown_60d": drawdown_60d,
            "confirmations": confirmations,
            "macd_negative_days_hit": macd_negative,
        }
    )

    return {
        "action": action,
        "reason": "；".join(reasons),
        "indicators": indicators,
        "buy_score": confirmations,
        "sell_signal": sell_signal,
        "strategy": "bond_plus_balance",
        "trend_buy": trend_buy,
        "pullback_buy": pullback_buy,
    }


class BondPlusBalanceStrategy:
    """固收+平衡策略。"""

    def __init__(self, account: Account, fund_code: str, params: Dict[str, Any] | None = None):
        self.account = account
        self.fund_code = fund_code
        self._params = params or {}
        self.bar_count = 0
        self.last_cycle_bar = -999
        self.cooldown_remaining = 0
        self.pending_batches: List[Dict[str, float]] = []

        self.p_max_position_ratio = float(self._params.get("max_position_ratio", MAX_POSITION_RATIO))
        self.p_dca_base_ratio = float(self._params.get("dca_base_ratio", DCA_BASE_RATIO))
        self.p_dca_interval = int(self._params.get("dca_interval", DCA_INTERVAL))
        self.p_drawdown_guard_threshold = float(self._params.get("drawdown_guard_threshold", DRAWDOWN_GUARD_THRESHOLD))
        self.p_momentum_confirm_threshold = int(self._params.get("momentum_confirm_threshold", MOMENTUM_CONFIRM_THRESHOLD))
        self.p_batch_ratios = list(self._params.get("batch_ratios", BATCH_RATIOS))
        self.p_min_cooldown = int(self._params.get("min_cooldown", MIN_COOLDOWN))
        self.p_max_cooldown = int(self._params.get("max_cooldown", MAX_COOLDOWN))
        self.p_min_history_days = int(self._params.get("min_history_days", MIN_HISTORY_DAYS))

    def on_bar(self, current_date: date, history_df: pd.DataFrame) -> Dict[str, Any]:
        if len(history_df) < self.p_min_history_days:
            return {}

        self.bar_count += 1
        signal = evaluate_bond_plus_balance_signal(history_df, params=self._params)
        position = self.account.get_position(self.fund_code)
        nav = float(signal["indicators"].get("current_nav", history_df.iloc[-1]["nav"]))

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if signal["action"] == "SELL" and position.total_shares > 1e-9:
            self.pending_batches = []
            self.cooldown_remaining = self.p_max_cooldown
            return {"sell_shares": position.total_shares}

        batch_signal = self._process_pending_batches(signal["indicators"], position, nav)
        if batch_signal:
            return batch_signal

        if self.cooldown_remaining > 0 or signal["action"] != "BUY":
            return {}

        if signal["buy_score"] < self.p_momentum_confirm_threshold:
            return {}
        if self.bar_count - self.last_cycle_bar < self.p_dca_interval:
            return {}

        self.last_cycle_bar = self.bar_count
        self._create_batch_plan()
        return self._execute_next_batch(position, nav)

    def _create_batch_plan(self):
        self.pending_batches = [
            {"trigger_bar": self.bar_count + idx * BATCH_INTERVAL, "cash_ratio": ratio}
            for idx, ratio in enumerate(self.p_batch_ratios)
        ]

    def _execute_next_batch(self, position: Any, nav: float) -> Dict[str, Any]:
        if not self.pending_batches:
            return {}
        batch = self.pending_batches.pop(0)
        raw_amount = self.account.cash * float(batch["cash_ratio"])
        buy_amount = self._apply_position_limit(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}
        return {"buy_amount": buy_amount}

    def _process_pending_batches(self, indicators: Dict[str, Any], position: Any, nav: float) -> Dict[str, Any]:
        if not self.pending_batches:
            return {}
        next_batch = self.pending_batches[0]
        if self.bar_count < int(next_batch["trigger_bar"]):
            return {}
        if not indicators.get("above_ma20", True):
            self.pending_batches = []
            return {}
        return self._execute_next_batch(position, nav)

    def _apply_position_limit(self, raw_amount: float, position: Any, nav: float) -> float:
        current_market_value = position.total_shares * nav
        total_assets = self.account.cash + current_market_value
        max_market_value = total_assets * self.p_max_position_ratio
        room = max_market_value - current_market_value
        capped_amount = min(raw_amount, room, self.account.cash)
        return max(capped_amount, 0.0)
