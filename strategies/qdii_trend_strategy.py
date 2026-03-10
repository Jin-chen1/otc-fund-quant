from __future__ import annotations

from datetime import date
from typing import Any, Dict

import pandas as pd

from ..analysis.indicators import calc_indicators
from ..core.account import Account


MAX_POSITION_RATIO = 0.72
DCA_BASE_RATIO = 0.03
MOMENTUM_ENTRY_THRESHOLD = 2
TREND_WINDOW_FAST = 20
TREND_WINDOW_MID = 60
TREND_WINDOW_SLOW = 120
PULLBACK_REENTRY_THRESHOLD = 0.04
ATR_EXIT_MULTIPLIER = 1.8
SELL_COOLDOWN = 30
MIN_COOLDOWN = 10
MAX_COOLDOWN = 30
MIN_HISTORY_DAYS = 140
TAIL_SIZE = 260


def evaluate_qdii_trend_signal(history_df: pd.DataFrame, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """QDII 趋势策略信号评估。"""
    p = params or {}
    min_history_days = int(p.get("min_history_days", MIN_HISTORY_DAYS))
    if len(history_df) < min_history_days:
        return {
            "action": "HOLD",
            "reason": f"历史数据不足{min_history_days}天，无法生成信号",
            "indicators": {},
            "buy_score": 0,
            "sell_signal": False,
            "strategy": "qdii_trend",
            "trend_entry": False,
            "pullback_reentry": False,
        }

    trend_window_fast = int(p.get("trend_window_fast", TREND_WINDOW_FAST))
    trend_window_mid = int(p.get("trend_window_mid", TREND_WINDOW_MID))
    trend_window_slow = int(p.get("trend_window_slow", TREND_WINDOW_SLOW))
    pullback_reentry_threshold = float(p.get("pullback_reentry_threshold", PULLBACK_REENTRY_THRESHOLD))
    atr_exit_multiplier = float(p.get("atr_exit_multiplier", ATR_EXIT_MULTIPLIER))
    tail_size = int(p.get("tail_size", TAIL_SIZE))

    lookback = max(tail_size, trend_window_slow + 20, 160)
    df = history_df.tail(lookback).copy()
    nav = df["nav"].astype(float)
    indicators = calc_indicators(history_df, tail_size=lookback, percentile_window=min(220, lookback), params=params)

    df["ma_fast"] = nav.rolling(window=trend_window_fast).mean()
    df["ma_mid"] = nav.rolling(window=trend_window_mid).mean()
    df["ma_slow"] = nav.rolling(window=trend_window_slow).mean()

    current_nav = float(nav.iloc[-1])
    ma_fast = float(df["ma_fast"].iloc[-1]) if pd.notna(df["ma_fast"].iloc[-1]) else None
    ma_mid = float(df["ma_mid"].iloc[-1]) if pd.notna(df["ma_mid"].iloc[-1]) else None
    ma_slow = float(df["ma_slow"].iloc[-1]) if pd.notna(df["ma_slow"].iloc[-1]) else None

    above_mid = bool(ma_mid is not None and current_nav > ma_mid)
    above_slow = bool(ma_slow is not None and current_nav > ma_slow)
    drawdown_base = float(nav.tail(trend_window_mid).max())
    drawdown_mid = (drawdown_base - current_nav) / drawdown_base if drawdown_base > 0 else 0.0
    mom_5d = float(indicators.get("mom_5d", 0.0))
    mom_10d = float(indicators.get("mom_10d", 0.0))
    mom_20d = float(indicators.get("mom_20d", 0.0))
    atr = float(indicators.get("atr", 0.0))
    atr_median = float(indicators.get("atr_median", atr))

    trend_entry = bool(above_mid and mom_10d > 0 and mom_20d > 0)
    pullback_reentry = bool(
        drawdown_mid >= pullback_reentry_threshold
        and indicators.get("above_ma20", False)
        and mom_5d > 0
    )
    weak_exit = bool(
        (not above_mid)
        and (not above_slow)
        and mom_10d < 0
        and mom_20d < 0
        and drawdown_mid >= pullback_reentry_threshold * 0.5
    )
    atr_weak_exit = bool(
        atr_median > 0
        and atr >= atr_median * atr_exit_multiplier
        and mom_10d < 0
        and not indicators.get("above_ma20", False)
    )
    sell_signal = bool(weak_exit or atr_weak_exit)

    trend_score = int(above_mid) + int(above_slow) + int(mom_10d > 0) + int(mom_20d > 0)

    if sell_signal:
        action = "SELL"
    elif trend_entry or pullback_reentry:
        action = "BUY"
    else:
        action = "HOLD"

    reasons = []
    if action == "BUY":
        if trend_entry:
            reasons.append("趋势成立：NAV站上中期均线，10/20日动量为正")
        if pullback_reentry:
            reasons.append(f"回撤重入：近中期回撤{drawdown_mid:.2%}后重新转强")
    elif action == "SELL":
        if weak_exit:
            reasons.append("跌破中期趋势线，20日动量转负")
        if atr_weak_exit:
            reasons.append("ATR放大且短期动量转弱，触发防守卖出")
    else:
        reasons.append("趋势未形成，继续观察")

    reasons.append(f"趋势评分={trend_score}/4")
    reasons.append(f"ATR={atr:.4f}，ATR中位数={atr_median:.4f}")

    indicators.update(
        {
            "ma_fast": ma_fast,
            "ma_mid": ma_mid,
            "ma_slow": ma_slow,
            "drawdown_mid": drawdown_mid,
            "trend_score": trend_score,
        }
    )

    buy_score = trend_score
    if pullback_reentry and buy_score < 2:
        buy_score = 2

    return {
        "action": action,
        "reason": "；".join(reasons),
        "indicators": indicators,
        "buy_score": buy_score,
        "sell_signal": sell_signal,
        "strategy": "qdii_trend",
        "trend_entry": trend_entry,
        "pullback_reentry": pullback_reentry,
    }


class QDIITrendStrategy:
    """QDII 趋势策略。"""

    def __init__(self, account: Account, fund_code: str, params: Dict[str, Any] | None = None):
        self.account = account
        self.fund_code = fund_code
        self._params = params or {}
        self.bar_count = 0
        self.last_buy_bar = -999
        self.cooldown_remaining = 0
        self.sell_cooldown_remaining = 0

        self.p_max_position_ratio = float(self._params.get("max_position_ratio", MAX_POSITION_RATIO))
        self.p_dca_base_ratio = float(self._params.get("dca_base_ratio", DCA_BASE_RATIO))
        self.p_momentum_entry_threshold = int(self._params.get("momentum_entry_threshold", MOMENTUM_ENTRY_THRESHOLD))
        self.p_trend_window_fast = int(self._params.get("trend_window_fast", TREND_WINDOW_FAST))
        self.p_sell_cooldown = int(self._params.get("sell_cooldown", SELL_COOLDOWN))
        self.p_min_cooldown = int(self._params.get("min_cooldown", MIN_COOLDOWN))
        self.p_max_cooldown = int(self._params.get("max_cooldown", MAX_COOLDOWN))
        self.p_min_history_days = int(self._params.get("min_history_days", MIN_HISTORY_DAYS))

    def on_bar(self, current_date: date, history_df: pd.DataFrame) -> Dict[str, Any]:
        if len(history_df) < self.p_min_history_days:
            return {}

        self.bar_count += 1
        if self.sell_cooldown_remaining > 0:
            self.sell_cooldown_remaining -= 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        signal = evaluate_qdii_trend_signal(history_df, params=self._params)
        position = self.account.get_position(self.fund_code)
        nav = float(signal["indicators"].get("current_nav", history_df.iloc[-1]["nav"]))

        if self.cooldown_remaining > 0:
            if self.cooldown_remaining <= max(self.p_max_cooldown - self.p_min_cooldown, 0) and signal.get("trend_entry"):
                self.cooldown_remaining = 0
            else:
                if signal["action"] == "SELL" and position.total_shares > 1e-9 and self.sell_cooldown_remaining <= 0:
                    self.cooldown_remaining = self.p_max_cooldown
                    self.sell_cooldown_remaining = self.p_sell_cooldown
                    return {"sell_shares": position.total_shares}
                return {}

        if signal["action"] == "SELL" and position.total_shares > 1e-9 and self.sell_cooldown_remaining <= 0:
            self.cooldown_remaining = self.p_max_cooldown
            self.sell_cooldown_remaining = self.p_sell_cooldown
            return {"sell_shares": position.total_shares}

        if signal["action"] != "BUY" or signal["buy_score"] < self.p_momentum_entry_threshold:
            return {}

        min_buy_gap = max(5, self.p_trend_window_fast // 4)
        if self.bar_count - self.last_buy_bar < min_buy_gap:
            return {}

        multiplier = 1.20 if signal.get("pullback_reentry") else 1.0
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
