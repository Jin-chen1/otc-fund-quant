from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pandas as pd

from ..analysis.indicators import calc_indicators, count_buy_signals, detect_regime
from ..core.account import Account


REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"

MAX_POSITION_RATIO = 0.72
SOFT_STOP_LOSS = -0.16
HARD_STOP_LOSS = -0.23
MIN_COOLDOWN = 10
MAX_COOLDOWN = 28
DCA_BASE_RATIO = 0.03
DCA_INTERVAL = 9
CHEAP_PERCENTILE = 0.18
BATCH_INTERVAL = 4
MIN_HISTORY_DAYS = 250
TREND_BATCH_RATIOS = [0.16, 0.12, 0.10]
RE_ENTRY_STRONG_RATIO = 0.75
RE_ENTRY_WEAK_RATIO = 0.55
LOW_POSITION_RATIO = 0.25
RSI_SELL_THRESHOLD = 68
ADX_TREND_THRESHOLD = 23
BULL_DCA_BOOST = 1.2
TRANSITION_BUY_RATIO = 0.10
TRANSITION_SELL_RATIO = 0.18
MOMENTUM_ENTRY_THRESHOLD = 2
FOLLOW_THROUGH_MIN_SCORE = 2
BEAR_EXIT_RSI = 58
TAIL_SIZE = 320
PERCENTILE_WINDOW = 220


def _is_follow_through_entry(indicators: Dict[str, Any], regime: str, signal_count: int, params: Dict[str, Any]) -> bool:
    if regime != REGIME_BULL:
        return False
    if signal_count < int(params.get("follow_through_min_score", FOLLOW_THROUGH_MIN_SCORE)):
        return False
    if not indicators.get("above_ma20_3d", False):
        return False
    if indicators.get("macd_hist", 0.0) <= 0:
        return False
    if float(indicators.get("mom_10d", 0.0)) <= 0:
        return False
    return bool(indicators.get("gold_cross", False) or indicators.get("macd_turn_positive", False))


def _active_equity_hk_sell_signal(indicators: Dict[str, Any], regime: str, params: Dict[str, Any]) -> bool:
    rsi_threshold = float(params.get("rsi_sell_threshold", RSI_SELL_THRESHOLD))
    bear_exit_rsi = float(params.get("bear_exit_rsi", BEAR_EXIT_RSI))
    if indicators.get("death_cross", False) and indicators.get("macd_5d_negative", False):
        return True
    if not indicators.get("above_ma20", True) and indicators.get("macd_5d_negative", False):
        return True
    if regime == REGIME_BEAR:
        if float(indicators.get("rsi", 50.0)) >= bear_exit_rsi:
            return True
        if float(indicators.get("mom_10d", 0.0)) < 0 and not indicators.get("above_ma20", True):
            return True
    if float(indicators.get("rsi", 50.0)) > max(rsi_threshold, 72) and not indicators.get("above_ma20", True):
        return True
    return False


def evaluate_active_equity_hk_signal(history_df: pd.DataFrame, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """港股主动权益专门策略信号评估。"""
    p = params or {}
    min_history_days = int(p.get("min_history_days", MIN_HISTORY_DAYS))
    if len(history_df) < min_history_days:
        return {
            "action": "HOLD",
            "reason": f"历史数据不足{min_history_days}天，无法生成信号",
            "indicators": {},
            "buy_score": 0,
            "sell_signal": False,
            "strategy": "active_equity_hk",
            "follow_through": False,
        }

    tail_size = int(p.get("tail_size", TAIL_SIZE))
    percentile_window = int(p.get("percentile_window", PERCENTILE_WINDOW))
    indicators = calc_indicators(history_df, tail_size=tail_size, percentile_window=percentile_window, params=p)
    regime = detect_regime(indicators, params=p)
    signal_count = count_buy_signals(indicators, params=p)
    momentum_threshold = int(p.get("momentum_entry_threshold", MOMENTUM_ENTRY_THRESHOLD))
    trend_buy = signal_count >= momentum_threshold and float(indicators.get("mom_10d", 0.0)) > 0
    follow_through = _is_follow_through_entry(indicators, regime, signal_count, p)
    sell_signal = _active_equity_hk_sell_signal(indicators, regime, p)

    reasons = []
    if indicators["is_cheap_zone"] and regime != REGIME_BEAR:
        action = "BUY"
        reasons.append(f"低估区（百分位 {indicators['percentile']:.1%}）→ 港股主动权益分批吸纳")
    elif sell_signal:
        action = "SELL"
        reasons.append("弱势确认，优先保护净值")
        if regime == REGIME_BEAR:
            reasons.append("处于熊市状态")
        if indicators.get("macd_5d_negative", False):
            reasons.append("MACD柱连续走弱")
        if not indicators.get("above_ma20", True):
            reasons.append("NAV跌破MA20")
    elif trend_buy or follow_through:
        action = "BUY"
        reasons.append(f"买入评分 {signal_count}/3")
        if follow_through:
            reasons.append("趋势延续确认后顺势建仓")
        reasons.append("强调中短期趋势同步")
    else:
        action = "HOLD"
        reasons.append(f"买入评分 {signal_count}/3，等待更强确认")

    reasons.append(f"市场状态: {regime}")
    reasons.append(f"趋势: NAV {'>' if indicators.get('above_ma20', False) else '<'} MA20")

    return {
        "action": action,
        "reason": "；".join(reasons),
        "indicators": indicators,
        "buy_score": signal_count,
        "sell_signal": sell_signal,
        "strategy": "active_equity_hk",
        "follow_through": follow_through,
    }


class ActiveEquityHKStrategy:
    """港股主动权益专门策略。"""

    def __init__(self, account: Account, fund_code: str, params: Dict[str, Any] | None = None):
        self.account = account
        self.fund_code = fund_code
        self._params = params or {}
        self.dca_counter = 0
        self.cooldown_remaining = 0
        self.soft_stop_triggered = False
        self.re_entry_ratio = 0.0
        self.pending_batches: List[Dict[str, float]] = []
        self.bar_count = 0
        self.prev_regime = "RANGE"
        self.current_regime = "RANGE"

        self.p_max_position_ratio = float(self._params.get("max_position_ratio", MAX_POSITION_RATIO))
        self.p_soft_stop_loss = float(self._params.get("soft_stop_loss", SOFT_STOP_LOSS))
        self.p_hard_stop_loss = float(self._params.get("hard_stop_loss", HARD_STOP_LOSS))
        self.p_min_cooldown = int(self._params.get("min_cooldown", MIN_COOLDOWN))
        self.p_max_cooldown = int(self._params.get("max_cooldown", MAX_COOLDOWN))
        self.p_dca_base_ratio = float(self._params.get("dca_base_ratio", DCA_BASE_RATIO))
        self.p_dca_interval = int(self._params.get("dca_interval", DCA_INTERVAL))
        self.p_batch_interval = int(self._params.get("batch_interval", BATCH_INTERVAL))
        self.p_min_history_days = int(self._params.get("min_history_days", MIN_HISTORY_DAYS))
        self.p_trend_batch_ratios = list(self._params.get("trend_batch_ratios", TREND_BATCH_RATIOS))
        self.p_re_entry_strong_ratio = float(self._params.get("re_entry_strong_ratio", RE_ENTRY_STRONG_RATIO))
        self.p_re_entry_weak_ratio = float(self._params.get("re_entry_weak_ratio", RE_ENTRY_WEAK_RATIO))
        self.p_low_position_ratio = float(self._params.get("low_position_ratio", LOW_POSITION_RATIO))
        self.p_transition_buy_ratio = float(self._params.get("transition_buy_ratio", TRANSITION_BUY_RATIO))
        self.p_transition_sell_ratio = float(self._params.get("transition_sell_ratio", TRANSITION_SELL_RATIO))
        self.p_momentum_entry_threshold = int(self._params.get("momentum_entry_threshold", MOMENTUM_ENTRY_THRESHOLD))
        self.p_tail_size = int(self._params.get("tail_size", TAIL_SIZE))
        self.p_percentile_window = int(self._params.get("percentile_window", PERCENTILE_WINDOW))
        self.p_bull_dca_boost = float(self._params.get("bull_dca_boost", BULL_DCA_BOOST))

        self.base_dca_amount = account.cash * self.p_dca_base_ratio

    def on_bar(self, current_date: date, history_df: pd.DataFrame) -> Dict[str, Any]:
        if len(history_df) < self.p_min_history_days:
            return {}

        self.bar_count += 1
        indicators = calc_indicators(history_df, tail_size=self.p_tail_size, percentile_window=self.p_percentile_window, params=self._params)
        position = self.account.get_position(self.fund_code)
        nav = indicators["current_nav"]
        signal_count = count_buy_signals(indicators, params=self._params)
        self.prev_regime = self.current_regime
        self.current_regime = detect_regime(indicators, params=self._params)

        stop_signal = self._check_stop_loss(position, nav)
        if stop_signal:
            return stop_signal

        batch_signal = self._process_pending_batches(indicators, position, nav)
        if batch_signal:
            return batch_signal

        self._update_cooldown(indicators)
        transition_signal = self._check_transition_signal(indicators, position, nav)
        if transition_signal:
            return transition_signal

        if self.cooldown_remaining > 0:
            return {}

        if indicators["is_cheap_zone"] and self.current_regime != REGIME_BEAR:
            return self._handle_cheap_zone(position, nav, indicators)
        return self._handle_normal_zone(position, nav, indicators, signal_count)

    def _check_stop_loss(self, position: Any, nav: float) -> Dict[str, Any]:
        if position.total_shares <= 0 or position.avg_cost <= 0:
            return {}
        pnl_ratio = (nav - position.avg_cost) / position.avg_cost
        if pnl_ratio <= self.p_hard_stop_loss:
            self._enter_cooldown()
            return {"sell_shares": position.total_shares}
        if pnl_ratio <= self.p_soft_stop_loss and not self.soft_stop_triggered:
            self.soft_stop_triggered = True
            sell_shares = self._apply_low_position_protection(position, nav, 0.5)
            if sell_shares > 0:
                return {"sell_shares": sell_shares}
        return {}

    def _update_cooldown(self, indicators: Dict[str, Any]):
        if self.cooldown_remaining <= 0:
            return
        self.cooldown_remaining -= 1
        days_elapsed = self.p_max_cooldown - self.cooldown_remaining
        if days_elapsed < self.p_min_cooldown:
            return
        if (
            indicators.get("above_ma20_3d", False)
            and indicators.get("macd_hist", 0.0) > 0
            and float(indicators.get("mom_10d", 0.0)) > 0
        ):
            self.cooldown_remaining = 0
            self.re_entry_ratio = 1.0

    def _check_transition_signal(self, indicators: Dict[str, Any], position: Any, nav: float) -> Dict[str, Any]:
        if self.prev_regime != self.current_regime and self.current_regime == REGIME_BULL:
            if _is_follow_through_entry(indicators, self.current_regime, max(2, count_buy_signals(indicators, params=self._params)), self._params):
                raw_amount = self.account.cash * self.p_transition_buy_ratio
                buy_amount = self._apply_position_limit(raw_amount, position, nav)
                if buy_amount >= 1.0:
                    return {"buy_amount": buy_amount}
        if self.prev_regime == REGIME_BULL and self.current_regime == REGIME_BEAR and position.total_shares > 0:
            if (not indicators.get("above_ma20", True)) or indicators.get("macd_5d_negative", False):
                sell_shares = position.total_shares * self.p_transition_sell_ratio
                if sell_shares > 0:
                    self.pending_batches = []
                    return {"sell_shares": sell_shares}
        return {}

    def _handle_cheap_zone(self, position: Any, nav: float, indicators: Dict[str, Any]) -> Dict[str, Any]:
        self.dca_counter += 1
        if self.dca_counter < self.p_dca_interval:
            return {}
        self.dca_counter = 0
        percentile = float(indicators.get("percentile", 1.0))
        multiplier = 1.6 if percentile < 0.10 else 1.3 if percentile < 0.15 else 1.0
        if self.current_regime == REGIME_BULL:
            multiplier *= self.p_bull_dca_boost
        raw_amount = self.base_dca_amount * multiplier
        buy_amount = self._apply_position_limit(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}
        return {"buy_amount": buy_amount}

    def _handle_normal_zone(self, position: Any, nav: float, indicators: Dict[str, Any], signal_count: int) -> Dict[str, Any]:
        self.dca_counter = 0
        follow_through = _is_follow_through_entry(indicators, self.current_regime, signal_count, self._params)
        if signal_count >= self.p_momentum_entry_threshold or follow_through:
            self._create_batch_plan()
            return self._execute_first_batch(position, nav, signal_count)
        if _active_equity_hk_sell_signal(indicators, self.current_regime, self._params) and position.total_shares > 0:
            self.pending_batches = []
            sell_shares = self._apply_low_position_protection(position, nav, 1.0)
            if sell_shares > 0:
                return {"sell_shares": sell_shares}
        return {}

    def _create_batch_plan(self):
        self.pending_batches = []
        for idx, ratio in enumerate(self.p_trend_batch_ratios):
            self.pending_batches.append({"trigger_bar": self.bar_count + idx * self.p_batch_interval, "cash_ratio": ratio})
        if self.pending_batches:
            self.pending_batches.pop(0)

    def _execute_first_batch(self, position: Any, nav: float, signal_count: int) -> Dict[str, Any]:
        ratio = float(self.p_trend_batch_ratios[0])
        if self.re_entry_ratio > 0:
            discount = self.p_re_entry_strong_ratio if signal_count >= 3 else self.p_re_entry_weak_ratio
            ratio *= discount
            self.re_entry_ratio = 0.0
        raw_amount = self.account.cash * ratio
        buy_amount = self._apply_position_limit(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}
        return {"buy_amount": buy_amount}

    def _process_pending_batches(self, indicators: Dict[str, Any], position: Any, nav: float) -> Dict[str, Any]:
        if not self.pending_batches:
            return {}
        batch = self.pending_batches[0]
        if self.bar_count < int(batch["trigger_bar"]):
            return {}
        if not indicators.get("above_ma20", True):
            self.pending_batches = []
            return {}
        self.pending_batches.pop(0)
        raw_amount = self.account.cash * float(batch["cash_ratio"])
        buy_amount = self._apply_position_limit(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}
        return {"buy_amount": buy_amount}

    def _apply_position_limit(self, raw_amount: float, position: Any, nav: float) -> float:
        current_market_value = position.total_shares * nav
        total_assets = self.account.cash + current_market_value
        max_market_value = total_assets * self.p_max_position_ratio
        room = max_market_value - current_market_value
        return max(min(raw_amount, room, self.account.cash), 0.0)

    def _get_position_ratio(self, position: Any, nav: float) -> float:
        total_assets = self.account.cash + position.total_shares * nav
        if total_assets <= 0:
            return 0.0
        return position.total_shares * nav / total_assets

    def _apply_low_position_protection(self, position: Any, nav: float, target_ratio: float) -> float:
        position_ratio = self._get_position_ratio(position, nav)
        sell_shares = position.total_shares * target_ratio
        if position_ratio < self.p_low_position_ratio and target_ratio > 0.5:
            sell_shares = position.total_shares * 0.5
        return sell_shares

    def _enter_cooldown(self):
        self.cooldown_remaining = self.p_max_cooldown
        self.dca_counter = 0
        self.pending_batches = []
        self.soft_stop_triggered = False
