from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pandas as pd

from ..analysis.indicators import calc_indicators, count_buy_signals, detect_regime
from ..core.account import Account
from .regime_adaptive_strategy import RegimeAdaptiveStrategy


REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"

MAX_POSITION_RATIO = 0.95
SOFT_STOP_LOSS = -0.18
HARD_STOP_LOSS = -0.25
MIN_COOLDOWN = 8
MAX_COOLDOWN = 25
DCA_BASE_RATIO = 0.06
DCA_INTERVAL = 7
CHEAP_PERCENTILE = 0.20
BATCH_INTERVAL = 3
MIN_HISTORY_DAYS = 250
TREND_BATCH_RATIOS = [0.25, 0.20, 0.15]
RE_ENTRY_STRONG_RATIO = 0.85
RE_ENTRY_WEAK_RATIO = 0.65
LOW_POSITION_RATIO = 0.30
RSI_SELL_THRESHOLD = 70
ADX_TREND_THRESHOLD = 25
BULL_DCA_BOOST = 1.6
TRANSITION_BUY_RATIO = 0.20
TRANSITION_SELL_RATIO = 0.10
MOMENTUM_ENTRY_THRESHOLD = 2
FOLLOW_THROUGH_MIN_SCORE = 1
TAIL_SIZE = 300
PERCENTILE_WINDOW = 250
TRANSITION_SELL_REQUIRES_BREAKDOWN = True
BULL_FOLLOW_THROUGH_ENTRY_ENABLED = True


def _is_follow_through_entry(indicators: Dict[str, Any], regime: str, signal_count: int, params: Dict[str, Any]) -> bool:
    if regime != REGIME_BULL:
        return False
    if signal_count < int(params.get("follow_through_min_score", FOLLOW_THROUGH_MIN_SCORE)):
        return False
    if not indicators.get("above_ma20_3d", False):
        return False
    if indicators.get("macd_hist", 0.0) <= 0:
        return False
    # 仅 RSI 舒适区不算趋势确认，至少需要 MACD 转正、金叉或中短期动量同步向上。
    if indicators.get("gold_cross", False) or indicators.get("macd_turn_positive", False):
        return True
    return bool(indicators.get("mom_5d", 0.0) > 0 and indicators.get("mom_10d", 0.0) > 0)


def _should_trim_on_transition(indicators: Dict[str, Any], regime: str) -> bool:
    if regime != REGIME_BEAR:
        return False
    return (not indicators.get("above_ma20", True)) or indicators.get("macd_5d_negative", False)


def _active_equity_cn_sell_signal(indicators: Dict[str, Any], regime: str, params: Dict[str, Any]) -> bool:
    rsi_threshold = float(params.get("rsi_sell_threshold", RSI_SELL_THRESHOLD))
    if indicators.get("death_cross", False) and indicators.get("macd_5d_negative", False):
        return True
    if regime == REGIME_BEAR and not indicators.get("above_ma20", True):
        if indicators.get("macd_5d_negative", False):
            return True
        if float(indicators.get("rsi", 50.0)) >= max(rsi_threshold - 10, 58):
            return True
    if (
        float(indicators.get("mom_5d", 0.0)) < 0
        and float(indicators.get("mom_10d", 0.0)) < 0
        and not indicators.get("above_ma20", True)
        and indicators.get("macd_5d_negative", False)
    ):
        return True
    return False


def evaluate_active_equity_cn_signal(history_df: pd.DataFrame, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """A股主动权益专门策略信号评估。"""
    p = params or {}
    min_history_days = int(p.get("min_history_days", MIN_HISTORY_DAYS))
    if len(history_df) < min_history_days:
        return {
            "action": "HOLD",
            "reason": f"历史数据不足{min_history_days}天，无法生成信号",
            "indicators": {},
            "buy_score": 0,
            "sell_signal": False,
            "strategy": "active_equity_cn",
            "follow_through": False,
        }

    tail_size = int(p.get("tail_size", TAIL_SIZE))
    percentile_window = int(p.get("percentile_window", PERCENTILE_WINDOW))
    indicators = calc_indicators(history_df, tail_size=tail_size, percentile_window=percentile_window, params=p)
    regime = detect_regime(indicators, params=p)
    signal_count = count_buy_signals(indicators, params=p)
    momentum_threshold = int(p.get("momentum_entry_threshold", MOMENTUM_ENTRY_THRESHOLD))
    trend_buy = signal_count >= momentum_threshold
    follow_through = _is_follow_through_entry(indicators, regime, signal_count, p)
    sell_signal = _active_equity_cn_sell_signal(indicators, regime, p)

    reasons = []
    if indicators["is_cheap_zone"]:
        action = "BUY"
        reasons.append(f"低估区（百分位 {indicators['percentile']:.1%}）→ 主动加仓")
        if regime == REGIME_BULL:
            reasons.append("牛市低估共振，放大仓位利用率")
    elif sell_signal:
        action = "SELL"
        if regime == REGIME_BEAR:
            reasons.append("熊市转弱确认，执行防守卖出")
        if indicators.get("death_cross", False):
            reasons.append("MA20/MA60 死叉")
        if indicators.get("macd_5d_negative", False):
            reasons.append("MACD柱连续5天为负")
        if not indicators.get("above_ma20", True):
            reasons.append("NAV跌破MA20")
    elif trend_buy or follow_through:
        action = "BUY"
        reasons.append(f"买入评分 {signal_count}/3")
        if follow_through:
            reasons.append("牛市延续确认，允许顺势跟进")
        if indicators.get("gold_cross", False):
            reasons.append("MA20/MA60 金叉")
        if indicators.get("macd_turn_positive", False):
            reasons.append("MACD柱由负转正")
        if indicators.get("above_ma20_3d", False):
            reasons.append("连续3日站上MA20")
    else:
        action = "HOLD"
        reasons.append(f"买入评分 {signal_count}/3，趋势确认不足")

    reasons.append(f"市场状态: {regime}")
    reasons.append(f"趋势: NAV {'>' if indicators.get('above_ma20', False) else '<'} MA20")

    return {
        "action": action,
        "reason": "；".join(reasons),
        "indicators": indicators,
        "buy_score": signal_count,
        "sell_signal": sell_signal,
        "strategy": "active_equity_cn",
        "follow_through": follow_through,
    }


class ActiveEquityCNStrategy(RegimeAdaptiveStrategy):
    """A股主动权益专门策略。

    回测执行层直接复用已验证过的状态自适应引擎，仅将默认参数切为
    主动权益收益优先版本，避免新策略 ID 引入一套尚未成熟的执行器。
    """

    def __init__(self, account: Account, fund_code: str, params: Dict[str, Any] | None = None):
        merged_params = {
            "max_position_ratio": MAX_POSITION_RATIO,
            "soft_stop_loss": SOFT_STOP_LOSS,
            "hard_stop_loss": HARD_STOP_LOSS,
            "min_cooldown": MIN_COOLDOWN,
            "max_cooldown": MAX_COOLDOWN,
            "dca_base_ratio": DCA_BASE_RATIO,
            "dca_interval": DCA_INTERVAL,
            "cheap_percentile": CHEAP_PERCENTILE,
            "batch_interval": BATCH_INTERVAL,
            "min_history_days": MIN_HISTORY_DAYS,
            "trend_batch_ratios": list(TREND_BATCH_RATIOS),
            "re_entry_strong_ratio": RE_ENTRY_STRONG_RATIO,
            "re_entry_weak_ratio": RE_ENTRY_WEAK_RATIO,
            "low_position_ratio": LOW_POSITION_RATIO,
            "rsi_sell_threshold": RSI_SELL_THRESHOLD,
            "adx_trend_threshold": ADX_TREND_THRESHOLD,
            "bull_dca_boost": BULL_DCA_BOOST,
            "transition_buy_ratio": TRANSITION_BUY_RATIO,
            "transition_sell_ratio": TRANSITION_SELL_RATIO,
            "tail_size": TAIL_SIZE,
            "percentile_window": PERCENTILE_WINDOW,
            "transition_sell_requires_breakdown": TRANSITION_SELL_REQUIRES_BREAKDOWN,
            "bull_follow_through_entry_enabled": BULL_FOLLOW_THROUGH_ENTRY_ENABLED,
        }
        if params:
            merged_params.update(params)
        super().__init__(account, fund_code, params=merged_params)
