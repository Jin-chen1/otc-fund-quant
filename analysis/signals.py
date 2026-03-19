"""
买卖信号生成模块 — 建议生成、信号点标记、历史回顾。
所有信号判断统一调用 analysis.indicators 中的指标计算。
"""

import pandas as pd
from typing import Dict, Any

from .indicators import (
    calc_indicators,
    count_buy_signals,
    check_sell_signal,
    CHEAP_PERCENTILE,
    RSI_SELL_THRESHOLD,
)
from ..strategies.bond_plus_balance_strategy import evaluate_bond_plus_balance_signal
from ..strategies.bond_stability_strategy import evaluate_bond_stability_signal
from ..strategies.qdii_trend_strategy import evaluate_qdii_trend_signal
from ..strategies.active_equity_cn_strategy import evaluate_active_equity_cn_signal
from ..strategies.active_equity_hk_strategy import evaluate_active_equity_hk_signal

# 同方向信号最小间隔天数
SIGNAL_MIN_INTERVAL = 5


def _get_strategy_min_rows(strategy: str) -> int:
    if strategy == "index_momentum":
        return 60
    if strategy in {"bond_stability", "qdii_trend", "bond_plus_balance"}:
        return 140
    return 250


def generate_recommendation(history_df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """根据历史数据生成今日买卖建议（V6策略）。

    Returns:
        {
            "action": "BUY" | "SELL" | "HOLD",
            "reason": str,
            "indicators": dict,
            "buy_score": int,
            "sell_signal": bool,
        }
    """
    if len(history_df) < 250:
        return {
            "action": "HOLD",
            "reason": "历史数据不足250天，无法生成信号",
            "indicators": {},
            "buy_score": 0,
            "sell_signal": False,
        }

    p = params or {}
    _rsi_sell_threshold = p.get("rsi_sell_threshold", RSI_SELL_THRESHOLD)

    indicators = calc_indicators(history_df, params=params)
    buy_score = count_buy_signals(indicators, params=params)
    sell_signal = check_sell_signal(indicators, params=params)

    reasons = []

    # 判断买卖
    if indicators["is_cheap_zone"]:
        action = "BUY"
        reasons.append(f"低估区（百分位 {indicators['percentile']:.1%}）→ 定投买入")
    elif sell_signal:
        action = "SELL"
        if indicators["rsi"] > 80:
            reasons.append(f"RSI={indicators['rsi']:.1f} 极度超买")
        elif indicators["death_cross"]:
            reasons.append("MA20/MA60 死叉")
            if indicators["rsi"] > _rsi_sell_threshold:
                reasons.append(f"RSI={indicators['rsi']:.1f} 超买")
            if indicators["macd_5d_negative"]:
                reasons.append("MACD柱连续5天为负")
        elif indicators["macd_5d_negative"] and not indicators["above_ma20"]:
            reasons.append("MACD柱连续5天为负 + NAV跌破MA20")
    elif buy_score >= 2:
        action = "BUY"
        if indicators["gold_cross"]:
            reasons.append("MA20/MA60 金叉")
        if 35 <= indicators["rsi"] <= 65:
            reasons.append(f"RSI={indicators['rsi']:.1f} 适中")
        if indicators["macd_turn_positive"]:
            reasons.append("MACD柱由负转正")
        reasons.append(f"买入信号评分 {buy_score}/3")
    else:
        action = "HOLD"
        reasons.append("无明确买卖信号，建议观望")
        if indicators["gold_cross"]:
            reasons.append("金叉出现但信号不足")
        if indicators["death_cross"]:
            reasons.append("死叉出现但卖出条件不足")

    # 补充指标状态
    trend = "上升" if indicators["above_ma20"] else "下降"
    reasons.append(f"当前趋势: NAV {'>' if indicators['above_ma20'] else '<'} MA20 ({trend})")

    return {
        "action": action,
        "reason": "；".join(reasons),
        "indicators": indicators,
        "buy_score": buy_score,
        "sell_signal": sell_signal,
    }


def generate_recommendation_regime(history_df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """基于市场状态自适应策略生成今日买卖建议。

    根据 regime（BULL/BEAR/RANGE）动态调整买卖阈值。

    Returns:
        {
            "action": "BUY" | "SELL" | "HOLD",
            "reason": str,
            "indicators": dict,
            "buy_score": int,
            "sell_signal": bool,
            "strategy": "regime_adaptive",
            "regime_info": { "current": str, "description": str },
        }
    """
    if len(history_df) < 250:
        return {
            "action": "HOLD",
            "reason": "历史数据不足250天，无法生成信号",
            "indicators": {},
            "buy_score": 0,
            "sell_signal": False,
            "strategy": "regime_adaptive",
            "regime_info": {"current": "UNKNOWN", "description": "数据不足"},
        }

    p = params or {}
    _rsi_sell_threshold = p.get("rsi_sell_threshold", RSI_SELL_THRESHOLD)

    indicators = calc_indicators(history_df, params=params)
    buy_score = count_buy_signals(indicators, params=params)
    sell_signal = check_sell_signal(indicators, params=params)
    regime = indicators.get("market_regime", "RANGE")

    regime_desc_map = {
        "BULL": "趋势上涨（ADX>25 且 NAV>MA60）",
        "BEAR": "趋势下跌（ADX>25 且 NAV<MA60）",
        "RANGE": "震荡盘整（ADX≤25，无明确趋势）",
    }


    reasons = []

    # --- 策略逻辑：根据 regime 调整买卖阈值 ---
    # 牛市：买入阈值降低(1分但需站上MA20确认)，卖出更严格(需RSI>75)
    # 熊市：买入阈值保持2分但需额外确认(MACD转正)，卖出更灵敏(RSI>60即卖)
    # 震荡：标准阈值(2分买入)，标准卖出

    if regime == "BULL":
        buy_threshold = 1
        rsi_sell = 75
    elif regime == "BEAR":
        buy_threshold = 2
        rsi_sell = 60
    else:  # RANGE
        buy_threshold = 2
        rsi_sell = _rsi_sell_threshold  # 70

    # 熊市卖出更灵敏：增加额外卖出条件
    bear_sell = False
    if regime == "BEAR" and not indicators["is_cheap_zone"]:
        if indicators["rsi"] > rsi_sell:
            bear_sell = True
        if indicators["death_cross"] and indicators["macd_5d_negative"]:
            bear_sell = True

    # 低估区定投（所有状态执行）
    if indicators["is_cheap_zone"]:
        action = "BUY"
        reasons.append(f"低估区（百分位 {indicators['percentile']:.1%}）→ 定投买入")
        if regime == "BULL":
            reasons.append("【牛市加速】低估+牛市，加大定投")
        elif regime == "BEAR":
            reasons.append("【熊市防守】低估区坚持定投，控制仓位")
    # 熊市额外卖出
    elif bear_sell:
        action = "SELL"
        reasons.append(f"【熊市防守卖出】RSI阈值降至{rsi_sell}")
        if indicators["rsi"] > rsi_sell:
            reasons.append(f"RSI={indicators['rsi']:.1f} 超过熊市阈值{rsi_sell}")
        if indicators["death_cross"]:
            reasons.append("MA20/MA60 死叉")
        if indicators["macd_5d_negative"]:
            reasons.append("MACD柱连续5天为负")
    # 卖出信号（多路径）
    elif sell_signal:
        action = "SELL"
        if indicators["rsi"] > 80:
            reasons.append(f"RSI={indicators['rsi']:.1f} 极度超买")
        elif indicators["death_cross"]:
            reasons.append("MA20/MA60 死叉")
            if indicators["rsi"] > _rsi_sell_threshold:
                reasons.append(f"RSI={indicators['rsi']:.1f} 超买")
            if indicators["macd_5d_negative"]:
                reasons.append("MACD柱连续5天为负")
        elif indicators["macd_5d_negative"] and not indicators["above_ma20"]:
            reasons.append("MACD柱连续5天为负 + NAV跌破MA20")
        if regime == "BULL":
            reasons.append("【牛市回调】卖出部分仓位")
        elif regime == "BEAR":
            reasons.append("【熊市确认】卖出大部分仓位")
        else:
            reasons.append("【震荡卖出】卖出部分仓位")
    # 趋势买入（阈值随 regime 变化）
    elif buy_score >= buy_threshold:
        # 牛市1分买入需额外确认：必须站上MA20
        if regime == "BULL" and buy_score == 1 and not indicators["above_ma20"]:
            action = "HOLD"
            reasons.append(f"信号评分 {buy_score}/3，牛市降低门槛但NAV未站上MA20，等待确认")
        # 熊市2分买入需额外确认：必须MACD转正
        elif regime == "BEAR" and buy_score == 2 and not indicators["macd_turn_positive"]:
            action = "HOLD"
            reasons.append(f"信号评分 {buy_score}/3，熊市需MACD转正确认，继续观望")
        else:
            action = "BUY"
            if indicators["gold_cross"]:
                reasons.append("MA20/MA60 金叉")
            if 35 <= indicators["rsi"] <= 65:
                reasons.append(f"RSI={indicators['rsi']:.1f} 适中")
            if indicators["macd_turn_positive"]:
                reasons.append("MACD柱由负转正")
            reasons.append(f"买入信号评分 {buy_score}/3 (阈值{buy_threshold})")
            if regime == "BULL":
                reasons.append("【牛市趋势】降低买入门槛，跟随趋势")
            elif regime == "BEAR":
                reasons.append("【熊市谨慎】信号确认，谨慎入场")
    else:
        action = "HOLD"
        reasons.append(f"信号评分 {buy_score}/3，未达{regime}阈值{buy_threshold}，观望")
        if indicators["gold_cross"]:
            reasons.append("金叉出现但信号不足")
        if indicators["death_cross"]:
            reasons.append("死叉出现但卖出条件不足")

    # 补充市场状态和趋势信息
    reasons.append(f"市场状态: {regime_desc_map.get(regime, '未知')}")
    trend = "上升" if indicators["above_ma20"] else "下降"
    reasons.append(f"当前趋势: NAV {'>' if indicators['above_ma20'] else '<'} MA20 ({trend})")
    adx_val = indicators.get("adx", 0)
    atr_val = indicators.get("atr", 0)
    reasons.append(f"ADX={adx_val:.1f} ATR={atr_val:.4f}")

    return {
        "action": action,
        "reason": "；".join(reasons),
        "indicators": indicators,
        "buy_score": buy_score,
        "sell_signal": sell_signal,
        "strategy": "regime_adaptive",
        "regime_info": {
            "current": regime,
            "description": regime_desc_map.get(regime, "未知"),
        },
    }


def generate_recommendation_bond_stability(history_df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, Any]:
    return evaluate_bond_stability_signal(history_df, params=params)


def generate_recommendation_qdii_trend(history_df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, Any]:
    return evaluate_qdii_trend_signal(history_df, params=params)


def generate_recommendation_bond_plus_balance(history_df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, Any]:
    return evaluate_bond_plus_balance_signal(history_df, params=params)


def generate_recommendation_active_equity_cn(history_df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, Any]:
    return evaluate_active_equity_cn_signal(history_df, params=params)


def generate_recommendation_active_equity_hk(history_df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, Any]:
    return evaluate_active_equity_hk_signal(history_df, params=params)


def validate_recommendation_payload(recommendation: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    """校验并归一化策略生成器返回值，避免下游处理时结构不稳定。"""
    if not isinstance(recommendation, dict):
        raise ValueError(f"策略 {strategy} 返回了非法推荐结果类型: {type(recommendation).__name__}")

    normalized = dict(recommendation)
    missing_keys = {"action", "reason"} - set(normalized.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"策略 {strategy} 返回的推荐结果缺少字段: {missing}")

    action = str(normalized.get("action", "")).strip().upper()
    if action not in {"BUY", "SELL", "HOLD"}:
        raise ValueError(f"策略 {strategy} 返回了非法 action: {normalized.get('action')}")
    normalized["action"] = action

    reason = normalized.get("reason")
    if reason is None:
        raise ValueError(f"策略 {strategy} 返回的推荐结果缺少有效 reason")
    normalized["reason"] = str(reason).strip() or "无明确理由"

    buy_score = normalized.get("buy_score", 0)
    try:
        normalized["buy_score"] = int(buy_score or 0)
    except (TypeError, ValueError):
        normalized["buy_score"] = 0

    normalized["sell_signal"] = bool(normalized.get("sell_signal", False))

    indicators = normalized.get("indicators")
    if indicators is None:
        normalized["indicators"] = {}
    elif not isinstance(indicators, dict):
        raise ValueError(f"策略 {strategy} 返回了非法 indicators 类型: {type(indicators).__name__}")
    else:
        normalized["indicators"] = indicators

    return normalized


def generate_weekly_review(history_df: pd.DataFrame, days: int = 7,
                           strategy: str = "v6",
                           params: Dict[str, Any] = None) -> list:
    """回溯最近N个交易日的信号判断，并用实际次日NAV验证对错。

    Args:
        history_df: 完整历史数据（含 date, nav 列）
        days: 回溯天数（默认7天）
        strategy: 策略名

    Returns:
        列表，每项包含 date, action, reason, nav, next_nav, change_pct, correct
    """
    min_rows = _get_strategy_min_rows(strategy)
    if len(history_df) < min_rows + 10:
        return []

    _rec_fn = _get_rec_fn(strategy)
    results = []
    total = len(history_df)

    for i in range(days + 1, 1, -1):
        if total - i < min_rows:
            continue

        # 用截至第 total-i 天的数据生成信号
        hist_slice = history_df.iloc[: total - i + 1]
        rec = validate_recommendation_payload(_rec_fn(hist_slice, params=params), strategy)

        signal_date = hist_slice.iloc[-1]["date"]
        signal_nav = float(hist_slice.iloc[-1]["nav"])

        # 次日NAV（信号发出后的下一个交易日）
        next_idx = total - i + 1
        if next_idx >= total:
            continue
        next_nav = float(history_df.iloc[next_idx]["nav"])
        change_pct = (next_nav - signal_nav) / signal_nav * 100

        # 判断对错
        action = rec["action"]
        if action == "BUY":
            correct = change_pct > 0
        elif action == "SELL":
            correct = change_pct < 0
        else:
            correct = None  # HOLD 不评判

        date_str = signal_date.strftime("%Y-%m-%d") if hasattr(signal_date, "strftime") else str(signal_date)

        results.append({
            "date": date_str,
            "action": action,
            "reason": rec["reason"],
            "nav": round(signal_nav, 4),
            "next_nav": round(next_nav, 4),
            "change_pct": round(change_pct, 2),
            "correct": correct,
        })

    return results


def generate_signal_points(history_df: pd.DataFrame, days: int = 1250,
                           after_date: str = None,
                           strategy: str = "v6",
                           params: Dict[str, Any] = None) -> Dict[str, Any]:
    """回溯最近N个交易日，标记每天的买卖信号点。默认5年(约1250个交易日)。

    Args:
        history_df: 完整历史数据
        days: 最多回溯多少天
        after_date: 增量模式 - 只计算此日期之后的信号（格式 "YYYY-MM-DD"）
        strategy: 策略名

    Returns:
        {
            "buy_dates": [...], "buy_navs": [...],
            "sell_dates": [...], "sell_navs": [...],
            "records": [{"date", "action", "reason", "nav", "next_nav", "correct"}, ...]
        }
    """
    empty = {"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": [], "records": []}
    min_rows = _get_strategy_min_rows(strategy)
    if len(history_df) < min_rows + 10:
        return empty

    _rec_fn = _get_rec_fn(strategy)

    buy_dates, buy_navs = [], []
    sell_dates, sell_navs = [], []
    records = []
    total = len(history_df)
    # 实际可回溯天数
    max_days = total - min_rows
    scan_days = min(days, max_days)

    # 信号去重：记录上次同方向信号的索引
    last_buy_idx = -999
    last_sell_idx = -999

    for i in range(scan_days, 0, -1):
        idx = total - i
        if idx < min_rows:
            continue

        d = history_df.iloc[idx]["date"]
        date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)

        # 增量模式：跳过已计算的日期
        if after_date and date_str <= after_date:
            continue

        hist_slice = history_df.iloc[:idx + 1]
        rec = validate_recommendation_payload(_rec_fn(hist_slice, params=params), strategy)
        nav_val = float(hist_slice.iloc[-1]["nav"])

        if rec["action"] == "BUY":
            # 信号去重：距上次BUY不足N天则跳过
            if idx - last_buy_idx < SIGNAL_MIN_INTERVAL:
                continue
            last_buy_idx = idx
            buy_dates.append(date_str)
            buy_navs.append(nav_val)
        elif rec["action"] == "SELL":
            # 信号去重：距上次SELL不足N天则跳过
            if idx - last_sell_idx < SIGNAL_MIN_INTERVAL:
                continue
            last_sell_idx = idx
            sell_dates.append(date_str)
            sell_navs.append(nav_val)
        else:
            continue  # HOLD 不记录

        # 多周期验证：看3天/5天/10天收益，任一为正即判正确
        correct = None
        best_next_nav = None
        for offset in [1, 3, 5, 10]:
            check_idx = idx + offset
            if check_idx < total:
                future_nav = float(history_df.iloc[check_idx]["nav"])
                if offset == 1:
                    best_next_nav = future_nav  # 次日NAV仍用于展示
                if rec["action"] == "BUY" and future_nav > nav_val:
                    correct = True
                elif rec["action"] == "SELL" and future_nav < nav_val:
                    correct = True
        # 如果所有周期都不满足且有数据，则判错
        if correct is None and idx + 1 < total:
            correct = False

        records.append({
            "date": date_str,
            "action": rec["action"],
            "reason": rec["reason"],
            "nav": round(nav_val, 4),
            "next_nav": round(best_next_nav, 4) if best_next_nav is not None else None,
            "correct": correct,
        })

    return {
        "buy_dates": buy_dates,
        "buy_navs": buy_navs,
        "sell_dates": sell_dates,
        "sell_navs": sell_navs,
        "records": records,
    }


def _get_rec_fn(strategy: str):
    """根据策略名返回对应的推荐函数。"""
    if strategy == "regime_adaptive":
        return generate_recommendation_regime
    if strategy == "index_momentum":
        return generate_recommendation_index_momentum
    if strategy == "bond_stability":
        return generate_recommendation_bond_stability
    if strategy == "qdii_trend":
        return generate_recommendation_qdii_trend
    if strategy == "bond_plus_balance":
        return generate_recommendation_bond_plus_balance
    if strategy == "active_equity_cn":
        return generate_recommendation_active_equity_cn
    if strategy == "active_equity_hk":
        return generate_recommendation_active_equity_hk
    return generate_recommendation


def generate_recommendation_index_momentum(history_df: pd.DataFrame,
                                           params: Dict[str, Any] = None) -> Dict[str, Any]:
    """基于指数动量策略生成今日买卖建议。

    核心逻辑：布林带区域划分 + 多维动量评分 + ATR自适应。

    Returns:
        {
            "action": "BUY" | "SELL" | "HOLD",
            "reason": str,
            "indicators": dict,
            "buy_score": int,      # 动量评分
            "sell_signal": bool,
            "strategy": "index_momentum",
            "bb_zone": str,        # OVERSOLD / NORMAL / OVERBOUGHT
        }
    """
    if len(history_df) < 60:
        return {
            "action": "HOLD",
            "reason": "历史数据不足60天，无法生成信号",
            "indicators": {},
            "buy_score": 0,
            "sell_signal": False,
            "strategy": "index_momentum",
            "bb_zone": "UNKNOWN",
        }

    p = params or {}
    rsi_overbought = p.get("rsi_overbought", 75)
    momentum_threshold = p.get("momentum_buy_threshold", 3)

    indicators = calc_indicators(history_df, params=params)
    bb_pos = indicators.get("bb_position", 0.5)

    # 布林带区域
    if bb_pos < 0.0:
        bb_zone = "OVERSOLD"
    elif bb_pos > 1.0:
        bb_zone = "OVERBOUGHT"
    else:
        bb_zone = "NORMAL"

    # 动量评分（6维）
    mom_score = 0
    if indicators.get("mom_5d", 0) > 0:
        mom_score += 1
    if indicators.get("mom_10d", 0) > 0:
        mom_score += 1
    if indicators.get("mom_20d", 0) > 0:
        mom_score += 1
    if indicators.get("above_ma20", False):
        mom_score += 1
    if indicators.get("macd_turn_positive", False):
        mom_score += 1
    touched = indicators.get("bb_touched_lower_3d", False)
    if touched and bb_pos > 0.3:
        mom_score += 1

    # 卖出信号判定
    sell_signal = False
    rsi = indicators.get("rsi", 50)
    # 路径1：超买区 + RSI确认
    if bb_zone == "OVERBOUGHT" and rsi > rsi_overbought:
        sell_signal = True
    # 路径2：弱势确认（5日动量负 + 跌破MA20 + MACD持续负）
    if (indicators.get("mom_5d", 0) < 0 and
            not indicators.get("above_ma20", True) and
            indicators.get("macd_5d_negative", False)):
        sell_signal = True

    reasons = []

    # ===== 信号判定 =====
    if bb_zone == "OVERSOLD":
        action = "BUY"
        reasons.append(f"超卖区（%%B={bb_pos:.2f}，价格低于布林带下轨）→ 加速定投")
        if indicators.get("bb_squeeze", False):
            reasons.append("布林带挤压中，可能即将突破")
    elif sell_signal:
        action = "SELL"
        if bb_zone == "OVERBOUGHT" and rsi > rsi_overbought:
            reasons.append(f"超买区（%%B={bb_pos:.2f}）+ RSI={rsi:.1f} 超买确认")
        if (indicators.get("mom_5d", 0) < 0 and
                not indicators.get("above_ma20", True)):
            reasons.append("5日动量转负 + NAV跌破MA20")
        if indicators.get("macd_5d_negative", False):
            reasons.append("MACD柱连续5天为负")
    elif bb_zone == "OVERBOUGHT":
        action = "HOLD"
        reasons.append(f"超买区（%%B={bb_pos:.2f}），暂停买入")
    elif mom_score >= momentum_threshold:
        action = "BUY"
        details = []
        if indicators.get("mom_5d", 0) > 0:
            details.append(f"5日动量+{indicators['mom_5d']:.2%}")
        if indicators.get("mom_20d", 0) > 0:
            details.append(f"20日动量+{indicators['mom_20d']:.2%}")
        if indicators.get("above_ma20", False):
            details.append("站上MA20")
        if indicators.get("macd_turn_positive", False):
            details.append("MACD转正")
        if touched and bb_pos > 0.3:
            details.append("下轨反弹")
        reasons.append(f"动量评分 {mom_score}/6 ≥ {momentum_threshold}")
        reasons.append("；".join(details))
    else:
        action = "HOLD"
        reasons.append(f"动量评分 {mom_score}/6，未达阈值{momentum_threshold}，观望")

    # 补充指标状态
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    if bb_upper and bb_lower:
        reasons.append(f"布林带: [{bb_lower:.4f}, {bb_upper:.4f}]")
    reasons.append(f"BB区域: {bb_zone}（%%B={bb_pos:.2f}）")
    trend = "上升" if indicators.get("above_ma20", False) else "下降"
    reasons.append(f"趋势: NAV {'>' if indicators.get('above_ma20', False) else '<'} MA20 ({trend})")
    atr = indicators.get("atr", 0)
    atr_median = indicators.get("atr_median", 0)
    if atr > 0 and atr_median > 0:
        scale = min(atr_median / atr, 2.0)
        reasons.append(f"ATR缩放: {scale:.2f}x")

    return {
        "action": action,
        "reason": "；".join(reasons),
        "indicators": indicators,
        "buy_score": mom_score,
        "sell_signal": sell_signal,
        "strategy": "index_momentum",
        "bb_zone": bb_zone,
    }
