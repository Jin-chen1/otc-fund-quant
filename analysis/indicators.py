"""
通用技术指标计算模块 — 项目唯一的指标计算入口。
被 analysis.signals、strategies、web 等模块共同调用。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


# ===== 策略常量 =====
CHEAP_PERCENTILE = 0.20
RSI_SELL_THRESHOLD = 70
ADX_TREND_THRESHOLD = 25


def calc_indicators(history_df: pd.DataFrame, tail_size: int = 600,
                    percentile_window: int = 500,
                    params: Dict[str, Any] = None) -> Dict[str, Any]:
    """计算全部技术指标，返回字典。

    Args:
        history_df: 包含 'date' 和 'nav' 列的 DataFrame（至少250行）。
        tail_size: 取最近多少行数据计算指标（默认600，保证500日百分位窗口）。
        percentile_window: 百分位计算窗口大小（默认500日）。
        params: 可选配置参数字典，含 cheap_percentile / adx_trend_threshold 等。

    Returns:
        包含各技术指标值的字典。
    """
    p = params or {}
    _cheap_percentile = p.get("cheap_percentile", CHEAP_PERCENTILE)
    _adx_trend_threshold = p.get("adx_trend_threshold", ADX_TREND_THRESHOLD)
    # 允许通过 params 覆盖窗口大小（短历史基金需要更小的窗口）
    tail_size = p.get("tail_size", tail_size)
    percentile_window = p.get("percentile_window", percentile_window)
    df = history_df.tail(tail_size).copy()
    nav = df["nav"]

    # 均线
    df["ma20"] = nav.rolling(window=20).mean()
    df["ma60"] = nav.rolling(window=60).mean()

    # RSI(14)
    delta = nav.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD 柱状图
    ema12 = nav.ewm(span=12, adjust=False).mean()
    ema26 = nav.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - signal_line
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line

    # ATR(14) — 基金无最高/最低价，用日收益波动近似
    df["tr"] = nav.diff().abs()
    df["atr"] = df["tr"].rolling(window=14).mean()

    # ADX(14) — 趋势强度
    df["up_move"] = nav.diff()
    df["down_move"] = -nav.diff()
    df["plus_dm"] = df["up_move"].apply(lambda x: max(x, 0))
    df["minus_dm"] = df["down_move"].apply(lambda x: max(x, 0))
    for i in range(1, len(df)):
        if df.iloc[i]["plus_dm"] <= df.iloc[i]["minus_dm"]:
            df.iloc[i, df.columns.get_loc("plus_dm")] = 0
        else:
            df.iloc[i, df.columns.get_loc("minus_dm")] = 0
    atr_s = df["atr"]
    plus_di = 100 * (df["plus_dm"].rolling(14).mean() / atr_s.replace(0, np.nan))
    minus_di = 100 * (df["minus_dm"].rolling(14).mean() / atr_s.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.rolling(14).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 百分位
    pct_window = df["nav"].tail(percentile_window)
    current_nav = float(last["nav"])
    percentile = float((pct_window < current_nav).mean())

    # 金叉/死叉
    gold_cross = bool((prev["ma20"] < prev["ma60"]) and (last["ma20"] > last["ma60"]))
    death_cross = bool((prev["ma20"] > prev["ma60"]) and (last["ma20"] < last["ma60"]))

    # MACD
    macd_turn_positive = bool((prev["macd_hist"] < 0) and (last["macd_hist"] >= 0))
    macd_5d_negative = bool((df["macd_hist"].tail(5) < 0).all())

    # MA20 位置
    above_ma20 = bool(current_nav > last["ma20"]) if pd.notna(last["ma20"]) else False

    last_3 = df.tail(3)
    above_ma20_3d = False
    if len(last_3) >= 3 and last_3["ma20"].notna().all():
        above_ma20_3d = bool((last_3["nav"] > last_3["ma20"]).all())

    rsi_val = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
    atr_val = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
    adx_val = float(last["adx"]) if pd.notna(last["adx"]) else 0.0
    ma60_val = float(last["ma60"]) if pd.notna(last["ma60"]) else None

    # 市场状态识别
    if adx_val > _adx_trend_threshold and current_nav > (ma60_val or current_nav):
        market_regime = "BULL"
    elif adx_val > _adx_trend_threshold:
        market_regime = "BEAR"
    else:
        market_regime = "RANGE"

    # 布林带（20日均线 ± 2倍标准差）
    bb_std = nav.rolling(window=20).std()
    ma20_val_float = float(last["ma20"]) if pd.notna(last["ma20"]) else None
    bb_std_val = float(bb_std.iloc[-1]) if pd.notna(bb_std.iloc[-1]) else 0.0
    if ma20_val_float and bb_std_val > 0:
        bb_upper = ma20_val_float + 2.0 * bb_std_val
        bb_lower = ma20_val_float - 2.0 * bb_std_val
        bb_position = (current_nav - bb_lower) / (bb_upper - bb_lower)  # %B
        bb_width = (bb_upper - bb_lower) / ma20_val_float  # 归一化带宽
    else:
        bb_upper = None
        bb_lower = None
        bb_position = 0.5
        bb_width = 0.0

    # 布林带挤压检测（带宽 < 近20日平均带宽的60%）
    bb_width_series = (nav.rolling(20).std() * 2) / df["ma20"]
    bb_width_series = bb_width_series.replace([np.inf, -np.inf], np.nan)
    bb_width_avg = float(bb_width_series.tail(20).mean()) if bb_width_series.tail(20).notna().any() else 0.0
    bb_squeeze = bool(bb_width < bb_width_avg * 0.6) if bb_width_avg > 0 else False

    # 近3日是否触及布林带下轨
    bb_touched_lower_3d = False
    if bb_lower is not None:
        last_3_nav = df["nav"].tail(3)
        bb_lower_series = df["ma20"].tail(3) - 2.0 * bb_std.tail(3)
        if bb_lower_series.notna().all():
            bb_touched_lower_3d = bool((last_3_nav <= bb_lower_series).any())

    # 多周期动量
    mom_5d = float(current_nav / float(nav.iloc[-6]) - 1) if len(nav) >= 6 else 0.0
    mom_10d = float(current_nav / float(nav.iloc[-11]) - 1) if len(nav) >= 11 else 0.0
    mom_20d = float(current_nav / float(nav.iloc[-21]) - 1) if len(nav) >= 21 else 0.0

    # ATR中位数（近60日ATR的中位数，用于自适应仓位计算）
    atr_median = float(df["atr"].tail(60).median()) if df["atr"].tail(60).notna().any() else atr_val

    return {
        "current_nav": current_nav,
        "percentile": percentile,
        "is_cheap_zone": percentile < _cheap_percentile,
        "gold_cross": gold_cross,
        "death_cross": death_cross,
        "rsi": rsi_val,
        "macd_turn_positive": macd_turn_positive,
        "macd_5d_negative": macd_5d_negative,
        "above_ma20": above_ma20,
        "above_ma20_3d": above_ma20_3d,
        "ma20": ma20_val_float,
        "ma60": ma60_val,
        "macd_hist": float(last["macd_hist"]) if pd.notna(last["macd_hist"]) else 0.0,
        "atr": atr_val,
        "adx": adx_val,
        "market_regime": market_regime,
        # 布林带指标
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_position": bb_position,
        "bb_width": bb_width,
        "bb_squeeze": bb_squeeze,
        "bb_touched_lower_3d": bb_touched_lower_3d,
        # 多周期动量
        "mom_5d": mom_5d,
        "mom_10d": mom_10d,
        "mom_20d": mom_20d,
        # ATR中位数
        "atr_median": atr_median,
    }


def count_buy_signals(indicators: Dict, params: Dict[str, Any] = None) -> int:
    """三重过滤买入信号计分：金叉/RSI/MACD转正。"""
    p = params or {}
    rsi_low = p.get("rsi_buy_low", 35)
    rsi_high = p.get("rsi_buy_high", 65)

    score = 0
    if indicators["gold_cross"]:
        score += 1
    if rsi_low <= indicators["rsi"] <= rsi_high:
        score += 1
    if indicators["macd_turn_positive"]:
        score += 1
    return score


def check_sell_signal(indicators: Dict, params: Dict[str, Any] = None) -> bool:
    """卖出信号（多路径触发）：
    1. 死叉 + (RSI>70 或 MACD连续5天负) — 原版组合
    2. RSI > 80 — 极度超买，独立触发
    3. MACD连续5天负 + NAV跌破MA20 — 持续走弱，无需等死叉
    """
    p = params or {}
    _rsi_sell_threshold = p.get("rsi_sell_threshold", RSI_SELL_THRESHOLD)

    # 路径1：原版组合（死叉 + 辅助确认）
    if indicators["death_cross"]:
        if indicators["rsi"] > _rsi_sell_threshold:
            return True
        if indicators["macd_5d_negative"]:
            return True

    # 路径2：极度超买独立触发
    _rsi_extreme = p.get("rsi_extreme_overbought", 80)
    if indicators["rsi"] > _rsi_extreme:
        return True

    # 路径3：持续弱势（MACD负 + 价格破MA20）
    if indicators["macd_5d_negative"] and not indicators["above_ma20"]:
        return True

    return False


def detect_regime(indicators: Dict, params: Dict[str, Any] = None) -> str:
    """识别当前市场状态。

    ADX > 25 且 NAV > MA60 → BULL
    ADX > 25 且 NAV < MA60 → BEAR
    ADX ≤ 25             → RANGE
    """
    p = params or {}
    _adx_trend_threshold = p.get("adx_trend_threshold", ADX_TREND_THRESHOLD)

    adx = indicators.get("adx", 0)
    nav = indicators["current_nav"]
    ma60 = indicators.get("ma60", nav)

    if adx > _adx_trend_threshold:
        if nav > (ma60 or nav):
            return "BULL"
        else:
            return "BEAR"
    return "RANGE"
