"""
图表数据模块 — 为前端图表提供格式化的时间序列数据。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def get_chart_data(history_df: pd.DataFrame, days: int = 365) -> Dict[str, Any]:
    """获取图表所需的数据（NAV、均线、RSI、MACD）。

    Args:
        history_df: 完整历史数据
        days: 显示最近多少天

    Returns:
        字典，包含各序列数据（list格式，便于JSON序列化）
    """
    from datetime import date as _date
    today = _date.today()
    history_df = history_df[history_df["date"] <= today]
    df = history_df.tail(max(days + 100, 360)).copy()
    nav = df["nav"]

    df["ma20"] = nav.rolling(window=20).mean()
    df["ma60"] = nav.rolling(window=60).mean()

    # RSI
    delta = nav.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = nav.ewm(span=12, adjust=False).mean()
    ema26 = nav.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - signal_line
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line

    # 百分位
    df["percentile"] = nav.rolling(window=250).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) >= 250 else np.nan,
        raw=False
    )

    # 取最后 days 天
    df = df.tail(days).copy()
    df = df.dropna(subset=["ma20"])

    # 转为 date 字符串
    if hasattr(df["date"].iloc[0], "strftime"):
        dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
    else:
        dates = df["date"].astype(str).tolist()

    return {
        "dates": dates,
        "nav": df["nav"].round(4).tolist(),
        "ma20": [round(v, 4) if pd.notna(v) else None for v in df["ma20"]],
        "ma60": [round(v, 4) if pd.notna(v) else None for v in df["ma60"]],
        "rsi": [round(v, 2) if pd.notna(v) else None for v in df["rsi"]],
        "macd_hist": [round(v, 6) if pd.notna(v) else None for v in df["macd_hist"]],
        "macd_line": [round(v, 6) if pd.notna(v) else None for v in df["macd_line"]],
        "macd_signal": [round(v, 6) if pd.notna(v) else None for v in df["macd_signal"]],
        "percentile": [round(v, 4) if pd.notna(v) else None for v in df["percentile"]],
    }
