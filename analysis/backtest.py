"""
区间回测模块 — 使用真实回测引擎计算各周期策略收益。
调用 core.Account + strategies 中的策略类，完整模拟 T+1 结算。
"""

import pandas as pd
from datetime import date as _date, timedelta

from ..core.account import Account
from ..strategies.trend_strategy import ValuationTrendHybridStrategy
from ..strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from ..strategies.index_momentum_strategy import IndexMomentumStrategy


def calc_period_returns(history_df: pd.DataFrame, strategy: str = "v6",
                        params: dict = None) -> list:
    """使用真实回测引擎（Account+Strategy）计算各周期的策略收益。

    完整复用回测逻辑：T+1结算、分批建仓、止损/冷静期、仓位上限、赎回费(FIFO)。
    同时返回基金本身涨跌幅用于对比。

    Args:
        history_df: 完整历史数据（date, nav 列）
        strategy: 策略名 ("v6" 或 "regime_adaptive")
        params: 可选策略参数字典

    Returns:
        list of dict, 每个含 label, days, strategy_pct, fund_pct
    """
    today = _date.today()

    # 统一日期格式为 datetime.date
    df = history_df[["date", "nav"]].copy()
    if hasattr(df["date"].iloc[0], "date") and callable(getattr(df["date"].iloc[0], "date", None)):
        df["date"] = df["date"].apply(lambda x: x.date())
    elif not isinstance(df["date"].iloc[0], _date):
        df["date"] = pd.to_datetime(df["date"]).dt.date
    df["nav"] = df["nav"].astype(float)
    df = df[df["date"] <= today].sort_values("date").reset_index(drop=True)

    min_data = 70 if strategy == "index_momentum" else 260
    if len(df) < min_data:
        return []

    end_nav = float(df.iloc[-1]["nav"])
    end_date = df.iloc[-1]["date"]

    INITIAL_CASH = 10000.0
    FUND_CODE = "__bt__"

    periods = [
        ("近5年", 365 * 5),
        ("近3年", 365 * 3),
        ("近1年", 365),
        ("近半年", 183),
        ("近1月", 30),
    ]

    results = []
    for label, target_days in periods:
        cutoff = end_date - timedelta(days=target_days)

        # 找到回测起始行（需要250行历史做指标预热）
        start_idx = None
        min_hist = 60 if strategy == "index_momentum" else 250
        for i in range(len(df)):
            if df.iloc[i]["date"] >= cutoff and i >= min_hist:
                start_idx = i
                break
        if start_idx is None:
            continue

        start_nav = float(df.iloc[start_idx]["nav"])
        start_date_val = df.iloc[start_idx]["date"]
        actual_days = (end_date - start_date_val).days

        # 跳过实际天数不足目标天数50%的周期（避免短历史基金出现重复结果）
        if actual_days < target_days * 0.5:
            continue

        fund_pct = (end_nav / start_nav - 1) * 100 if start_nav > 0 else 0.0

        # 创建全新的账户和策略实例
        account = Account(initial_cash=INITIAL_CASH)
        if strategy == "index_momentum":
            strat = IndexMomentumStrategy(account, FUND_CODE, params=params)
        elif strategy == "regime_adaptive":
            strat = RegimeAdaptiveStrategy(account, FUND_CODE, params=params)
        else:
            strat = ValuationTrendHybridStrategy(account, FUND_CODE, params=params)

        # 模拟引擎循环（T+1结算）
        pending_buy = 0.0
        pending_sell = 0.0

        for i in range(start_idx, len(df)):
            current_date = df.iloc[i]["date"]
            current_nav = float(df.iloc[i]["nav"])

            # 步骤1：结算昨日挂单（以今日NAV成交）
            if pending_buy > 0:
                try:
                    account.buy(FUND_CODE, pending_buy, current_nav, current_date)
                except ValueError:
                    pass  # 资金不足，跳过
                pending_buy = 0.0

            if pending_sell > 0:
                pos = account.get_position(FUND_CODE)
                actual_sell = min(pending_sell, pos.total_shares)
                if actual_sell > 1e-9:
                    try:
                        account.sell(FUND_CODE, actual_sell, current_nav, current_date)
                    except ValueError:
                        pass
                pending_sell = 0.0

            # 步骤2：获取 T-1 历史数据供策略使用
            hist_slice = df.iloc[:i][["date", "nav"]].copy()
            if len(hist_slice) < min_hist:
                continue

            # 步骤3：策略决策
            signal = strat.on_bar(current_date, hist_slice)

            # 步骤4：挂单（下一个交易日结算）
            buy_amount = signal.get("buy_amount", 0.0)
            sell_shares = signal.get("sell_shares", 0.0)

            if buy_amount > 0 and account.cash >= buy_amount:
                pending_buy = buy_amount
            elif sell_shares > 0:
                pos = account.get_position(FUND_CODE)
                if pos.total_shares >= sell_shares:
                    pending_sell = sell_shares

        # 期末估值
        pos = account.get_position(FUND_CODE)
        final_value = account.cash + pos.total_shares * end_nav
        strategy_pct = (final_value / INITIAL_CASH - 1) * 100

        results.append({
            "label": label,
            "days": actual_days,
            "strategy_pct": round(strategy_pct, 2),
            "fund_pct": round(fund_pct, 2),
            "start_date": start_date_val.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        })

    return results
