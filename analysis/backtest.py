"""
区间回测模块 — 使用真实回测引擎计算各周期策略收益。
调用 core.Account + strategies 中的策略类，完整模拟 T+1 结算。
"""

from datetime import date as _date, timedelta
from typing import Any

import pandas as pd

from ..core.account import Account
from ..strategies.bond_plus_balance_strategy import BondPlusBalanceStrategy
from ..strategies.bond_stability_strategy import BondStabilityStrategy
from ..strategies.qdii_trend_strategy import QDIITrendStrategy
from ..strategies.trend_strategy import ValuationTrendHybridStrategy
from ..strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from ..strategies.index_momentum_strategy import IndexMomentumStrategy
from ..strategies.active_equity_cn_strategy import ActiveEquityCNStrategy
from ..strategies.active_equity_hk_strategy import ActiveEquityHKStrategy


BACKTEST_INITIAL_CASH = 10000.0
BACKTEST_FUND_CODE = "__bt__"


def _get_strategy_min_history(strategy: str) -> int:
    if strategy == "index_momentum":
        return 60
    if strategy in {"bond_stability", "qdii_trend", "bond_plus_balance"}:
        return 140
    return 250


def _build_strategy(strategy: str, account: Account, fund_code: str, params: dict | None):
    if strategy == "index_momentum":
        return IndexMomentumStrategy(account, fund_code, params=params)
    if strategy == "regime_adaptive":
        return RegimeAdaptiveStrategy(account, fund_code, params=params)
    if strategy == "active_equity_cn":
        return ActiveEquityCNStrategy(account, fund_code, params=params)
    if strategy == "active_equity_hk":
        return ActiveEquityHKStrategy(account, fund_code, params=params)
    if strategy == "bond_stability":
        return BondStabilityStrategy(account, fund_code, params=params)
    if strategy == "qdii_trend":
        return QDIITrendStrategy(account, fund_code, params=params)
    if strategy == "bond_plus_balance":
        return BondPlusBalanceStrategy(account, fund_code, params=params)
    return ValuationTrendHybridStrategy(account, fund_code, params=params)


def _normalize_history_df(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=["date", "nav"])

    df = history_df[["date", "nav"]].copy()
    if hasattr(df["date"].iloc[0], "date") and callable(getattr(df["date"].iloc[0], "date", None)):
        df["date"] = df["date"].apply(lambda x: x.date())
    elif not isinstance(df["date"].iloc[0], _date):
        df["date"] = pd.to_datetime(df["date"]).dt.date
    df["nav"] = df["nav"].astype(float)
    today = _date.today()
    return df[df["date"] <= today].sort_values("date").reset_index(drop=True)


def _resolve_period_window(
    df: pd.DataFrame,
    strategy: str,
    target_days: int,
) -> tuple[int, _date, _date, int] | None:
    min_hist = _get_strategy_min_history(strategy)
    min_data = min_hist + 10
    if len(df) < min_data:
        return None

    end_date = df.iloc[-1]["date"]
    cutoff = end_date - timedelta(days=target_days)
    start_idx = None
    for i in range(len(df)):
        if df.iloc[i]["date"] >= cutoff and i >= min_hist:
            start_idx = i
            break
    if start_idx is None:
        return None

    start_date = df.iloc[start_idx]["date"]
    actual_days = (end_date - start_date).days
    if actual_days < target_days * 0.5:
        return None

    return start_idx, start_date, end_date, actual_days


def _simulate_backtest_window(
    df: pd.DataFrame,
    *,
    strategy: str,
    params: dict | None,
    start_idx: int,
    initial_cash: float = BACKTEST_INITIAL_CASH,
    fund_code: str = BACKTEST_FUND_CODE,
    include_trades: bool = False,
) -> dict[str, Any]:
    min_hist = _get_strategy_min_history(strategy)
    account = Account(initial_cash=initial_cash)
    strat = _build_strategy(strategy, account, fund_code, params)
    pending_buy: dict[str, Any] | None = None
    pending_sell: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []

    for i in range(start_idx, len(df)):
        current_date = df.iloc[i]["date"]
        current_nav = float(df.iloc[i]["nav"])

        if pending_buy is not None:
            try:
                shares = account.buy(fund_code, pending_buy["amount"], current_nav, current_date)
            except ValueError:
                shares = None
            if include_trades and shares is not None:
                trades.append(
                    {
                        "date": current_date.strftime("%Y-%m-%d"),
                        "action": "BUY",
                        "nav": round(current_nav, 4),
                        "amount": round(float(pending_buy["amount"]), 2),
                        "shares": round(float(shares), 4),
                        "decision_date": pending_buy["decision_date"].strftime("%Y-%m-%d"),
                    }
                )
            pending_buy = None

        if pending_sell is not None:
            position = account.get_position(fund_code)
            actual_sell = min(float(pending_sell["shares"]), position.total_shares)
            if actual_sell > 1e-9:
                try:
                    result = account.sell(fund_code, actual_sell, current_nav, current_date)
                except ValueError:
                    result = None
                if include_trades and result is not None:
                    trades.append(
                        {
                            "date": current_date.strftime("%Y-%m-%d"),
                            "action": "SELL",
                            "nav": round(current_nav, 4),
                            "amount": round(float(result["cash_obtained"]), 2),
                            "shares": round(float(actual_sell), 4),
                            "decision_date": pending_sell["decision_date"].strftime("%Y-%m-%d"),
                        }
                    )
            pending_sell = None

        hist_slice = df.iloc[:i][["date", "nav"]].copy()
        if len(hist_slice) < min_hist:
            continue

        signal = strat.on_bar(current_date, hist_slice)
        buy_amount = float(signal.get("buy_amount", 0.0) or 0.0)
        sell_shares = float(signal.get("sell_shares", 0.0) or 0.0)

        if buy_amount > 0 and account.cash >= buy_amount:
            pending_buy = {"amount": buy_amount, "decision_date": current_date}
        elif sell_shares > 0:
            position = account.get_position(fund_code)
            if position.total_shares >= sell_shares:
                pending_sell = {"shares": sell_shares, "decision_date": current_date}

    end_nav = float(df.iloc[-1]["nav"])
    position = account.get_position(fund_code)
    final_value = account.cash + position.total_shares * end_nav
    return {
        "final_value": final_value,
        "end_nav": end_nav,
        "trades": trades,
    }


def get_backtest_trades(
    history_df: pd.DataFrame,
    strategy: str = "v6",
    params: dict | None = None,
    days: int = 365,
) -> dict[str, Any]:
    """提取最近窗口内的真实回测成交点和成交明细。"""
    df = _normalize_history_df(history_df)
    window = _resolve_period_window(df, strategy, days)
    empty_points = {
        "buy_dates": [],
        "buy_navs": [],
        "sell_dates": [],
        "sell_navs": [],
    }
    if window is None:
        return {"trade_points": empty_points, "trade_history": []}

    start_idx, _, _, _ = window
    simulation = _simulate_backtest_window(
        df,
        strategy=strategy,
        params=params,
        start_idx=start_idx,
        include_trades=True,
    )
    trades = simulation["trades"]

    trade_points = {
        "buy_dates": [trade["date"] for trade in trades if trade["action"] == "BUY"],
        "buy_navs": [trade["nav"] for trade in trades if trade["action"] == "BUY"],
        "sell_dates": [trade["date"] for trade in trades if trade["action"] == "SELL"],
        "sell_navs": [trade["nav"] for trade in trades if trade["action"] == "SELL"],
    }
    return {
        "trade_points": trade_points,
        "trade_history": list(reversed(trades)),
    }


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
    df = _normalize_history_df(history_df)
    if df.empty:
        return []

    end_nav = float(df.iloc[-1]["nav"])
    end_date = df.iloc[-1]["date"]

    periods = [
        ("近5年", 365 * 5),
        ("近3年", 365 * 3),
        ("近1年", 365),
        ("近半年", 183),
        ("近1月", 30),
    ]

    results = []
    for label, target_days in periods:
        window = _resolve_period_window(df, strategy, target_days)
        if window is None:
            continue

        start_idx, start_date_val, _, actual_days = window
        start_nav = float(df.iloc[start_idx]["nav"])

        fund_pct = (end_nav / start_nav - 1) * 100 if start_nav > 0 else 0.0
        simulation = _simulate_backtest_window(
            df,
            strategy=strategy,
            params=params,
            start_idx=start_idx,
            include_trades=False,
        )
        strategy_pct = (simulation["final_value"] / BACKTEST_INITIAL_CASH - 1) * 100

        results.append({
            "label": label,
            "days": actual_days,
            "strategy_pct": round(strategy_pct, 2),
            "fund_pct": round(fund_pct, 2),
            "start_date": start_date_val.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        })

    return results
