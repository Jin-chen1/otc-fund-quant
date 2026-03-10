import os
import sys
from math import sin

import pandas as pd


PROJECT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from otc_fund_quant.analysis.backtest import calc_period_returns
from otc_fund_quant.analysis.signals import (
    generate_recommendation_bond_plus_balance,
    generate_recommendation_bond_stability,
    generate_recommendation_qdii_trend,
)


def _history_from_navs(navs, start="2024-01-01"):
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(navs)).date,
            "nav": [float(v) for v in navs],
        }
    )


def _gentle_uptrend(length=180, base=1.0, step=0.0004):
    return [base + idx * step + 0.0003 * sin(idx / 12) for idx in range(length)]


def _sideways(length=180, base=1.0):
    return [base + 0.004 * sin(idx / 3) + 0.002 * sin(idx / 11) for idx in range(length)]


def _uptrend_then_break(length=180, base=1.0):
    values = []
    for idx in range(length):
        if idx < length - 25:
            values.append(base + idx * 0.0005 + 0.0004 * sin(idx / 10))
        else:
            last_val = values[-1]
            values.append(last_val * 0.992)
    return values


def test_bond_stability_recommendation_handles_insufficient_history():
    history_df = _history_from_navs(_gentle_uptrend(length=80))
    rec = generate_recommendation_bond_stability(history_df)
    assert rec["action"] == "HOLD"
    assert "历史数据不足" in rec["reason"]


def test_bond_stability_recommendation_triggers_buy_on_stable_trend():
    history_df = _history_from_navs(_gentle_uptrend())
    rec = generate_recommendation_bond_stability(history_df)
    assert rec["action"] == "BUY"
    assert rec["buy_score"] >= 1


def test_bond_stability_recommendation_triggers_sell_on_breakdown():
    history_df = _history_from_navs(_uptrend_then_break())
    rec = generate_recommendation_bond_stability(history_df)
    assert rec["action"] == "SELL"
    assert rec["sell_signal"] is True


def test_qdii_trend_recommendation_returns_hold_for_sideways_data():
    history_df = _history_from_navs(_sideways())
    rec = generate_recommendation_qdii_trend(history_df)
    assert rec["action"] == "HOLD"


def test_qdii_trend_recommendation_triggers_buy_on_trend():
    history_df = _history_from_navs(_gentle_uptrend(step=0.001))
    rec = generate_recommendation_qdii_trend(history_df)
    assert rec["action"] == "BUY"
    assert rec["buy_score"] >= 2


def test_qdii_trend_recommendation_triggers_sell_on_weak_exit():
    history_df = _history_from_navs(_uptrend_then_break())
    rec = generate_recommendation_qdii_trend(history_df)
    assert rec["action"] == "SELL"


def test_bond_plus_balance_recommendation_triggers_buy_on_confirmed_trend():
    history_df = _history_from_navs(_gentle_uptrend(step=0.0007))
    rec = generate_recommendation_bond_plus_balance(history_df)
    assert rec["action"] == "BUY"
    assert rec["buy_score"] >= 1


def test_bond_plus_balance_recommendation_triggers_sell_on_drawdown():
    history_df = _history_from_navs(_uptrend_then_break())
    rec = generate_recommendation_bond_plus_balance(history_df)
    assert rec["action"] == "SELL"
    assert rec["sell_signal"] is True


def test_bond_plus_balance_recommendation_handles_hold_state():
    history_df = _history_from_navs(_sideways())
    rec = generate_recommendation_bond_plus_balance(history_df)
    assert rec["action"] == "HOLD"


def test_new_strategies_backtest_smoke():
    history_df = _history_from_navs(_gentle_uptrend(length=240, step=0.0007))
    for strategy in ("bond_stability", "qdii_trend", "bond_plus_balance"):
        results = calc_period_returns(history_df, strategy=strategy, params={})
        assert isinstance(results, list)
        assert results
        assert {"label", "days", "strategy_pct", "fund_pct", "start_date", "end_date"} <= set(results[0].keys())
