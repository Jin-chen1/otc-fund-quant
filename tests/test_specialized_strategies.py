import os
import sys
from math import sin

import pandas as pd


PROJECT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from otc_fund_quant.analysis import backtest as backtest_module
from otc_fund_quant.analysis.backtest import calc_period_returns
from otc_fund_quant.analysis.signals import (
    _get_rec_fn,
    generate_recommendation_active_equity_cn,
    generate_recommendation_active_equity_hk,
    generate_recommendation_bond_plus_balance,
    generate_recommendation_bond_stability,
    generate_recommendation_qdii_trend,
)
from otc_fund_quant.analysis.strategy_registry import get_strategy_definition, is_known_strategy
from otc_fund_quant.core.account import Account
from otc_fund_quant.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy


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


def _plateau(length=320, base=1.0):
    return [base + min(idx, 150) * 0.0007 + 0.0002 * sin(idx / 8) for idx in range(length)]


def _cn_breakdown_not_cheap(length=320, base=1.0):
    values = []
    for idx in range(length):
        if idx < length - 30:
            values.append(base + idx * 0.0006 + 0.0004 * sin(idx / 10))
        else:
            values.append(values[-1] * 0.999)
    return values


def _hk_follow_through_buy(length=320, base=1.0):
    return [
        base
        - 0.000039542043588170395 * idx
        + 0.003147543357536982 * sin(idx / 12.63815363135844)
        + 0.00580524150174086 * sin(idx / 23.439087143067113)
        - max(idx - 280, 0) * 0.00014395631626196338
        for idx in range(length)
    ]


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
    history_df = _history_from_navs(_gentle_uptrend(length=420, step=0.0007))
    for strategy in ("bond_stability", "qdii_trend", "bond_plus_balance", "active_equity_cn", "active_equity_hk"):
        results = calc_period_returns(history_df, strategy=strategy, params={})
        assert isinstance(results, list)
        assert results
        assert {"label", "days", "strategy_pct", "fund_pct", "start_date", "end_date"} <= set(results[0].keys())


def test_strategy_registry_knows_new_active_equity_strategies():
    assert is_known_strategy("active_equity_cn") is True
    assert is_known_strategy("active_equity_hk") is True
    assert get_strategy_definition("active_equity_cn").label == "A股主动权益"
    assert get_strategy_definition("active_equity_hk").label == "港股主动权益"


def test_signal_dispatch_routes_new_active_equity_strategies():
    assert _get_rec_fn("active_equity_cn") is generate_recommendation_active_equity_cn
    assert _get_rec_fn("active_equity_hk") is generate_recommendation_active_equity_hk


def test_active_equity_cn_recommendation_triggers_buy_on_trend_follow_through():
    history_df = _history_from_navs(_gentle_uptrend(length=320, step=0.0008))
    rec = generate_recommendation_active_equity_cn(history_df)
    assert rec["action"] == "BUY"
    assert rec["buy_score"] >= 1


def test_active_equity_cn_recommendation_holds_on_plateau():
    history_df = _history_from_navs(_plateau())
    rec = generate_recommendation_active_equity_cn(history_df)
    assert rec["action"] == "HOLD"


def test_active_equity_cn_recommendation_sells_on_breakdown_without_cheap_override():
    history_df = _history_from_navs(_cn_breakdown_not_cheap())
    rec = generate_recommendation_active_equity_cn(history_df)
    assert rec["action"] == "SELL"
    assert rec["sell_signal"] is True


def test_active_equity_hk_recommendation_can_buy_on_strong_follow_through():
    history_df = _history_from_navs(_hk_follow_through_buy())
    rec = generate_recommendation_active_equity_hk(history_df)
    assert rec["action"] == "BUY"
    assert rec["buy_score"] >= 2


def test_active_equity_hk_recommendation_holds_on_mild_uptrend():
    history_df = _history_from_navs(_gentle_uptrend(length=320, step=0.0008))
    rec = generate_recommendation_active_equity_hk(history_df)
    assert rec["action"] == "HOLD"


def test_active_equity_hk_recommendation_sells_on_breakdown():
    history_df = _history_from_navs(_uptrend_then_break(length=320))
    rec = generate_recommendation_active_equity_hk(history_df)
    assert rec["action"] == "SELL"
    assert rec["sell_signal"] is True


def test_get_backtest_trades_uses_fill_date_and_preserves_trade_details(monkeypatch):
    history_df = _history_from_navs(
        [1.00, 1.01, 1.02, 1.03, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22],
        start="2026-01-01",
    )

    class FakeStrategy:
        def __init__(self, account):
            self.account = account
            self.call_count = 0

        def on_bar(self, current_date, history_df):
            self.call_count += 1
            if self.call_count == 1:
                return {"buy_amount": 100.0}
            if self.call_count == 3:
                position = self.account.get_position("__bt__")
                return {"sell_shares": position.total_shares / 2}
            return {}

    monkeypatch.setattr(backtest_module, "_get_strategy_min_history", lambda strategy: 3)
    monkeypatch.setattr(
        backtest_module,
        "_build_strategy",
        lambda strategy, account, fund_code, params: FakeStrategy(account),
    )

    payload = backtest_module.get_backtest_trades(history_df, strategy="fake", params={}, days=5)

    trade_points = payload["trade_points"]
    trade_history = payload["trade_history"]

    assert trade_points["buy_dates"] == ["2026-01-10"]
    assert trade_points["sell_dates"] == ["2026-01-12"]
    assert trade_history[0]["action"] == "SELL"
    assert trade_history[0]["date"] == "2026-01-12"
    assert trade_history[0]["decision_date"] == "2026-01-11"
    assert trade_history[0]["shares"] == 43.8596
    assert trade_history[0]["amount"] == 50.98
    assert trade_history[1]["action"] == "BUY"
    assert trade_history[1]["date"] == "2026-01-10"
    assert trade_history[1]["decision_date"] == "2026-01-09"
    assert trade_history[1]["shares"] == 87.7193
    assert trade_history[1]["amount"] == 100.0


def _make_regime_strategy(params=None):
    account = Account(initial_cash=1000.0)
    strategy = RegimeAdaptiveStrategy(account, "015916", params=params or {})
    account.buy("015916", 400.0, 1.0, pd.Timestamp("2026-01-01").date())
    position = account.get_position("015916")
    return strategy, position


def test_regime_transition_sell_defaults_to_original_behavior():
    strategy, position = _make_regime_strategy({"transition_sell_ratio": 0.25})
    strategy.prev_regime = "BULL"
    strategy.current_regime = "BEAR"

    signal = strategy._check_regime_transition(
        position,
        nav=1.05,
        indicators={"above_ma20": True, "macd_5d_negative": False},
    )

    assert signal["sell_shares"] == position.total_shares * 0.25


def test_regime_transition_sell_can_require_breakdown_confirmation():
    strategy, position = _make_regime_strategy(
        {
            "transition_sell_ratio": 0.25,
            "transition_sell_requires_breakdown": True,
        }
    )
    strategy.prev_regime = "BULL"
    strategy.current_regime = "BEAR"

    hold_signal = strategy._check_regime_transition(
        position,
        nav=1.05,
        indicators={"above_ma20": True, "macd_5d_negative": False},
    )
    sell_signal = strategy._check_regime_transition(
        position,
        nav=1.02,
        indicators={"above_ma20": False, "macd_5d_negative": False},
    )

    assert hold_signal == {}
    assert sell_signal["sell_shares"] == position.total_shares * 0.25


def test_bull_follow_through_entry_requires_trend_confirmation():
    strategy, position = _make_regime_strategy(
        {
            "bull_follow_through_entry_enabled": True,
            "trend_batch_ratios": [0.25, 0.20, 0.15],
        }
    )
    strategy.current_regime = "BULL"

    weak_signal = strategy._handle_normal_zone(
        {
            "gold_cross": False,
            "rsi": 50,
            "macd_turn_positive": False,
            "above_ma20_3d": True,
            "macd_hist": 0.02,
            "death_cross": False,
            "macd_5d_negative": False,
        },
        position,
        nav=1.02,
    )
    follow_signal = strategy._handle_normal_zone(
        {
            "gold_cross": False,
            "rsi": 75,
            "macd_turn_positive": True,
            "above_ma20_3d": True,
            "macd_hist": 0.02,
            "death_cross": False,
            "macd_5d_negative": False,
        },
        position,
        nav=1.02,
    )

    assert weak_signal == {}
    assert follow_signal["buy_amount"] > 0


def test_bull_follow_through_entry_disabled_keeps_original_threshold():
    strategy, position = _make_regime_strategy(
        {
            "bull_follow_through_entry_enabled": False,
            "trend_batch_ratios": [0.25, 0.20, 0.15],
        }
    )
    strategy.current_regime = "BULL"

    signal = strategy._handle_normal_zone(
        {
            "gold_cross": False,
            "rsi": 75,
            "macd_turn_positive": True,
            "above_ma20_3d": True,
            "macd_hist": 0.02,
            "death_cross": False,
            "macd_5d_negative": False,
        },
        position,
        nav=1.02,
    )

    assert signal == {}
