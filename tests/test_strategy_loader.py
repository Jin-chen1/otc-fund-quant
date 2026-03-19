import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


sys.modules.setdefault("akshare", MagicMock())

PROJECT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from otc_fund_quant.config import loader as strategy_loader


@pytest.fixture(autouse=True)
def _reset_loader():
    strategy_loader.reload_config()
    yield
    strategy_loader.reload_config()


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_resolve_strategy_context_maps_active_fund_to_profile(monkeypatch):
    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试主动权益"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: "active_a")

    context = strategy_loader.resolve_strategy_context("007343", "")

    assert context.profile_id == "equity_active_cn"
    assert context.default_strategy == "active_equity_cn"
    assert context.effective_strategy == "active_equity_cn"
    assert [item["id"] for item in context.available_strategies] == ["v6", "regime_adaptive", "active_equity_cn"]


def test_resolve_strategy_context_maps_index_fund_to_profile(monkeypatch):
    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试指数基金"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: "index_a")

    context = strategy_loader.resolve_strategy_context("022464", "")

    assert context.profile_id == "equity_index_cn"
    assert context.default_strategy == "index_momentum"
    assert context.effective_strategy == "index_momentum"
    assert [item["id"] for item in context.available_strategies] == ["v6", "regime_adaptive", "index_momentum"]


def test_resolve_strategy_context_adjusts_disallowed_strategy(monkeypatch):
    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试主动权益"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: "active_a")

    context = strategy_loader.resolve_strategy_context("007343", "index_momentum")

    assert context.effective_strategy == "active_equity_cn"
    assert context.strategy_adjusted is True
    assert context.adjustment_reason == "当前画像不支持 index_momentum，已切换为默认策略"


@pytest.mark.parametrize(
    ("fund_type", "profile_id", "default_strategy", "available_ids", "requested_strategy", "effective_strategy"),
    [
        ("active_a", "equity_active_cn", "active_equity_cn", ["v6", "regime_adaptive", "active_equity_cn"], "", "active_equity_cn"),
        ("active_hk", "equity_active_hk", "active_equity_hk", ["v6", "regime_adaptive", "active_equity_hk"], "", "active_equity_hk"),
        ("index_a", "equity_index_cn", "index_momentum", ["v6", "regime_adaptive", "index_momentum"], "", "index_momentum"),
        ("index_hk", "equity_index_hk", "index_momentum", ["v6", "regime_adaptive", "index_momentum"], "", "index_momentum"),
        ("bond_pure", "bond_pure", "bond_stability", ["bond_stability"], "", "bond_stability"),
        ("bond_plus", "bond_plus", "bond_plus_balance", ["v6", "regime_adaptive", "bond_plus_balance"], "", "bond_plus_balance"),
        ("qdii", "qdii_global", "qdii_trend", ["v6", "regime_adaptive", "qdii_trend"], "", "qdii_trend"),
        ("bond_pure", "bond_pure", "bond_stability", ["bond_stability"], "regime_adaptive", "bond_stability"),
        ("bond_plus", "bond_plus", "bond_plus_balance", ["v6", "regime_adaptive", "bond_plus_balance"], "regime_adaptive", "regime_adaptive"),
        ("qdii", "qdii_global", "qdii_trend", ["v6", "regime_adaptive", "qdii_trend"], "regime_adaptive", "regime_adaptive"),
    ],
)
def test_resolve_strategy_context_category_matrix(
    monkeypatch,
    fund_type,
    profile_id,
    default_strategy,
    available_ids,
    requested_strategy,
    effective_strategy,
):
    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试基金"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: fund_type)

    context = strategy_loader.resolve_strategy_context("999999", requested_strategy)

    assert context.profile_id == profile_id
    assert context.default_strategy == default_strategy
    assert context.effective_strategy == effective_strategy
    assert [item["id"] for item in context.available_strategies] == available_ids


@pytest.mark.parametrize(
    ("fund_type", "strategy", "expected_subset"),
    [
        ("active_a", "active_equity_cn", {"max_position_ratio": 0.95, "dca_base_ratio": 0.06, "bull_dca_boost": 1.6}),
        ("active_hk", "v6", {"max_position_ratio": 0.70, "dca_base_ratio": 0.03, "dca_interval": 9}),
        ("active_hk", "regime_adaptive", {"bull_dca_boost": 1.30, "transition_buy_ratio": 0.12, "transition_sell_ratio": 0.35}),
        ("active_hk", "active_equity_hk", {"max_position_ratio": 0.72, "dca_base_ratio": 0.03, "bear_exit_rsi": 58}),
        ("index_hk", "index_momentum", {"max_position_ratio": 0.85, "momentum_buy_threshold": 3, "atr_scale_max": 1.6}),
        ("bond_pure", "bond_stability", {"max_position_ratio": 0.60, "volatility_guard_window": 60, "drawdown_exit_threshold": 0.025}),
        ("bond_plus", "bond_plus_balance", {"max_position_ratio": 0.68, "drawdown_guard_threshold": 0.05, "batch_ratios": [0.12, 0.08]}),
        ("qdii", "qdii_trend", {"max_position_ratio": 0.72, "trend_window_slow": 120, "atr_exit_multiplier": 1.8}),
    ],
)
def test_resolve_strategy_context_applies_category_specific_params(monkeypatch, fund_type, strategy, expected_subset):
    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试基金"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: fund_type)

    context = strategy_loader.resolve_strategy_context("999999", strategy)

    for key, expected_value in expected_subset.items():
        assert context.strategy_params[key] == expected_value


def test_resolve_strategy_context_applies_015916_regime_override(monkeypatch):
    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试主动权益"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: "active_a")

    context = strategy_loader.resolve_strategy_context("015916", "regime_adaptive")

    assert context.profile_id == "equity_active_cn"
    assert context.effective_strategy == "regime_adaptive"
    expected_subset = {
        "max_position_ratio": 0.95,
        "dca_base_ratio": 0.06,
        "trend_batch_ratios": [0.25, 0.20, 0.15],
        "bull_dca_boost": 1.6,
        "transition_buy_ratio": 0.20,
        "transition_sell_ratio": 0.10,
        "transition_sell_requires_breakdown": True,
        "bull_follow_through_entry_enabled": True,
    }
    for key, expected_value in expected_subset.items():
        assert context.strategy_params[key] == expected_value


def test_resolve_strategy_context_keeps_non_overridden_active_fund_on_profile_defaults(monkeypatch):
    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试主动权益"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: "active_a")

    context = strategy_loader.resolve_strategy_context("007343", "active_equity_cn")

    assert context.strategy_params["max_position_ratio"] == 0.95
    assert context.strategy_params["dca_base_ratio"] == 0.06
    assert context.strategy_params["bull_dca_boost"] == 1.6
    assert context.strategy_params["transition_sell_requires_breakdown"] is True
    assert context.strategy_params["bull_follow_through_entry_enabled"] is True


def test_profile_param_merge_precedence(monkeypatch, tmp_path):
    legacy_path = tmp_path / "strategy_params.json"
    profile_path = tmp_path / "strategy_profiles.json"
    _write_json(
        legacy_path,
        {
            "default": {
                "v6": {"max_position_ratio": 0.8},
                "regime_adaptive": {"_inherit": "v6", "bull_dca_boost": 1.5},
                "index_momentum": {"max_position_ratio": 0.95},
            }
        },
    )
    _write_json(
        profile_path,
        {
            "version": 2,
            "strategy_defaults": {
                "v6": {"alpha": 1, "beta": 1},
                "regime_adaptive": {"alpha": 2},
                "index_momentum": {"alpha": 3},
            },
            "profiles": {
                "equity_active_cn": {
                    "label": "A股主动权益",
                    "description": "主动权益基金默认画像",
                    "default_strategy": "v6",
                    "allowed_strategies": ["v6", "regime_adaptive"],
                    "strategy_params": {
                        "v6": {"beta": 2, "gamma": 2},
                        "regime_adaptive": {}
                    },
                }
            },
            "fund_type_profiles": {
                "active_a": "equity_active_cn"
            },
            "fund_overrides": {
                "000001": {
                    "strategy_params": {
                        "v6": {"gamma": 3, "delta": 4}
                    }
                }
            },
        },
    )
    monkeypatch.setattr(strategy_loader, "_LEGACY_CONFIG_PATH", str(legacy_path))
    monkeypatch.setattr(strategy_loader, "_PROFILE_CONFIG_PATH", str(profile_path))
    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试主动权益"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: "active_a")
    strategy_loader.reload_config()

    context = strategy_loader.resolve_strategy_context("000001", "v6")

    assert context.strategy_params == {"alpha": 1, "beta": 2, "gamma": 3, "delta": 4}


def test_load_strategy_params_falls_back_to_legacy_when_profile_missing(monkeypatch, tmp_path):
    legacy_path = tmp_path / "strategy_params.json"
    _write_json(
        legacy_path,
        {
            "default": {
                "v6": {"max_position_ratio": 0.8, "dca_interval": 7},
                "regime_adaptive": {"_inherit": "v6", "bull_dca_boost": 1.5},
                "index_momentum": {"max_position_ratio": 0.95},
            },
            "000001": {
                "v6": {"dca_interval": 5}
            },
        },
    )
    monkeypatch.setattr(strategy_loader, "_LEGACY_CONFIG_PATH", str(legacy_path))
    monkeypatch.setattr(strategy_loader, "_PROFILE_CONFIG_PATH", str(tmp_path / "missing_strategy_profiles.json"))
    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试基金"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: "active_a")
    strategy_loader.reload_config()

    params = strategy_loader.load_strategy_params("000001", "v6")

    assert params == {"max_position_ratio": 0.8, "dca_interval": 5}


def test_resolve_strategy_context_uses_legacy_only_when_profile_file_missing(monkeypatch):
    legacy_context = strategy_loader.ResolvedStrategyContext(
        fund_code="000001",
        fund_name="测试基金",
        fund_type="active_a",
        profile_id="legacy_default",
        profile_label="兼容旧配置",
        requested_strategy="v6",
        effective_strategy="v6",
        default_strategy="v6",
        available_strategies=[{"id": "v6", "label": "估值趋势", "description": "desc"}],
        strategy_params={"max_position_ratio": 0.8},
        strategy_adjusted=False,
        adjustment_reason=None,
    )
    monkeypatch.setattr(strategy_loader, "_resolve_profile_strategy_context", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing profile")) )
    monkeypatch.setattr(strategy_loader, "_resolve_legacy_strategy_context", lambda *args, **kwargs: legacy_context)

    context = strategy_loader.resolve_strategy_context("000001", "v6")

    assert context.profile_id == "legacy_default"
    assert context.effective_strategy == "v6"


def test_resolve_strategy_context_does_not_silently_fallback_on_profile_errors(monkeypatch):
    monkeypatch.setattr(strategy_loader, "_resolve_profile_strategy_context", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("画像 broken 未定义")) )
    monkeypatch.setattr(strategy_loader, "_resolve_legacy_strategy_context", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fallback")) )

    with pytest.raises(ValueError, match="画像 broken 未定义"):
        strategy_loader.resolve_strategy_context("000001", "v6")
