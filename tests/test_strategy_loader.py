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
    assert context.default_strategy == "regime_adaptive"
    assert context.effective_strategy == "regime_adaptive"
    assert [item["id"] for item in context.available_strategies] == ["v6", "regime_adaptive"]


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

    assert context.effective_strategy == "regime_adaptive"
    assert context.strategy_adjusted is True
    assert context.adjustment_reason == "当前画像不支持 index_momentum，已切换为默认策略"


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
