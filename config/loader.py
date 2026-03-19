"""
策略配置加载与画像解析。

优先使用 config/strategy_profiles.json 的画像分层配置；若新配置缺失或解析失败，
则回退到 legacy config/strategy_params.json。
"""

from __future__ import annotations

import copy
import json
import os
import logging
from dataclasses import dataclass
from typing import Any, Optional

from otc_fund_quant.analysis.strategy_registry import get_strategy_catalog, is_known_strategy
from otc_fund_quant.nav_estimator.core.fund_classifier import FundClassifier
from otc_fund_quant.nav_estimator.data.fetcher.fund_fetcher import FundFetcher

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_LEGACY_CONFIG_PATH = os.path.join(_CONFIG_DIR, "strategy_params.json")
_PROFILE_CONFIG_PATH = os.path.join(_CONFIG_DIR, "strategy_profiles.json")

_legacy_config_cache: Optional[dict[str, Any]] = None
_profile_config_cache: Optional[dict[str, Any]] = None
_fund_fetcher: Optional[FundFetcher] = None
_fund_classifier: Optional[FundClassifier] = None
logger = logging.getLogger(__name__)

ACTIVE_EQUITY_CN_DEFAULTS: dict[str, Any] = {
    "max_position_ratio": 0.92,
    "soft_stop_loss": -0.18,
    "hard_stop_loss": -0.25,
    "min_cooldown": 6,
    "max_cooldown": 18,
    "dca_base_ratio": 0.05,
    "dca_interval": 6,
    "cheap_percentile": 0.2,
    "batch_interval": 3,
    "min_history_days": 250,
    "trend_batch_ratios": [0.25, 0.20, 0.15],
    "re_entry_strong_ratio": 0.9,
    "re_entry_weak_ratio": 0.75,
    "low_position_ratio": 0.3,
    "rsi_sell_threshold": 70,
    "adx_trend_threshold": 25,
    "bull_dca_boost": 1.4,
    "transition_buy_ratio": 0.15,
    "transition_sell_ratio": 0.12,
    "momentum_entry_threshold": 2,
    "follow_through_min_score": 1,
    "tail_size": 320,
    "percentile_window": 250,
}

ACTIVE_EQUITY_HK_DEFAULTS: dict[str, Any] = {
    "max_position_ratio": 0.72,
    "soft_stop_loss": -0.16,
    "hard_stop_loss": -0.23,
    "min_cooldown": 10,
    "max_cooldown": 28,
    "dca_base_ratio": 0.03,
    "dca_interval": 9,
    "cheap_percentile": 0.18,
    "batch_interval": 4,
    "min_history_days": 250,
    "trend_batch_ratios": [0.16, 0.12, 0.10],
    "re_entry_strong_ratio": 0.75,
    "re_entry_weak_ratio": 0.55,
    "low_position_ratio": 0.25,
    "rsi_sell_threshold": 68,
    "adx_trend_threshold": 23,
    "bull_dca_boost": 1.2,
    "transition_buy_ratio": 0.1,
    "transition_sell_ratio": 0.18,
    "momentum_entry_threshold": 2,
    "follow_through_min_score": 2,
    "bear_exit_rsi": 58,
    "tail_size": 320,
    "percentile_window": 220,
}

PROFILE_TEMPLATES: dict[str, dict[str, Any]] = {
    "equity_active_cn": {
        "label": "A股主动权益",
        "description": "主动权益基金默认画像",
        "default_strategy": "active_equity_cn",
        "allowed_strategies": ["v6", "regime_adaptive", "active_equity_cn"],
        "strategy_params": {
            "v6": {},
            "regime_adaptive": {},
            "active_equity_cn": {},
        },
    },
    "equity_active_hk": {
        "label": "港股主动权益",
        "description": "港股主动权益默认画像",
        "default_strategy": "active_equity_hk",
        "allowed_strategies": ["v6", "regime_adaptive", "active_equity_hk"],
        "strategy_params": {
            "v6": {
                "max_position_ratio": 0.70,
                "dca_base_ratio": 0.03,
                "dca_interval": 9,
                "soft_stop_loss": -0.16,
                "hard_stop_loss": -0.23,
                "re_entry_strong_ratio": 0.75,
                "re_entry_weak_ratio": 0.55,
            },
            "regime_adaptive": {
                "max_position_ratio": 0.70,
                "dca_base_ratio": 0.03,
                "dca_interval": 9,
                "soft_stop_loss": -0.16,
                "hard_stop_loss": -0.23,
                "re_entry_strong_ratio": 0.75,
                "re_entry_weak_ratio": 0.55,
                "bull_dca_boost": 1.30,
                "transition_buy_ratio": 0.12,
                "transition_sell_ratio": 0.35,
            },
            "active_equity_hk": {},
        },
    },
    "equity_index_cn": {
        "label": "A股指数",
        "description": "A股指数/ETF联接默认画像",
        "default_strategy": "index_momentum",
        "allowed_strategies": ["v6", "regime_adaptive", "index_momentum"],
        "strategy_params": {
            "v6": {},
            "regime_adaptive": {},
            "index_momentum": {},
        },
    },
    "equity_index_hk": {
        "label": "港股指数",
        "description": "港股指数默认画像",
        "default_strategy": "index_momentum",
        "allowed_strategies": ["v6", "regime_adaptive", "index_momentum"],
        "strategy_params": {
            "v6": {
                "max_position_ratio": 0.75,
                "dca_base_ratio": 0.03,
            },
            "regime_adaptive": {
                "max_position_ratio": 0.75,
                "dca_base_ratio": 0.03,
                "bull_dca_boost": 1.25,
                "transition_buy_ratio": 0.10,
                "transition_sell_ratio": 0.32,
            },
            "index_momentum": {
                "max_position_ratio": 0.85,
                "dca_base_ratio": 0.06,
                "dca_normal_interval": 6,
                "momentum_buy_threshold": 3,
                "rsi_overbought": 80,
                "atr_scale_max": 1.6,
                "trend_batch_ratios": [0.20, 0.15, 0.10],
            },
        },
    },
    "bond_pure": {
        "label": "纯债",
        "description": "纯债基金保守画像",
        "default_strategy": "bond_stability",
        "allowed_strategies": ["bond_stability"],
        "strategy_params": {
            "bond_stability": {
                "max_position_ratio": 0.60,
                "dca_base_ratio": 0.02,
                "dca_interval": 10,
                "volatility_guard_window": 60,
                "pullback_buy_threshold": 0.015,
                "drawdown_exit_threshold": 0.025,
                "trend_confirm_window": 5,
                "reentry_window": 3,
                "min_history_days": 140,
                "tail_size": 240,
                "percentile_window": 120,
            },
        },
    },
    "bond_plus": {
        "label": "固收+",
        "description": "固收+基金保守画像",
        "default_strategy": "bond_plus_balance",
        "allowed_strategies": ["v6", "regime_adaptive", "bond_plus_balance"],
        "strategy_params": {
            "v6": {
                "max_position_ratio": 0.65,
                "dca_base_ratio": 0.03,
                "dca_interval": 8,
                "cheap_percentile": 0.18,
                "soft_stop_loss": -0.12,
                "hard_stop_loss": -0.18,
                "trend_batch_ratios": [0.12, 0.10, 0.08],
            },
            "regime_adaptive": {
                "max_position_ratio": 0.65,
                "dca_base_ratio": 0.03,
                "dca_interval": 8,
                "cheap_percentile": 0.18,
                "soft_stop_loss": -0.12,
                "hard_stop_loss": -0.18,
                "trend_batch_ratios": [0.12, 0.10, 0.08],
                "adx_trend_threshold": 28,
                "bull_dca_boost": 1.20,
                "transition_buy_ratio": 0.10,
                "transition_sell_ratio": 0.25,
            },
            "bond_plus_balance": {
                "max_position_ratio": 0.68,
                "dca_base_ratio": 0.03,
                "dca_interval": 8,
                "drawdown_guard_threshold": 0.05,
                "momentum_confirm_threshold": 1,
                "macd_negative_days": 4,
                "batch_ratios": [0.12, 0.08],
                "min_cooldown": 8,
                "max_cooldown": 20,
                "min_history_days": 140,
                "tail_size": 240,
            },
        },
    },
    "qdii_global": {
        "label": "QDII",
        "description": "QDII基金默认画像",
        "default_strategy": "qdii_trend",
        "allowed_strategies": ["v6", "regime_adaptive", "qdii_trend"],
        "strategy_params": {
            "v6": {
                "max_position_ratio": 0.70,
                "dca_base_ratio": 0.03,
                "dca_interval": 9,
                "cheap_percentile": 0.18,
                "min_cooldown": 10,
                "max_cooldown": 30,
                "soft_stop_loss": -0.15,
                "hard_stop_loss": -0.22,
                "tail_size": 300,
                "percentile_window": 220,
            },
            "regime_adaptive": {
                "max_position_ratio": 0.70,
                "dca_base_ratio": 0.03,
                "dca_interval": 9,
                "cheap_percentile": 0.18,
                "min_cooldown": 10,
                "max_cooldown": 30,
                "soft_stop_loss": -0.15,
                "hard_stop_loss": -0.22,
                "tail_size": 300,
                "percentile_window": 220,
                "adx_trend_threshold": 23,
                "bull_dca_boost": 1.25,
                "transition_buy_ratio": 0.10,
                "transition_sell_ratio": 0.30,
            },
            "qdii_trend": {
                "max_position_ratio": 0.72,
                "dca_base_ratio": 0.03,
                "momentum_entry_threshold": 2,
                "trend_window_fast": 20,
                "trend_window_mid": 60,
                "trend_window_slow": 120,
                "pullback_reentry_threshold": 0.04,
                "atr_exit_multiplier": 1.8,
                "sell_cooldown": 30,
                "min_cooldown": 10,
                "max_cooldown": 30,
                "min_history_days": 140,
                "tail_size": 260,
            },
        },
    },
}

FUND_TYPE_PROFILE_MAP: dict[str, str] = {
    "active_a": "equity_active_cn",
    "active_hk": "equity_active_hk",
    "index_a": "equity_index_cn",
    "index_hk": "equity_index_hk",
    "bond_pure": "bond_pure",
    "bond_plus": "bond_plus",
    "qdii": "qdii_global",
}


@dataclass(frozen=True)
class ResolvedStrategyContext:
    fund_code: str
    fund_name: Optional[str]
    fund_type: Optional[str]
    profile_id: str
    profile_label: str
    requested_strategy: str
    effective_strategy: str
    default_strategy: str
    available_strategies: list[dict[str, str]]
    strategy_params: dict[str, Any]
    strategy_adjusted: bool
    adjustment_reason: Optional[str]

    def to_strategy_context_payload(self) -> dict[str, Any]:
        return {
            "fund_type": self.fund_type,
            "profile_id": self.profile_id,
            "profile_label": self.profile_label,
            "requested_strategy": self.requested_strategy,
            "effective_strategy": self.effective_strategy,
            "default_strategy": self.default_strategy,
            "strategy_adjusted": self.strategy_adjusted,
            "adjustment_reason": self.adjustment_reason,
            "available_strategies": copy.deepcopy(self.available_strategies),
        }


def _load_json_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_legacy_config() -> dict[str, Any]:
    global _legacy_config_cache
    if _legacy_config_cache is not None:
        return _legacy_config_cache
    _legacy_config_cache = _load_json_config(_LEGACY_CONFIG_PATH)
    return _legacy_config_cache


def _load_profile_config() -> dict[str, Any]:
    global _profile_config_cache
    if _profile_config_cache is not None:
        return _profile_config_cache
    if not os.path.exists(_PROFILE_CONFIG_PATH):
        raise FileNotFoundError(_PROFILE_CONFIG_PATH)
    _profile_config_cache = _load_json_config(_PROFILE_CONFIG_PATH)
    return _profile_config_cache


def _get_fund_fetcher() -> FundFetcher:
    global _fund_fetcher
    if _fund_fetcher is None:
        _fund_fetcher = FundFetcher()
    return _fund_fetcher


def _get_fund_classifier() -> FundClassifier:
    global _fund_classifier
    if _fund_classifier is None:
        _fund_classifier = FundClassifier()
    return _fund_classifier


def _fetch_fund_info(fund_code: str) -> dict[str, Any]:
    return _get_fund_fetcher().get_fund_info(fund_code)


def _classify_fund(fund_code: str, fund_info: dict[str, Any] | None = None) -> str:
    return _get_fund_classifier().classify(fund_code, fund_info=fund_info)


def _extract_fund_name(fund_info: dict[str, Any] | None) -> Optional[str]:
    if not isinstance(fund_info, dict):
        return None
    fund_name = str(fund_info.get("name", "")).strip()
    return fund_name or None


def _safe_get_fund_info(fund_code: str, fund_info: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if fund_info is not None:
        return fund_info
    try:
        return _fetch_fund_info(fund_code)
    except Exception:
        return None


def _merge_params(*sources: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        if isinstance(source, dict):
            merged.update(copy.deepcopy(source))
    return merged


def _normalize_strategy_id(strategy: str | None, *, fallback: str = "v6") -> str:
    normalized = str(strategy or "").strip()
    if not normalized:
        return fallback
    if is_known_strategy(normalized):
        return normalized
    return fallback


def _resolve_effective_strategy(
    requested_strategy: str,
    allowed_strategies: list[str],
    default_strategy: str,
) -> tuple[str, bool, Optional[str]]:
    normalized_default = _normalize_strategy_id(default_strategy, fallback="v6")
    requested = str(requested_strategy or "").strip()
    if not requested:
        return normalized_default, False, None

    if requested in allowed_strategies:
        return requested, False, None

    if is_known_strategy(requested):
        reason = f"当前画像不支持 {requested}，已切换为默认策略"
    else:
        reason = f"未知策略 {requested}，已切换为默认策略"
    return normalized_default, True, reason


def _load_legacy_strategy_params_from_config(
    config: dict[str, Any],
    fund_code: str,
    strategy: str,
) -> dict[str, Any]:
    defaults = config.get("default", {})
    base_params = copy.deepcopy(defaults.get("v6", {}))

    if strategy in {"index_momentum", "bond_stability", "qdii_trend", "bond_plus_balance"}:
        params = copy.deepcopy(defaults.get(strategy, {}))
        params.update(copy.deepcopy(config.get(fund_code, {}).get(strategy, {})))
        return params

    if strategy in {"active_equity_cn", "active_equity_hk"}:
        dedicated_defaults = (
            copy.deepcopy(ACTIVE_EQUITY_CN_DEFAULTS)
            if strategy == "active_equity_cn"
            else copy.deepcopy(ACTIVE_EQUITY_HK_DEFAULTS)
        )
        params = dedicated_defaults
        params.update(copy.deepcopy(config.get(fund_code, {}).get(strategy, {})))
        return params

    if strategy == "v6":
        params = base_params
        params.update(copy.deepcopy(config.get(fund_code, {}).get("v6", {})))
        return params

    fund_v6_overrides = copy.deepcopy(config.get(fund_code, {}).get("v6", {}))
    if fund_v6_overrides:
        base_params.update(fund_v6_overrides)

    regime_defaults = copy.deepcopy(defaults.get("regime_adaptive", {}))
    regime_defaults.pop("_inherit", None)
    params = {**base_params, **regime_defaults}
    params.update(copy.deepcopy(config.get(fund_code, {}).get("regime_adaptive", {})))
    return params


def _resolve_legacy_strategy_context(
    fund_code: str,
    requested_strategy: str,
    *,
    fund_info: dict[str, Any] | None = None,
) -> ResolvedStrategyContext:
    legacy_config = _load_legacy_config()
    normalized_requested = str(requested_strategy or "").strip()
    default_strategy = "v6"
    allowed_strategies = [item["id"] for item in get_strategy_catalog()]
    effective_strategy, strategy_adjusted, adjustment_reason = _resolve_effective_strategy(
        normalized_requested,
        allowed_strategies,
        default_strategy,
    )
    strategy_params = _load_legacy_strategy_params_from_config(legacy_config, fund_code, effective_strategy)

    resolved_fund_info = _safe_get_fund_info(fund_code, fund_info=fund_info)
    fund_name = _extract_fund_name(resolved_fund_info)
    fund_type = None
    if resolved_fund_info is not None:
        try:
            fund_type = _classify_fund(fund_code, resolved_fund_info)
        except Exception:
            fund_type = None

    return ResolvedStrategyContext(
        fund_code=fund_code,
        fund_name=fund_name,
        fund_type=fund_type,
        profile_id="legacy_default",
        profile_label="兼容旧配置",
        requested_strategy=normalized_requested,
        effective_strategy=effective_strategy,
        default_strategy=default_strategy,
        available_strategies=get_strategy_catalog(allowed_strategies),
        strategy_params=strategy_params,
        strategy_adjusted=strategy_adjusted,
        adjustment_reason=adjustment_reason,
    )


def _resolve_profile_strategy_context(
    fund_code: str,
    requested_strategy: str,
    *,
    fund_info: dict[str, Any] | None = None,
) -> ResolvedStrategyContext:
    profile_config = _load_profile_config()
    strategy_defaults = profile_config["strategy_defaults"]
    profiles = profile_config["profiles"]
    fund_type_profiles = profile_config["fund_type_profiles"]
    fund_overrides = profile_config.get("fund_overrides", {})
    fund_override = fund_overrides.get(fund_code, {})

    resolved_fund_info = _safe_get_fund_info(fund_code, fund_info=fund_info)
    fund_name = _extract_fund_name(resolved_fund_info)

    profile_id = str(fund_override.get("profile", "")).strip()
    fund_type = None
    if resolved_fund_info is not None:
        fund_type = _classify_fund(fund_code, resolved_fund_info)

    if not profile_id:
        if not fund_type:
            raise ValueError(f"无法识别基金 {fund_code} 的基金类型")
        profile_id = str(fund_type_profiles[fund_type]).strip()

    if profile_id not in profiles:
        raise KeyError(f"画像 {profile_id} 未定义")

    profile = profiles[profile_id]
    allowed_strategies = [
        strategy_id
        for strategy_id in profile.get("allowed_strategies", [])
        if is_known_strategy(strategy_id)
    ]
    if not allowed_strategies:
        raise ValueError(f"画像 {profile_id} 未配置可用策略")

    default_strategy = _normalize_strategy_id(
        profile.get("default_strategy"),
        fallback=allowed_strategies[0],
    )
    effective_strategy, strategy_adjusted, adjustment_reason = _resolve_effective_strategy(
        requested_strategy,
        allowed_strategies,
        default_strategy,
    )

    strategy_params = _merge_params(
        strategy_defaults.get(effective_strategy, {}),
        profile.get("strategy_params", {}).get(effective_strategy, {}),
        fund_override.get("strategy_params", {}).get(effective_strategy, {}),
    )

    return ResolvedStrategyContext(
        fund_code=fund_code,
        fund_name=fund_name,
        fund_type=fund_type,
        profile_id=profile_id,
        profile_label=str(profile.get("label", profile_id)).strip() or profile_id,
        requested_strategy=str(requested_strategy or "").strip(),
        effective_strategy=effective_strategy,
        default_strategy=default_strategy,
        available_strategies=get_strategy_catalog(allowed_strategies),
        strategy_params=strategy_params,
        strategy_adjusted=strategy_adjusted,
        adjustment_reason=adjustment_reason,
    )


def resolve_strategy_context(
    fund_code: str,
    requested_strategy: str = "",
    *,
    fund_info: dict[str, Any] | None = None,
) -> ResolvedStrategyContext:
    """根据基金代码解析最终策略上下文。"""
    normalized_fund_code = str(fund_code or "").strip()
    normalized_requested_strategy = str(requested_strategy or "").strip()

    if not normalized_fund_code:
        raise ValueError("fund_code 不能为空")

    try:
        context = _resolve_profile_strategy_context(
            fund_code=normalized_fund_code,
            requested_strategy=normalized_requested_strategy,
            fund_info=fund_info,
        )
        logger.info(
            "strategy_context_resolved fund_code=%s requested_strategy=%s source=profile profile_id=%s effective_strategy=%s default_strategy=%s available_count=%s",
            normalized_fund_code,
            normalized_requested_strategy or "AUTO",
            context.profile_id,
            context.effective_strategy,
            context.default_strategy,
            len(context.available_strategies),
        )
        return context
    except FileNotFoundError as exc:
        logger.warning(
            "strategy_context_legacy_fallback fund_code=%s requested_strategy=%s reason=%s",
            normalized_fund_code,
            normalized_requested_strategy or "AUTO",
            exc,
        )
        context = _resolve_legacy_strategy_context(
            fund_code=normalized_fund_code,
            requested_strategy=normalized_requested_strategy,
            fund_info=fund_info,
        )
        logger.info(
            "strategy_context_resolved fund_code=%s requested_strategy=%s source=legacy profile_id=%s effective_strategy=%s default_strategy=%s available_count=%s",
            normalized_fund_code,
            normalized_requested_strategy or "AUTO",
            context.profile_id,
            context.effective_strategy,
            context.default_strategy,
            len(context.available_strategies),
        )
        return context
    except Exception as exc:
        logger.exception(
            "strategy_context_failed fund_code=%s requested_strategy=%s error=%s",
            normalized_fund_code,
            normalized_requested_strategy or "AUTO",
            exc,
        )
        raise


def load_strategy_params(fund_code: str, strategy: str) -> dict[str, Any]:
    """兼容旧接口，只返回最终参数。"""
    return resolve_strategy_context(fund_code, strategy).strategy_params


def reload_config():
    """强制重新加载配置文件（修改配置后调用）。"""
    global _legacy_config_cache, _profile_config_cache
    _legacy_config_cache = None
    _profile_config_cache = None
    return {
        "legacy": _load_legacy_config(),
        "profile": _load_profile_config() if os.path.exists(_PROFILE_CONFIG_PATH) else None,
    }


def get_all_fund_configs() -> dict[str, list[str]]:
    """获取所有已配置基金的策略键列表。"""
    if os.path.exists(_PROFILE_CONFIG_PATH):
        try:
            config = _load_profile_config()
            fund_overrides = config.get("fund_overrides", {})
            return {
                fund_code: sorted(strategy_params.keys())
                for fund_code, override in fund_overrides.items()
                for strategy_params in [override.get("strategy_params", {})]
                if strategy_params
            }
        except Exception:
            pass

    config = _load_legacy_config()
    return {k: list(v.keys()) for k, v in config.items() if k != "default"}


def build_profile_config_from_legacy(
    legacy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将旧策略配置转换为画像分层配置。"""
    config = copy.deepcopy(legacy_config or _load_legacy_config())
    defaults = copy.deepcopy(config.get("default", {}))

    v6_defaults = defaults.get("v6", {})
    regime_defaults = copy.deepcopy(v6_defaults)
    regime_defaults.update(copy.deepcopy(defaults.get("regime_adaptive", {})))
    regime_defaults.pop("_inherit", None)

    strategy_defaults = {
        "v6": copy.deepcopy(v6_defaults),
        "regime_adaptive": regime_defaults,
        "active_equity_cn": copy.deepcopy(ACTIVE_EQUITY_CN_DEFAULTS),
        "active_equity_hk": copy.deepcopy(ACTIVE_EQUITY_HK_DEFAULTS),
        "index_momentum": copy.deepcopy(defaults.get("index_momentum", {})),
        "bond_stability": copy.deepcopy(defaults.get("bond_stability", {})),
        "qdii_trend": copy.deepcopy(defaults.get("qdii_trend", {})),
        "bond_plus_balance": copy.deepcopy(defaults.get("bond_plus_balance", {})),
    }

    fund_overrides: dict[str, Any] = {}
    for fund_code, strategy_map in config.items():
        if fund_code == "default" or not isinstance(strategy_map, dict):
            continue
        override_strategy_params = {
            strategy_id: copy.deepcopy(params)
            for strategy_id, params in strategy_map.items()
            if isinstance(params, dict) and params
        }
        if override_strategy_params:
            fund_overrides[fund_code] = {
                "strategy_params": override_strategy_params,
            }

    return {
        "version": 2,
        "strategy_defaults": strategy_defaults,
        "profiles": copy.deepcopy(PROFILE_TEMPLATES),
        "fund_type_profiles": copy.deepcopy(FUND_TYPE_PROFILE_MAP),
        "fund_overrides": fund_overrides,
    }
