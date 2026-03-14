import os
import sys
import http.client
import sqlite3
from concurrent.futures import Future
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests


sys.modules.setdefault("akshare", MagicMock())

PROJECT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from otc_fund_quant.nav_estimator.core.batch_context import BatchContext
from otc_fund_quant.nav_estimator.core.fund_classifier import FundClassifier
from otc_fund_quant.nav_estimator.core.quality_gate import QualityGateError
from otc_fund_quant.nav_estimator.core.nav_engine import NAVEngine
from otc_fund_quant.config import loader as strategy_loader
from otc_fund_quant.config.loader import ResolvedStrategyContext
from otc_fund_quant.nav_estimator.config.settings import CACHE_DIR
from otc_fund_quant.nav_estimator.data.fetcher.base_fetcher import BaseFetcher
from otc_fund_quant.nav_estimator.data.fetcher.fund_fetcher import FundFetcher
from otc_fund_quant.nav_estimator.estimator.active_equity_estimator import ActiveEquityEstimator
from otc_fund_quant.nav_estimator.estimator.bond_estimator import BondEstimator
from otc_fund_quant.nav_estimator.estimator.index_estimator import IndexEstimator
from otc_fund_quant.nav_estimator.estimator.qdii_hk_estimator import QDIIHKEstimator
from otc_fund_quant.data import sqlite_utils
from otc_fund_quant.web import app as web_app


def _client():
    web_app.app.config["TESTING"] = True
    return web_app.app.test_client()


@pytest.fixture(autouse=True)
def _reset_backtest_runtime_state():
    web_app._reset_backtest_runtime_state(wait=False)
    yield
    web_app._reset_backtest_runtime_state(wait=False)


def _resolved_strategy_context(
    *,
    fund_code: str = "007343",
    fund_name: str | None = "测试基金",
    fund_type: str | None = "active_a",
    profile_id: str = "equity_active_cn",
    profile_label: str = "A股主动权益",
    requested_strategy: str = "v6",
    effective_strategy: str = "v6",
    default_strategy: str = "regime_adaptive",
    strategy_params: dict | None = None,
    strategy_adjusted: bool = False,
    adjustment_reason: str | None = None,
    available_ids: list[str] | None = None,
):
    catalog_map = {
        "v6": {"id": "v6", "label": "估值趋势", "description": "desc-v6"},
        "regime_adaptive": {"id": "regime_adaptive", "label": "状态自适应", "description": "desc-regime"},
        "index_momentum": {"id": "index_momentum", "label": "指数动量", "description": "desc-index"},
        "bond_stability": {"id": "bond_stability", "label": "纯债稳健", "description": "desc-bond"},
        "qdii_trend": {"id": "qdii_trend", "label": "QDII 趋势", "description": "desc-qdii"},
        "bond_plus_balance": {"id": "bond_plus_balance", "label": "固收+平衡", "description": "desc-bond-plus"},
    }
    return ResolvedStrategyContext(
        fund_code=fund_code,
        fund_name=fund_name,
        fund_type=fund_type,
        profile_id=profile_id,
        profile_label=profile_label,
        requested_strategy=requested_strategy,
        effective_strategy=effective_strategy,
        default_strategy=default_strategy,
        available_strategies=[catalog_map[item] for item in (available_ids or ["v6", "regime_adaptive"])],
        strategy_params=strategy_params or {"max_position_ratio": 1.0},
        strategy_adjusted=strategy_adjusted,
        adjustment_reason=adjustment_reason,
    )


def _active_holdings():
    return pd.DataFrame(
        [
            {"code": "600000", "name": "A", "weight": 20.0, "market": "A股"},
            {"code": "000001", "name": "B", "weight": 15.0, "market": "A股"},
        ]
    )


def _active_batch_context_with_errors():
    context = BatchContext(data_as_of_date="2026-03-08")
    context.a_prices_error = "proxy failed"
    context.a_index_errors["000300"] = "index proxy failed"
    context.fx_error = "fx proxy failed"
    return context


def _qdii_batch_context_with_errors():
    context = BatchContext(data_as_of_date="2026-03-08")
    context.a_prices_error = "proxy failed"
    context.hk_prices_error = "proxy failed"
    context.a_index_errors["000300"] = "index proxy failed"
    context.hk_index_errors["HSI"] = "hk index proxy failed"
    context.fx_error = "fx proxy failed"
    return context


def test_retry_on_error_disables_trust_env_after_proxy_failure(monkeypatch):
    monkeypatch.setenv("OTC_FUND_QUANT_PROXY_MODE", "auto")
    original_init = requests.sessions.Session.__init__
    session_states = []
    state = {"count": 0}

    @BaseFetcher.retry_on_error(max_retries=2, delay=0)
    def flaky():
        session = requests.Session()
        session_states.append(session.trust_env)
        state["count"] += 1
        if state["count"] == 1:
            raise RuntimeError("ProxyError: Unable to connect to proxy")
        return "ok"

    assert flaky() == "ok"
    assert session_states == [True, False]
    assert requests.sessions.Session.__init__ is original_init


def test_retry_on_error_direct_mode_disables_trust_env_immediately(monkeypatch):
    monkeypatch.setenv("OTC_FUND_QUANT_PROXY_MODE", "direct")

    @BaseFetcher.retry_on_error(max_retries=1, delay=0)
    def direct_call():
        session = requests.Session()
        return session.trust_env

    assert direct_call() is False


def test_request_scope_latches_direct_mode_after_first_proxy_failure(monkeypatch):
    monkeypatch.setenv("OTC_FUND_QUANT_PROXY_MODE", "auto")
    session_states = []
    state = {"count": 0}

    @BaseFetcher.retry_on_error(max_retries=2, delay=0)
    def flaky():
        session = requests.Session()
        session_states.append(session.trust_env)
        state["count"] += 1
        if state["count"] == 1:
            raise RuntimeError("ProxyError: Unable to connect to proxy")
        return "ok"

    @BaseFetcher.retry_on_error(max_retries=1, delay=0)
    def followup():
        session = requests.Session()
        session_states.append(session.trust_env)
        return "done"

    with BaseFetcher.request_scope():
        assert flaky() == "ok"
        assert followup() == "done"

    assert session_states == [True, False, False]


def test_request_scope_sets_and_restores_strict_mode():
    assert BaseFetcher._get_strict_mode_latched() is None
    with BaseFetcher.request_scope(strict=False):
        assert BaseFetcher._get_strict_mode_latched() is False
    assert BaseFetcher._get_strict_mode_latched() is None


def test_request_scope_latches_non_strict_mode():
    with BaseFetcher.request_scope(strict=False):
        assert BaseFetcher._get_strict_mode_latched() is False
    assert BaseFetcher._get_strict_mode_latched() is None


def test_proxy_error_classifier_excludes_plain_connection_abort():
    error = RuntimeError(f"Connection aborted: {http.client.RemoteDisconnected('Remote end closed connection without response')}")
    assert BaseFetcher._is_proxy_related_error(error) is False


def test_active_equity_non_strict_ignores_batch_context_errors():
    estimator = ActiveEquityEstimator()
    holdings_df = _active_holdings()
    context = _active_batch_context_with_errors()

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", side_effect=ValueError("未获取到资产配置数据")), \
         patch.object(estimator.fund_fetcher, "get_estimated_stock_position_series_latest", return_value=90.75), \
         patch.object(estimator.fund_fetcher, "extract_holdings_report_metadata", return_value=("2025Q4", "2025-12-31")), \
         patch.object(estimator.stock_fetcher, "get_a_share_prices_by_codes", return_value={
             "600000": {"change_pct": 1.2, "data_as_of_date": "2026-03-08"},
             "000001": {"change_pct": -0.5, "data_as_of_date": "2026-03-08"},
         }), \
         patch.object(estimator.index_fetcher, "get_a_index_return", return_value=0.6), \
         patch.object(estimator.fx_fetcher, "get_hkd_cny_daily_change", return_value=0.0):
        result = estimator.estimate(
            fund_code="007343",
            last_nav=3.0,
            nav_date="2026-03-07",
            target_date="2026-03-08",
            holdings=holdings_df,
            strict=False,
            batch_context=context,
        )

    assert result["position_source"] == "default_assumption_fallback"
    assert result["estimated_position_series_latest"] == 90.75
    assert result["estimated_nav"] > 0
    assert any("回退" in item or "转为即时获取" in item for item in result["warnings"])


def test_active_equity_strict_fails_when_quality_gate_not_met():
    estimator = ActiveEquityEstimator()
    holdings_df = _active_holdings()
    context = _active_batch_context_with_errors()

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", return_value={
        "position": 85.0,
        "report_date": "2025-12-31",
        "raw_report_period": "2025-12-31",
        "snapshot_source": "pingzhongdata_asset_allocation",
    }), \
         patch.object(estimator.fund_fetcher, "extract_holdings_report_metadata", return_value=("2025Q4", "2025-12-31")), \
         patch.object(estimator.stock_fetcher, "_get_a_share_quotes_live_by_codes_partial", return_value=({}, ["600000", "000001"])), \
         patch.object(estimator.stock_fetcher, "_get_hk_share_quotes_live_by_codes_partial", return_value=({}, [])), \
         patch.object(estimator.index_fetcher, "get_a_index_return_live", side_effect=ValueError("proxy index down")), \
         patch.object(estimator.fx_fetcher, "get_hkd_cny_daily_change_live", return_value={
             "value": 0.0,
             "source": "safe_mid_rate",
             "source_disagreements": [],
             "data_as_of_date": "2026-03-08",
         }):
        with pytest.raises(QualityGateError) as exc_info:
            estimator.estimate(
                fund_code="007343",
                last_nav=3.0,
                nav_date="2026-03-07",
                target_date="2026-03-08",
                holdings=holdings_df,
                strict=True,
                batch_context=context,
            )

    assert "缺少基金披露股票总仓位" not in exc_info.value.quality_details["quality_gate_failed_reasons"]


def test_active_equity_strict_requires_disclosed_stock_position():
    estimator = ActiveEquityEstimator()
    holdings_df = _active_holdings()

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", side_effect=ValueError("未获取到资产配置数据")), \
         patch.object(estimator.fund_fetcher, "extract_holdings_report_metadata", return_value=("2025Q4", "2025-12-31")):
        with pytest.raises(QualityGateError) as exc_info:
            estimator.estimate(
                fund_code="007343",
                last_nav=3.0,
                nav_date="2026-03-07",
                target_date="2026-03-08",
                holdings=holdings_df,
                strict=True,
            )

    assert exc_info.value.quality_details["quality_gate_failed_reasons"] == ["缺少基金披露股票总仓位"]


def test_qdii_active_non_strict_ignores_batch_context_errors():
    estimator = QDIIHKEstimator()
    holdings_df = pd.DataFrame(
        [{"code": "600000", "name": "A", "weight": 20.0, "market": "A股"}]
    )
    context = _qdii_batch_context_with_errors()

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", side_effect=ValueError("未获取到资产配置数据")), \
         patch.object(estimator.fund_fetcher, "extract_holdings_report_metadata", return_value=("2025Q4", "2025-12-31")), \
         patch.object(estimator.stock_fetcher, "get_a_share_prices_by_codes", return_value={
             "600000": {"change_pct": 1.2, "data_as_of_date": "2026-03-08"},
         }), \
         patch.object(estimator.index_fetcher, "get_a_index_return", return_value=0.5), \
         patch.object(estimator.fx_fetcher, "get_hkd_cny_daily_change", return_value=0.0):
        result = estimator.estimate(
            fund_code="007343",
            last_nav=3.0,
            nav_date="2026-03-07",
            target_date="2026-03-08",
            is_index_fund=False,
            holdings=holdings_df,
            proxy_components=[{"code": "000300", "name": "沪深300", "market": "A股", "weight": 1.0}],
            strict=False,
            batch_context=context,
        )

    assert result["position_source"] == "default_assumption_fallback"
    assert result["estimated_nav"] > 0


def test_qdii_index_strict_supports_us_proxy_components():
    estimator = QDIIHKEstimator()

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", return_value={
        "position": 95.0,
        "report_date": "2025-12-31",
        "raw_report_period": "2025-12-31",
        "snapshot_source": "pingzhongdata_asset_allocation",
    }), \
         patch.object(estimator.index_fetcher, "get_global_index_return_live", return_value={
             "value": 1.2,
             "source": "eastmoney_global_index_quote",
             "source_disagreements": [],
             "data_as_of_date": "2026-03-08",
         }), \
         patch.object(estimator.fx_fetcher, "get_usd_cny_daily_change_live", return_value={
             "value": 0.3,
             "source": "safe_mid_rate",
             "source_disagreements": [],
             "data_as_of_date": "2026-03-08",
         }):
        result = estimator.estimate(
            fund_code="110011",
            last_nav=5.0,
            nav_date="2026-03-07",
            target_date="2026-03-08",
            is_index_fund=True,
            market_profile="us",
            proxy_components=[{"code": "SPX", "name": "标普500", "market": "美股", "weight": 1.0}],
            strict=True,
        )

    assert result["quality_gate_passed"] is True
    assert result["qdii_market_profile"] == "us"
    assert result["estimated_nav"] > 0


def test_index_estimator_non_strict_falls_back_to_live_index_fetch():
    estimator = IndexEstimator()
    context = BatchContext(data_as_of_date="2026-03-08")
    context.a_index_errors["000300"] = "proxy failed"

    with patch.object(estimator.index_fetcher, "get_a_index_return", return_value=0.8):
        result = estimator.estimate(
            fund_code="110011",
            last_nav=1.0,
            nav_date="2026-03-07",
            index_code="000300",
            strict=False,
            batch_context=context,
            position=93.0,
        )

    assert result["estimated_nav"] > 0


def test_index_estimator_strict_ignores_batch_context():
    estimator = IndexEstimator()
    context = BatchContext(data_as_of_date="2026-03-08")
    context.a_index_returns["000300"] = 9.9

    with patch.object(estimator.index_fetcher, "get_a_index_return_live", return_value={
        "value": 0.8,
        "source": "sina_index_single_quote",
        "source_disagreements": [],
        "data_as_of_date": "2026-03-08",
    }):
        result = estimator.estimate(
            fund_code="110011",
            last_nav=1.0,
            nav_date="2026-03-07",
            index_code="000300",
            strict=True,
            batch_context=context,
            position=93.0,
        )

    assert result["index_return"] == 0.8


@pytest.mark.parametrize(
    ("fund_code", "security_code", "tracking_name"),
    [
        ("018345", "562500", "中证机器人ETF"),
        ("012733", "159819", "人工智能ETF"),
        ("007467", "512890", "红利低波ETF"),
        ("023598", "513780", "港股创新药ETF"),
        ("024663", "159246", "创业板人工智能ETF富国"),
    ],
)
def test_index_estimator_strict_uses_linked_etf_tracking_target_quote(fund_code, security_code, tracking_name):
    estimator = IndexEstimator()

    with patch.object(estimator.stock_fetcher, "get_a_share_quote_live", return_value={
        "value": 1.23,
        "source": "eastmoney_single_quote",
        "source_disagreements": [],
        "data_as_of_date": "2026-03-10",
    }) as mock_stock_quote, \
         patch.object(estimator.index_fetcher, "get_a_index_return_live", side_effect=AssertionError("should not fetch index quote")):
        result = estimator.estimate(
            fund_code=fund_code,
            last_nav=1.221,
            nav_date="2026-03-09",
            index_code=security_code,
            tracking_target={
                "target_type": "linked_etf_a_share",
                "security_code": security_code,
                "market": "A股",
                "tracking_name": tracking_name,
            },
            strict=True,
            position=95.0,
        )

    mock_stock_quote.assert_called_once_with(security_code, strict=True)
    assert result["index_code"] == security_code
    assert result["index_return"] == 1.23
    assert result["quality_gate_passed"] is True


def test_stock_fetcher_routes_a_share_etf_codes_to_correct_exchange():
    from otc_fund_quant.nav_estimator.data.fetcher.stock_fetcher import StockFetcher

    assert StockFetcher._get_a_secid("600000") == "1.600000"
    assert StockFetcher._get_a_secid("562500") == "1.562500"
    assert StockFetcher._get_a_secid("560880") == "1.560880"
    assert StockFetcher._get_a_secid("159915") == "0.159915"


def test_stock_fetcher_builds_correct_sina_symbols_for_a_share_etfs():
    from otc_fund_quant.nav_estimator.data.fetcher.stock_fetcher import StockFetcher

    assert StockFetcher._get_a_index_code_for_sina("600000") == "sh600000"
    assert StockFetcher._get_a_index_code_for_sina("562500") == "sh562500"
    assert StockFetcher._get_a_index_code_for_sina("560880") == "sh560880"
    assert StockFetcher._get_a_index_code_for_sina("159915") == "sz159915"


def test_bond_estimator_non_strict_falls_back_to_live_fetch():
    estimator = BondEstimator()
    context = BatchContext(data_as_of_date="2026-03-08")
    context.bond_index_errors["comprehensive"] = "proxy failed"
    context.a_index_errors["000300"] = "proxy failed"
    context.a_index_errors["000832"] = "proxy failed"

    with patch.object(estimator.bond_fetcher, "get_bond_index_snapshot", return_value={
        "change_pct": 0.1,
        "prev_date": "2026-03-06",
        "curr_date": "2026-03-08",
    }), \
         patch.object(estimator.index_fetcher, "get_a_index_return", side_effect=[0.5, 0.2]):
        result = estimator.estimate(
            fund_code="001234",
            last_nav=1.0,
            nav_date="2026-03-07",
            fund_type="plus",
            stock_position=10.0,
            convertible_position=10.0,
            strict=False,
            batch_context=context,
        )

    assert result["estimated_nav"] > 0


def test_non_retryable_stock_position_error_short_circuits():
    state = {"count": 0}

    @BaseFetcher.retry_on_error(max_retries=3, delay=0)
    def flaky():
        state["count"] += 1
        raise ValueError("未获取到资产配置数据")

    with pytest.raises(ValueError):
        flaky()
    assert state["count"] == 1


def test_non_strict_retry_budget_uses_two_attempts():
    state = {"count": 0}

    @BaseFetcher.retry_on_error(max_retries=3, delay=0)
    def flaky():
        state["count"] += 1
        raise RuntimeError("boom")

    with BaseFetcher.request_scope(strict=False):
        with pytest.raises(RuntimeError):
            flaky()

    assert state["count"] == 2


def test_non_strict_retry_budget_is_lighter():
    state = {"count": 0}

    @BaseFetcher.retry_on_error(max_retries=3, delay=0)
    def flaky():
        state["count"] += 1
        raise RuntimeError("boom")

    with BaseFetcher.request_scope(strict=False):
        with pytest.raises(RuntimeError):
            flaky()

    assert state["count"] == 2


def test_non_retryable_error_normalizes_whitespace():
    error = ValueError("未获取到 007343 的资产配置\n数据")
    assert BaseFetcher._is_non_retryable_error(error) is True


def test_non_retryable_duplicate_a_index_mapping_short_circuits():
    state = {"count": 0}

    @BaseFetcher.retry_on_error(max_retries=3, delay=0)
    def flaky():
        state["count"] += 1
        raise ValueError("已识别指数名称 中证医药卫生，但映射到多个A股指数代码: 中证医药(000933), 中证医药(399933)")

    with pytest.raises(ValueError):
        flaky()

    assert state["count"] == 1


def test_non_retryable_unmapped_a_index_short_circuits():
    state = {"count": 0}

    @BaseFetcher.retry_on_error(max_retries=3, delay=0)
    def flaky():
        state["count"] += 1
        raise ValueError("已识别指数名称 恒生A股电网设备，但无法映射到A股指数代码")

    with pytest.raises(ValueError):
        flaky()

    assert state["count"] == 1


def test_base_fetcher_cache_uses_isolated_temp_directory_in_tests():
    assert str(BaseFetcher.cache.cache.directory) != str(CACHE_DIR)


def test_fund_stock_position_snapshot_reads_pingzhongdata_asset_allocation():
    fetcher = FundFetcher()
    js_text = (
        'var Data_assetAllocation = {"series":[{"name":"股票占净比","data":[94.63,93.84,91.18,80.92]},'
        '{"name":"债券占净比","data":[1.7,4.47,4.39,4.94]}],"categories":["2025-03-31","2025-06-30","2025-09-30","2025-12-31"]};'
    )

    with patch.object(fetcher, "_get_pingzhongdata_text", return_value=js_text):
        snapshot = fetcher.get_fund_stock_position_snapshot("900001")

    assert snapshot["position"] == 80.92
    assert snapshot["report_date"] == "2025-12-31"
    assert snapshot["raw_report_period"] == "2025-12-31"
    assert snapshot["snapshot_source"] == "pingzhongdata_asset_allocation"


def test_estimated_stock_position_series_latest_reads_pingzhongdata_series():
    fetcher = FundFetcher()
    js_text = 'var Data_fundSharesPositions = [[1770566400000,80.92],[1772640000000,90.75]];'

    with patch.object(fetcher, "_get_pingzhongdata_text", return_value=js_text):
        latest_position = fetcher.get_estimated_stock_position_series_latest("900002")

    assert latest_position == 90.75


def test_resolve_live_source_uses_backup_when_primary_unavailable():
    result = BaseFetcher.resolve_live_source(
        primary_fetcher=lambda: (_ for _ in ()).throw(ValueError("primary down")),
        backup_fetcher=lambda: BaseFetcher.build_live_payload(
            value=1.2,
            source="backup_source",
            source_priority=2,
            raw={"change_pct": 1.2},
            data_as_of_date="2026-03-08",
        ),
        strict=True,
        label="测试实时源",
        threshold=1.0,
    )

    assert result["source"] == "backup_source"
    assert any("已切换至备源" in item for item in result["warnings"])


def test_resolve_live_source_strict_rejects_large_disagreement():
    with pytest.raises(ValueError):
        BaseFetcher.resolve_live_source(
            primary_fetcher=lambda: BaseFetcher.build_live_payload(
                value=5.0,
                source="primary",
                source_priority=1,
                raw={"change_pct": 5.0},
                data_as_of_date="2026-03-08",
            ),
            backup_fetcher=lambda: BaseFetcher.build_live_payload(
                value=1.0,
                source="backup",
                source_priority=2,
                raw={"change_pct": 1.0},
                data_as_of_date="2026-03-08",
            ),
            strict=True,
            label="测试实时源",
            threshold=2.0,
            probe_backup_on_primary_success=True,
        )


def test_resolve_live_source_primary_success_skips_backup_by_default():
    state = {"backup_called": False}

    result = BaseFetcher.resolve_live_source(
        primary_fetcher=lambda: BaseFetcher.build_live_payload(
            value=1.0,
            source="primary",
            source_priority=1,
            raw={"change_pct": 1.0},
            data_as_of_date="2026-03-08",
        ),
        backup_fetcher=lambda: (
            state.__setitem__("backup_called", True)
            or BaseFetcher.build_live_payload(
                value=1.1,
                source="backup",
                source_priority=2,
                raw={"change_pct": 1.1},
                data_as_of_date="2026-03-08",
            )
        ),
        strict=True,
        label="测试实时源",
        threshold=2.0,
    )

    assert result["source"] == "primary"
    assert state["backup_called"] is False


def test_resolve_live_source_can_force_probe_backup_on_primary_success():
    state = {"backup_called": False}

    result = BaseFetcher.resolve_live_source(
        primary_fetcher=lambda: BaseFetcher.build_live_payload(
            value=1.0,
            source="primary",
            source_priority=1,
            raw={"change_pct": 1.0},
            data_as_of_date="2026-03-08",
        ),
        backup_fetcher=lambda: (
            state.__setitem__("backup_called", True)
            or BaseFetcher.build_live_payload(
                value=1.1,
                source="backup",
                source_priority=2,
                raw={"change_pct": 1.1},
                data_as_of_date="2026-03-08",
            )
        ),
        strict=True,
        label="测试实时源",
        threshold=2.0,
        probe_backup_on_primary_success=True,
    )

    assert result["source"] == "primary"
    assert state["backup_called"] is True


def test_get_a_share_prices_by_codes_no_full_market_call():
    from otc_fund_quant.nav_estimator.data.fetcher.stock_fetcher import StockFetcher

    fetcher = StockFetcher()
    with patch.object(fetcher, "get_a_share_prices", side_effect=AssertionError("should not call full market")), \
         patch.object(fetcher, "_get_a_share_quote", side_effect=lambda code: {"name": code, "price": 1.0, "change_pct": 0.1, "volume": 100}):
        result = fetcher.get_a_share_prices_by_codes(["600000", "000001"])

    assert set(result.keys()) == {"600000", "000001"}


def test_get_hk_prices_by_codes_no_full_market_call():
    from otc_fund_quant.nav_estimator.data.fetcher.stock_fetcher import StockFetcher

    fetcher = StockFetcher()
    with patch.object(fetcher, "get_hk_prices", side_effect=AssertionError("should not call full market")), \
         patch.object(fetcher, "_get_hk_share_quote", side_effect=lambda code: {"name": code, "price": 1.0, "change_pct": 0.1, "volume": 100}):
        result = fetcher.get_hk_prices_by_codes(["00700"])

    assert set(result.keys()) == {"00700"}


def test_live_a_share_quote_can_fall_back_to_backup_source():
    from otc_fund_quant.nav_estimator.data.fetcher.stock_fetcher import StockFetcher

    fetcher = StockFetcher()
    with patch.object(fetcher, "_get_a_share_quote", side_effect=ValueError("primary down")), \
         patch.object(fetcher, "_build_sina_a_share_payload", return_value=(
             {"name": "A", "price": 10.0, "change_pct": 1.5, "volume": 1000.0},
             "2026-03-08",
         )):
        payload = fetcher.get_a_share_quote_live("600000", strict=True)

    assert payload["source"] == "sina_single_quote"
    assert payload["raw"]["change_pct"] == 1.5


def test_hk_index_live_uses_lightweight_primary_before_backup():
    from otc_fund_quant.nav_estimator.data.fetcher.index_fetcher import IndexFetcher

    fetcher = IndexFetcher()
    with patch.object(fetcher, "_get_hk_index_return_em_once", side_effect=ValueError("primary down")), \
         patch.object(fetcher, "get_hk_index_return", side_effect=AssertionError("should not call old retrying source")), \
         patch.object(fetcher, "_get_hk_index_return_sina_single", return_value=(1.2, "2026-03-08")):
        payload = fetcher.get_hk_index_return_live("HSI", strict=True)

    assert payload["source"] == "sina_index_single_quote"
    assert payload["value"] == 1.2


def test_hk_index_live_falls_back_when_primary_returns_nan():
    from otc_fund_quant.nav_estimator.data.fetcher.index_fetcher import IndexFetcher

    fetcher = IndexFetcher()
    fake_hk_df = pd.DataFrame([{"代码": "HSTECH", "涨跌幅": float("nan")}])

    with patch("otc_fund_quant.nav_estimator.data.fetcher.index_fetcher.ak.stock_hk_index_spot_em", return_value=fake_hk_df), \
         patch.object(fetcher, "_get_hk_index_return_sina_single", return_value=(1.23, "2026-03-10")):
        payload = fetcher.get_hk_index_return_live("HSTECH", strict=True)

    assert payload["source"] == "sina_index_single_quote"
    assert payload["value"] == 1.23


def test_fund_classifier_short_circuits_tracking_target_calibration_without_lookup():
    classifier = FundClassifier()

    fund_info = {
        "code": "018345",
        "name": "华夏中证机器人交易型开放式指数证券投资基金发起式联接基金",
        "type": "股票型-标准指数",
        "benchmark": "中证机器人指数收益率×95%＋人民币活期存款税后利率×5%",
    }

    with patch.object(classifier.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=AssertionError("should not lookup benchmark index")):
        assert classifier.classify("018345", fund_info=fund_info) == "index_a"


def test_fund_classifier_short_circuits_tracking_target_calibration_for_semiconductor_feeder():
    classifier = FundClassifier()

    fund_info = {
        "code": "020640",
        "name": "广发中证半导体材料设备主题交易型开放式指数证券投资基金发起式联接基金",
        "type": "股票型-标准指数",
        "benchmark": "中证半导体材料设备主题指数收益率×95%+人民币活期存款税后利率×5%",
    }

    with patch.object(classifier.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=AssertionError("should not lookup benchmark index")):
        assert classifier.classify("020640", fund_info=fund_info) == "index_a"


@pytest.mark.parametrize(
    ("fund_code", "fund_name", "benchmark_text"),
    [
        ("012733", "易方达中证人工智能主题ETF联接A", "中证人工智能主题指数收益率×95%+银行活期存款利率(税后)×5%"),
        ("012734", "易方达中证人工智能主题ETF联接C", "中证人工智能主题指数收益率×95%+银行活期存款利率(税后)×5%"),
        ("007467", "华泰柏瑞中证红利低波动交易型开放式指数证券投资基金联接基金", "中证红利低波动指数收益率×95%+银行活期存款利率(税后)×5%"),
        ("023598", "景顺长城中证港股通创新药交易型开放式指数证券投资基金发起式联接基金", "中证港股通创新药指数收益率×95%+银行活期存款利率(税后)×5%"),
        ("024663", "富国创业板人工智能交易型开放式指数证券投资基金发起式联接基金", "创业板人工智能指数收益率×95%+银行活期存款利率(税后)×5%"),
    ],
)
def test_fund_classifier_short_circuits_tracking_target_calibration_for_additional_feeders(
    fund_code,
    fund_name,
    benchmark_text,
):
    classifier = FundClassifier()

    fund_info = {
        "code": fund_code,
        "name": fund_name,
        "type": "股票型-标准指数",
        "benchmark": benchmark_text,
    }

    with patch.object(classifier.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=AssertionError("should not lookup benchmark index")):
        assert classifier.classify(fund_code, fund_info=fund_info) == "index_a"


def test_a_index_live_standardizes_unsupported_index_code_error():
    from otc_fund_quant.nav_estimator.data.fetcher.index_fetcher import IndexFetcher

    fetcher = IndexFetcher()
    with patch.object(fetcher, "_get_a_index_live_coverage_snapshot", return_value={"codes": ["sh930001"]}), \
         patch.object(fetcher, "_get_a_index_return_sina_single", side_effect=ValueError("新浪指数实时行情为空: sh930001")), \
         patch.object(fetcher, "_get_a_index_return_em", side_effect=ValueError("未找到A股指数 930001")):
        with pytest.raises(ValueError, match="A股指数 930001 实时行情源不支持该指数代码"):
            fetcher.get_a_index_return_live("930001", strict=True)


def test_a_index_live_unsupported_error_uses_short_negative_cache():
    from otc_fund_quant.nav_estimator.data.fetcher.index_fetcher import IndexFetcher

    fetcher = IndexFetcher()
    cache_key = fetcher._build_a_index_live_unsupported_cache_key("930001")
    BaseFetcher.cache.delete(cache_key)
    state = {"primary": 0, "backup": 0}

    def primary_side_effect(_index_code):
        state["primary"] += 1
        raise ValueError("新浪指数实时行情为空: sh930001")

    def backup_side_effect(_index_code):
        state["backup"] += 1
        raise ValueError("未找到A股指数 930001")

    with patch.object(fetcher, "_get_a_index_live_coverage_snapshot", return_value={"codes": ["sh930001"]}), \
         patch.object(fetcher, "_get_a_index_return_sina_single", side_effect=primary_side_effect), \
         patch.object(fetcher, "_get_a_index_return_em", side_effect=backup_side_effect):
        for _ in range(2):
            with pytest.raises(ValueError, match="A股指数 930001 实时行情源不支持该指数代码"):
                fetcher.get_a_index_return_live("930001", strict=True)

    assert state == {"primary": 1, "backup": 1}


def test_a_index_live_coverage_snapshot_uses_short_cache():
    from otc_fund_quant.nav_estimator.data.fetcher.index_fetcher import IndexFetcher

    fetcher = IndexFetcher()
    BaseFetcher.cache.delete(fetcher._build_a_index_live_coverage_cache_key())
    fake_sina_df = pd.DataFrame([{"代码": "sh000300"}, {"代码": "sz399006"}])
    fake_catalog_snapshot = {
        "records": [
            {"code": "930001", "name": "中证机器人指数", "normalized_name": "中证机器人", "source_symbol": "中证系列指数"},
            {"code": "930999", "name": "恒生A股电网设备指数", "normalized_name": "恒生a股电网设备", "source_symbol": "中证系列指数"},
        ]
    }

    with patch("otc_fund_quant.nav_estimator.data.fetcher.index_fetcher.ak.stock_zh_index_spot_sina", return_value=fake_sina_df) as mock_sina, \
         patch.object(FundFetcher, "get_shared_a_index_catalog_snapshot", return_value=fake_catalog_snapshot):
        first_snapshot = fetcher._get_a_index_live_coverage_snapshot()
        second_snapshot = fetcher._get_a_index_live_coverage_snapshot()

    assert first_snapshot == second_snapshot
    assert {"sh000300", "sz399006", "sh930001", "sh930999"}.issubset(set(first_snapshot["codes"]))
    assert mock_sina.call_count == 1


def test_a_index_live_coverage_snapshot_short_circuits_unsupported_code():
    from otc_fund_quant.nav_estimator.data.fetcher.index_fetcher import IndexFetcher

    fetcher = IndexFetcher()
    BaseFetcher.cache.delete(fetcher._build_a_index_live_unsupported_cache_key("930001"))

    with patch.object(fetcher, "_get_a_index_live_coverage_snapshot", return_value={"codes": ["sh000300", "sz399006"]}), \
         patch.object(fetcher, "_get_a_index_return_sina_single", side_effect=AssertionError("should not probe primary")), \
         patch.object(fetcher, "_get_a_index_return_em", side_effect=AssertionError("should not probe backup")):
        with pytest.raises(ValueError, match="A股指数 930001 实时行情源不支持该指数代码"):
            fetcher.get_a_index_return_live("930001", strict=True)


def test_qdii_market_profile_supports_hk_us_and_rejects_global():
    fetcher = FundFetcher()

    assert fetcher.resolve_qdii_market_profile({"benchmark": "恒生指数收益率*50%+标普500收益率*50%"}) == "hk_us_mixed"
    assert fetcher.resolve_qdii_market_profile({"benchmark": "标普500指数收益率"}) == "us"
    assert fetcher.resolve_qdii_market_profile({"benchmark": "恒生科技指数收益率"}) == "hk"
    assert fetcher.resolve_qdii_market_profile({"benchmark": "全球市场收益率"}) == "unsupported"


def test_qdii_proxy_components_fall_back_to_baskets():
    fetcher = FundFetcher()

    hk_components = fetcher.resolve_qdii_proxy_components({"benchmark": "港股市场表现"})
    us_components = fetcher.resolve_qdii_proxy_components({"benchmark": "美国市场表现"})
    mixed_components = fetcher.resolve_qdii_proxy_components({"benchmark": "香港及美国市场表现"})

    assert {item["code"] for item in hk_components} == {"HSI", "HSTECH"}
    assert {item["code"] for item in us_components} == {"SPX", "NDX"}
    assert {item["code"] for item in mixed_components} == {"HSI", "HSTECH", "SPX", "NDX"}


def test_qdii_proxy_components_support_explicit_us_indices():
    fetcher = FundFetcher()
    components = fetcher.resolve_qdii_proxy_components({"benchmark": "标普500收益率*60%+纳斯达克100收益率*40%"})

    assert components[0]["code"] == "SPX"
    assert round(components[0]["weight"], 2) == 0.60
    assert components[1]["code"] == "NDX"
    assert round(components[1]["weight"], 2) == 0.40


def test_partial_quote_helpers_return_prices_and_missing_codes():
    from otc_fund_quant.nav_estimator.data.fetcher.stock_fetcher import StockFetcher

    fetcher = StockFetcher()
    def quote_side_effect(code):
        if code == "600000":
            return {"name": "a", "price": 1, "change_pct": 0.1, "volume": 1}
        raise RuntimeError("boom")

    with patch.object(fetcher, "_get_a_share_quote", side_effect=quote_side_effect):
        prices, missing = fetcher._get_a_share_prices_by_codes_partial(["600000", "000001"])

    assert set(prices.keys()) == {"600000"}
    assert missing == ["000001"]


def test_parallel_quote_workers_inherit_parent_request_state(monkeypatch):
    from otc_fund_quant.nav_estimator.data.fetcher.stock_fetcher import StockFetcher

    fetcher = StockFetcher()
    seen_states = []

    def quote_side_effect(code):
        seen_states.append((BaseFetcher._is_direct_mode_latched(), BaseFetcher._get_strict_mode_latched()))
        return {"name": code, "price": 1, "change_pct": 0.1, "volume": 1}

    with patch.object(fetcher, "_get_a_share_quote", side_effect=quote_side_effect):
        with BaseFetcher.request_scope(strict=False):
            BaseFetcher._set_direct_mode_latched(True)
            prices, missing = fetcher._get_a_share_prices_by_codes_partial(["600000", "000001"])

    assert missing == []
    assert list(prices.keys()) == ["600000", "000001"]
    assert all(item == (True, False) for item in seen_states)


def test_partial_quote_helpers_preserve_input_order():
    from otc_fund_quant.nav_estimator.data.fetcher.stock_fetcher import StockFetcher

    fetcher = StockFetcher()

    with patch.object(fetcher, "_get_a_share_quote", side_effect=lambda code: {"name": code, "price": 1, "change_pct": 0.1, "volume": 1}):
        prices, missing = fetcher._get_a_share_prices_by_codes_partial(["600000", "000001", "600000"])

    assert list(prices.keys()) == ["600000", "000001"]
    assert missing == []


def test_build_batch_context_non_strict_skips_expensive_prefetch():
    from otc_fund_quant.nav_estimator.core.nav_engine import NAVEngine

    engine = NAVEngine(strict=False)
    preloaded_inputs = [
        {
            "fund_code": "007343",
            "fund_type": "active_a",
            "fund_info": {"code": "007343", "name": "test", "type": "混合型-偏股", "benchmark": "沪深300"},
            "portfolio_holdings": pd.DataFrame([{"code": "600000", "market": "A股"}]),
            "active_proxy_components": [{"code": "000300", "market": "A股", "weight": 1.0}],
        }
    ]

    with patch.object(engine.stock_fetcher, "get_a_share_prices_by_codes", side_effect=AssertionError("should skip prefetch")), \
         patch.object(engine.stock_fetcher, "get_hk_prices_by_codes", side_effect=AssertionError("should skip prefetch")), \
         patch.object(engine.index_fetcher, "get_a_index_return", side_effect=AssertionError("should skip prefetch")), \
         patch.object(engine.index_fetcher, "get_hk_index_return", side_effect=AssertionError("should skip prefetch")), \
         patch.object(engine.bond_fetcher, "get_bond_index_snapshot", side_effect=AssertionError("should skip prefetch")), \
         patch.object(engine.fx_fetcher, "get_hkd_cny_daily_change", return_value=0.0):
        context = engine._build_batch_context(preloaded_inputs, strict=False)

    assert context.a_prices == {}
    assert context.hk_prices == {}
    assert context.a_index_returns == {}
    assert context.hk_index_returns == {}


def test_build_batch_context_strict_skips_all_realtime_prefetch():
    from otc_fund_quant.nav_estimator.core.nav_engine import NAVEngine

    engine = NAVEngine(strict=True)
    preloaded_inputs = [
        {
            "fund_code": "007343",
            "fund_type": "active_a",
            "fund_info": {"code": "007343", "name": "test", "type": "混合型-偏股", "benchmark": "沪深300"},
            "portfolio_holdings": pd.DataFrame([{"code": "600000", "market": "A股"}]),
            "active_proxy_components": [{"code": "000300", "market": "A股", "weight": 1.0}],
        }
    ]

    with patch.object(engine.stock_fetcher, "get_a_share_prices_by_codes", side_effect=AssertionError("strict should not prefetch a quotes")), \
         patch.object(engine.stock_fetcher, "get_hk_prices_by_codes", side_effect=AssertionError("strict should not prefetch hk quotes")), \
         patch.object(engine.index_fetcher, "get_a_index_return", side_effect=AssertionError("strict should not prefetch a index")), \
         patch.object(engine.index_fetcher, "get_hk_index_return", side_effect=AssertionError("strict should not prefetch hk index")), \
         patch.object(engine.bond_fetcher, "get_bond_index_snapshot", side_effect=AssertionError("strict should not prefetch bond index")), \
         patch.object(engine.fx_fetcher, "get_hkd_cny_daily_change", side_effect=AssertionError("strict should not prefetch fx")):
        context = engine._build_batch_context(preloaded_inputs, strict=True)

    assert context.a_prices == {}
    assert context.hk_prices == {}
    assert context.a_index_returns == {}
    assert context.hk_index_returns == {}
    assert context.bond_index_returns == {}
    assert context.hkd_cny_daily_change is None


def test_nav_engine_accepts_us_qdii_profile():
    from otc_fund_quant.nav_estimator.core.nav_engine import NAVEngine

    engine = NAVEngine(strict=True)
    profile = engine._validate_supported_qdii_profile(
        "110011",
        {
            "code": "110011",
            "name": "test",
            "type": "QDII-混合",
            "benchmark": "标普500收益率",
            "investment_strategy": "",
            "investment_target": "",
        },
    )

    assert profile == "us"


def test_nav_engine_rejects_unsupported_qdii_profile():
    from otc_fund_quant.nav_estimator.core.nav_engine import NAVEngine, UnsupportedFundError

    engine = NAVEngine(strict=True)

    with pytest.raises(UnsupportedFundError):
        engine._validate_supported_qdii_profile(
            "110011",
            {
                "code": "110011",
                "name": "test",
                "type": "QDII-混合",
                "benchmark": "全球市场收益率",
                "investment_strategy": "",
                "investment_target": "",
            },
        )


def test_active_equity_non_strict_never_calls_full_market_fallback():
    estimator = ActiveEquityEstimator()
    holdings_df = _active_holdings()
    context = _active_batch_context_with_errors()

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", side_effect=ValueError("未获取到资产配置数据")), \
         patch.object(estimator.fund_fetcher, "extract_holdings_report_metadata", return_value=("2025Q4", "2025-12-31")), \
         patch.object(estimator.stock_fetcher, "_get_a_share_prices_by_codes_partial", return_value=({}, ["600000", "000001"])), \
         patch.object(estimator.stock_fetcher, "get_a_share_prices", side_effect=AssertionError("should not call full market")), \
         patch.object(estimator.index_fetcher, "get_a_index_return", return_value=0.6), \
         patch.object(estimator.fx_fetcher, "get_hkd_cny_daily_change", return_value=0.0):
        result = estimator.estimate(
            fund_code="007343",
            last_nav=3.0,
            nav_date="2026-03-07",
            target_date="2026-03-08",
            holdings=holdings_df,
            strict=False,
            batch_context=context,
        )

    assert result["estimated_nav"] > 0
    assert set(result["missing_quotes"]) == {"600000", "000001"}


def test_qdii_non_strict_never_calls_full_market_fallback():
    estimator = QDIIHKEstimator()
    holdings_df = pd.DataFrame(
        [
            {"code": "00700", "name": "Tencent", "weight": 10.0, "market": "港股"},
            {"code": "00941", "name": "CM", "weight": 8.0, "market": "港股"},
        ]
    )
    context = _qdii_batch_context_with_errors()

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", side_effect=ValueError("未获取到资产配置数据")), \
         patch.object(estimator.fund_fetcher, "extract_holdings_report_metadata", return_value=("2025Q4", "2025-12-31")), \
         patch.object(estimator.stock_fetcher, "_get_hk_prices_by_codes_partial", return_value=({}, ["00700", "00941"])), \
         patch.object(estimator.stock_fetcher, "get_hk_prices", side_effect=AssertionError("should not call full market")), \
         patch.object(estimator.index_fetcher, "get_hk_index_return", side_effect=RuntimeError("proxy")), \
         patch.object(estimator.fx_fetcher, "get_hkd_cny_daily_change", return_value=0.0):
        result = estimator.estimate(
            fund_code="123456",
            last_nav=1.0,
            nav_date="2026-03-07",
            target_date="2026-03-08",
            is_index_fund=False,
            holdings=holdings_df,
            strict=False,
            batch_context=context,
        )

    assert result["estimated_nav"] > 0
    assert set(result["missing_quotes"]) == {"00700", "00941"}


def test_lookup_a_index_code_by_name_matches_normalized_name():
    fetcher = FundFetcher()
    snapshot = {
        "records": [
            {"code": "930001", "name": "测试规范化查码指数", "normalized_name": "测试规范化查码", "source_symbol": "中证系列指数"},
            {"code": "000906", "name": "中证800", "normalized_name": "中证800", "source_symbol": "沪深重要指数"},
        ]
    }

    with patch.object(FundFetcher, "get_shared_a_index_catalog_snapshot", return_value=snapshot):
        resolved = fetcher._lookup_a_index_code_by_name("测试规范化查码")

    assert resolved["code"] == "930001"
    assert resolved["name"] == "测试规范化查码指数"


def test_a_index_catalog_snapshot_uses_shared_disk_cache():
    fetcher = FundFetcher()
    BaseFetcher.cache.delete(fetcher._build_a_index_catalog_cache_key())
    fake_df = pd.DataFrame(
        [
            {"代码": f"{930000 + idx:06d}", "名称": f"测试目录指数{idx}"}
            for idx in range(FundFetcher.A_INDEX_CATALOG_MIN_RECORDS + 5)
        ]
    )

    with patch("otc_fund_quant.nav_estimator.data.fetcher.fund_fetcher.ak.stock_zh_index_spot_em", return_value=fake_df) as mock_spot:
        first_snapshot = fetcher.get_shared_a_index_catalog_snapshot()
        second_snapshot = fetcher.get_shared_a_index_catalog_snapshot()

    assert first_snapshot["records"] == second_snapshot["records"]
    assert mock_spot.call_count == len(fetcher.A_INDEX_SPOT_SYMBOL_CANDIDATES)
    assert len(first_snapshot["records"]) >= FundFetcher.A_INDEX_CATALOG_MIN_RECORDS


def test_a_index_catalog_snapshot_ignores_invalid_small_cached_snapshot():
    fetcher = FundFetcher()
    cache_key = fetcher._build_a_index_catalog_cache_key()
    BaseFetcher.cache.set(
        cache_key,
        {
            "kind": "a_index_catalog_snapshot",
            "schema_version": FundFetcher.A_INDEX_CACHE_SCHEMA_VERSION,
            "refreshed_at": "2026-03-10T09:00:00",
            "records": [{"code": "930001", "name": "中证机器人指数", "normalized_name": "中证机器人", "source_symbol": "中证系列指数"}],
        },
        ttl=3600,
    )
    fake_df = pd.DataFrame(
        [
            {"代码": f"{930100 + idx:06d}", "名称": f"重建目录指数{idx}"}
            for idx in range(FundFetcher.A_INDEX_CATALOG_MIN_RECORDS + 3)
        ]
    )

    with patch("otc_fund_quant.nav_estimator.data.fetcher.fund_fetcher.ak.stock_zh_index_spot_em", return_value=fake_df) as mock_spot:
        snapshot = fetcher.get_shared_a_index_catalog_snapshot()

    assert len(snapshot["records"]) >= FundFetcher.A_INDEX_CATALOG_MIN_RECORDS
    assert all(record["name"].startswith("重建目录指数") for record in snapshot["records"][:3])
    assert mock_spot.call_count == len(fetcher.A_INDEX_SPOT_SYMBOL_CANDIDATES)


def test_lookup_a_index_code_by_name_prefers_calibration_over_legacy_positive_cache():
    fetcher = FundFetcher()
    legacy_cache_key = BaseFetcher._build_cache_key(FundFetcher._lookup_a_index_code_by_name.__wrapped__, (fetcher, "中证港股通综合"), {})
    BaseFetcher.cache.set(legacy_cache_key, {"code": "000008", "name": "综合指数"}, ttl=3600)

    with patch.object(FundFetcher, "get_shared_a_index_catalog_snapshot", side_effect=AssertionError("should not load catalog")):
        resolved = fetcher._lookup_a_index_code_by_name("中证港股通综合")

    assert resolved == {"code": "930930", "name": "中证港股通综合"}


def test_lookup_a_index_code_by_name_uses_alias_calibration_before_catalog():
    fetcher = FundFetcher()

    with patch.object(FundFetcher, "get_shared_a_index_catalog_snapshot", side_effect=AssertionError("should not load catalog")):
        resolved = fetcher._lookup_a_index_code_by_name("中证医药卫生")

    assert resolved == {"code": "000933", "name": "中证医药卫生"}


def test_resolve_a_index_code_supports_extended_aliases_and_dynamic_lookup():
    fetcher = FundFetcher()

    def lookup_side_effect(index_name):
        mapping = {
            "中证机器人": {"code": "930001", "name": "中证机器人指数"},
            "中证机器人指数": {"code": "930001", "name": "中证机器人指数"},
            "恒生A股电网设备": {"code": "930999", "name": "恒生A股电网设备指数"},
            "恒生A股电网设备指数": {"code": "930999", "name": "恒生A股电网设备指数"},
        }
        return mapping[index_name]

    with patch.object(fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect):
        assert fetcher.resolve_a_index_code({"benchmark": "中证800指数收益率×95%+存款利率×5%"}) == "000906"
        assert fetcher.resolve_a_index_code({"benchmark": "中证机器人指数收益率×95%+存款利率×5%"}) == "930001"
        assert fetcher.resolve_a_index_code({"benchmark": "恒生A股电网设备指数收益率×95%+存款利率×5%"}) == "930999"


def test_resolve_a_index_code_uses_benchmark_only():
    fetcher = FundFetcher()

    with pytest.raises(ValueError, match="未识别到任何A股权益指数名称"):
        fetcher.resolve_a_index_code(
            {
                "name": "华夏中证机器人交易型开放式指数证券投资基金发起式联接基金",
                "type": "股票型-标准指数",
                "benchmark": "",
                "investment_target": "本基金紧密跟踪标的指数，力争获得与指数相近的收益",
            }
        )


def test_resolve_hk_index_code_uses_benchmark_only():
    fetcher = FundFetcher()

    with pytest.raises(ValueError, match="未识别到任何港股权益指数名称"):
        fetcher.resolve_hk_index_code(
            {
                "name": "南方恒生科技交易型开放式指数证券投资基金发起式联接基金（QDII）",
                "type": "QDII-股票",
                "benchmark": "",
                "investment_target": "本基金紧密跟踪恒生科技指数，力争获得与指数相近的收益",
            }
        )


def test_resolve_index_tracking_target_uses_linked_etf_calibration_for_feeder_funds():
    fetcher = FundFetcher()

    ai_linked_a_target = fetcher.resolve_index_tracking_target(
        {
            "code": "012733",
            "name": "易方达中证人工智能主题ETF联接A",
            "type": "股票型-标准指数",
            "benchmark": "中证人工智能主题指数收益率×95%+银行活期存款利率(税后)×5%",
        }
    )
    ai_linked_c_target = fetcher.resolve_index_tracking_target(
        {
            "code": "012734",
            "name": "易方达中证人工智能主题ETF联接C",
            "type": "股票型-标准指数",
            "benchmark": "中证人工智能主题指数收益率×95%+银行活期存款利率(税后)×5%",
        }
    )
    low_vol_target = fetcher.resolve_index_tracking_target(
        {
            "code": "007467",
            "name": "华泰柏瑞中证红利低波动交易型开放式指数证券投资基金联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证红利低波动指数收益率×95%+银行活期存款利率(税后)×5%",
        }
    )
    semiconductor_target = fetcher.resolve_index_tracking_target(
        {
            "code": "020640",
            "name": "广发中证半导体材料设备主题交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证半导体材料设备主题指数收益率×95%+人民币活期存款税后利率×5%",
        }
    )
    robot_target = fetcher.resolve_index_tracking_target(
        {
            "code": "018345",
            "name": "华夏中证机器人交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证机器人指数收益率×95%＋人民币活期存款税后利率×5%",
        }
    )
    power_grid_target = fetcher.resolve_index_tracking_target(
        {
            "code": "023639",
            "name": "国泰恒生A股电网设备交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "恒生A股电网设备指数收益率*95%+银行活期存款利率(税后)*5%",
        }
    )
    innovation_drug_target = fetcher.resolve_index_tracking_target(
        {
            "code": "023598",
            "name": "景顺长城中证港股通创新药交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证港股通创新药指数收益率×95%+银行活期存款利率(税后)×5%",
        }
    )
    ai_target = fetcher.resolve_index_tracking_target(
        {
            "code": "024663",
            "name": "富国创业板人工智能交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "创业板人工智能指数收益率×95%+银行活期存款利率(税后)×5%",
        }
    )
    vanilla_target = fetcher.resolve_index_tracking_target(
        {
            "code": "018291",
            "name": "广发新兴成长灵活配置混合型证券投资基金",
            "type": "混合型-灵活配置",
            "benchmark": "中证800指数收益率×65%+一年期人民币定期存款利率（税后）×35%",
        }
    )

    assert ai_linked_a_target == {
        "target_type": "linked_etf_a_share",
        "security_code": "159819",
        "market": "A股",
        "tracking_name": "人工智能ETF",
    }
    assert ai_linked_c_target == {
        "target_type": "linked_etf_a_share",
        "security_code": "159819",
        "market": "A股",
        "tracking_name": "人工智能ETF",
    }
    assert low_vol_target == {
        "target_type": "linked_etf_a_share",
        "security_code": "512890",
        "market": "A股",
        "tracking_name": "红利低波ETF",
    }
    assert semiconductor_target == {
        "target_type": "linked_etf_a_share",
        "security_code": "560780",
        "market": "A股",
        "tracking_name": "广发中证半导体材料设备主题ETF",
    }
    assert robot_target == {
        "target_type": "linked_etf_a_share",
        "security_code": "562500",
        "market": "A股",
        "tracking_name": "中证机器人ETF",
    }
    assert power_grid_target == {
        "target_type": "linked_etf_a_share",
        "security_code": "560880",
        "market": "A股",
        "tracking_name": "恒生A股电网设备ETF",
    }
    assert innovation_drug_target == {
        "target_type": "linked_etf_a_share",
        "security_code": "513780",
        "market": "A股",
        "tracking_name": "港股创新药ETF",
    }
    assert ai_target == {
        "target_type": "linked_etf_a_share",
        "security_code": "159246",
        "market": "A股",
        "tracking_name": "创业板人工智能ETF富国",
    }
    assert vanilla_target["target_type"] == "a_index"
    assert vanilla_target["code"] == "000906"


def test_lookup_a_index_code_by_name_uses_negative_cache_for_ambiguous_names():
    fetcher = FundFetcher()
    index_name = "测试歧义指数性能用例"
    BaseFetcher.cache.delete(fetcher._build_a_index_lookup_negative_cache_key(index_name))
    snapshot = {
        "records": [
            {"code": "000001", "name": index_name, "normalized_name": index_name, "source_symbol": "沪深重要指数"},
            {"code": "399001", "name": index_name, "normalized_name": index_name, "source_symbol": "深证系列指数"},
        ]
    }

    with patch.object(FundFetcher, "get_shared_a_index_catalog_snapshot", return_value=snapshot) as mock_snapshot:
        for _ in range(2):
            with pytest.raises(ValueError, match="映射到多个A股指数代码"):
                fetcher._lookup_a_index_code_by_name(index_name)

    assert mock_snapshot.call_count == 1


def test_lookup_a_index_code_by_name_uses_negative_cache_for_unmapped_names():
    fetcher = FundFetcher()
    index_name = "测试未映射指数性能用例"
    BaseFetcher.cache.delete(fetcher._build_a_index_lookup_negative_cache_key(index_name))
    snapshot = {
        "records": [
            {"code": "000906", "name": "中证800", "normalized_name": "中证800", "source_symbol": "沪深重要指数"},
            {"code": "000300", "name": "沪深300", "normalized_name": "沪深300", "source_symbol": "沪深重要指数"},
        ]
    }

    with patch.object(FundFetcher, "get_shared_a_index_catalog_snapshot", return_value=snapshot) as mock_snapshot:
        for _ in range(2):
            with pytest.raises(ValueError, match="无法映射到A股指数代码"):
                fetcher._lookup_a_index_code_by_name(index_name)

    assert mock_snapshot.call_count == 1


def test_resolve_active_proxy_components_supports_a_share_etf_proxy_calibration():
    fetcher = FundFetcher()

    components = fetcher.resolve_active_proxy_components(
        {
            "code": "025209",
            "benchmark": "中证全指半导体产品与设备指数收益率×70%+恒生指数收益率×10%+中债-综合指数（全价）收益率×20%",
        }
    )

    assert components == [
        {
            "code": "512480",
            "name": "半导体ETF",
            "market": "A股",
            "weight": 0.875,
            "target_type": "a_share_etf_proxy",
            "quote_code": "512480",
        },
        {
            "code": "HSI",
            "name": "恒生指数",
            "market": "港股",
            "weight": 0.125,
            "target_type": "index",
            "quote_code": "HSI",
        },
    ]


@pytest.mark.parametrize(
    ("fund_code", "benchmark_text", "expected_component"),
    [
        (
            "900001",
            "中证红利低波动指数收益率×95%+中债-综合指数（全价）收益率×5%",
            {
                "code": "512890",
                "name": "红利低波ETF",
                "market": "A股",
                "weight": 1.0,
                "target_type": "a_share_etf_proxy",
                "quote_code": "512890",
            },
        ),
        (
            "900002",
            "中证港股通创新药指数收益率×95%+银行活期存款利率(税后)×5%",
            {
                "code": "513780",
                "name": "港股创新药ETF",
                "market": "A股",
                "weight": 1.0,
                "target_type": "a_share_etf_proxy",
                "quote_code": "513780",
            },
        ),
        (
            "900003",
            "创业板人工智能指数收益率×95%+银行活期存款利率(税后)×5%",
            {
                "code": "159246",
                "name": "创业板人工智能ETF富国",
                "market": "A股",
                "weight": 1.0,
                "target_type": "a_share_etf_proxy",
                "quote_code": "159246",
            },
        ),
        (
            "900004",
            "中证人工智能主题指数收益率×80%+银行活期存款利率(税后)×20%",
            {
                "code": "159819",
                "name": "人工智能ETF",
                "market": "A股",
                "weight": 1.0,
                "target_type": "a_share_etf_proxy",
                "quote_code": "159819",
            },
        ),
    ],
)
def test_resolve_active_proxy_components_supports_additional_a_share_etf_proxy_calibrations(
    fund_code,
    benchmark_text,
    expected_component,
):
    fetcher = FundFetcher()

    components = fetcher.resolve_active_proxy_components(
        {
            "code": fund_code,
            "benchmark": benchmark_text,
        }
    )

    assert components == [expected_component]


def test_active_equity_estimator_strict_supports_ai_theme_a_share_etf_proxy_component():
    estimator = ActiveEquityEstimator()
    holdings_df = pd.DataFrame([{"code": "600000", "name": "A", "weight": 10.0, "market": "A股"}])
    proxy_components = [
        {"code": "159819", "quote_code": "159819", "name": "人工智能ETF", "market": "A股", "weight": 1.0, "target_type": "a_share_etf_proxy"},
    ]

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", return_value={
        "position": 80.0,
        "report_date": "2025-12-31",
        "raw_report_period": "2025-12-31",
        "snapshot_source": "pingzhongdata_asset_allocation",
    }), \
         patch.object(estimator.fund_fetcher, "extract_holdings_report_metadata", return_value=("2025Q4", "2025-12-31")), \
         patch.object(estimator.stock_fetcher, "_get_a_share_quotes_live_by_codes_partial", return_value=({
             "600000": {
                 "value": 1.0,
                 "source": "eastmoney_single_quote",
                 "source_disagreements": [],
                 "data_as_of_date": "2026-03-13",
                 "raw": {"change_pct": 1.0},
             }
         }, [])), \
         patch.object(estimator.stock_fetcher, "_get_hk_share_quotes_live_by_codes_partial", return_value=({}, [])), \
         patch.object(estimator.stock_fetcher, "get_a_share_quote_live", return_value={
             "value": -1.55,
             "source": "eastmoney_single_quote",
             "source_disagreements": [],
             "data_as_of_date": "2026-03-13",
             "raw": {"change_pct": -1.55},
         }) as mock_etf_quote, \
         patch.object(estimator.fx_fetcher, "get_hkd_cny_daily_change_live", return_value={
             "value": 0.0,
             "source": "safe_mid_rate",
             "source_disagreements": [],
             "data_as_of_date": "2026-03-13",
             "raw": {"change_pct": 0.0},
         }), \
         patch.object(estimator.index_fetcher, "get_a_index_return_live", side_effect=AssertionError("should not use a-index fetcher for AI ETF proxy")):
        result = estimator.estimate(
            fund_code="017811",
            last_nav=1.8044,
            nav_date="2026-03-12",
            target_date="2026-03-13",
            holdings=holdings_df,
            proxy_components=proxy_components,
            strict=True,
        )

    mock_etf_quote.assert_called_once_with("159819", strict=True)
    assert result["estimated_nav"] > 0
    assert result["quality_gate_passed"] is True
    assert result["proxy_components"][0]["target_type"] == "a_share_etf_proxy"
    assert result["proxy_components"][0]["quote_code"] == "159819"
    assert result["used_sources"]["proxy_tracking_targets"]["159819"] == "eastmoney_single_quote"


def test_nav_engine_strict_run_supports_active_ai_theme_proxy_benchmark():
    engine = NAVEngine(strict=True)
    holdings_df = pd.DataFrame([{"code": "600000", "name": "A", "weight": 10.0, "market": "A股"}])
    fund_info = {
        "code": "017811",
        "name": "东方人工智能主题混合型证券投资基金",
        "type": "混合型-偏股",
        "benchmark": "中证人工智能主题指数收益率×80%+银行活期存款利率(税后)×20%",
        "management_fee": 0.015,
        "custody_fee": 0.002,
    }

    def estimate_success(**kwargs):
        return {
            "fund_code": kwargs["fund_code"],
            "estimated_nav": 1.79,
            "estimated_return": -0.8,
            "nav_date": kwargs["nav_date"],
            "warnings": [],
        }

    with patch.object(engine.fund_fetcher, "get_fund_info", return_value=fund_info), \
         patch.object(engine.fund_fetcher, "get_previous_official_nav", return_value=(1.8044, "2026-03-12")), \
         patch.object(engine.fund_fetcher, "get_portfolio_holdings", return_value=holdings_df), \
         patch.object(engine.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=AssertionError("should not lookup AI theme index dynamically")), \
         patch.object(engine.classifier.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=AssertionError("should not lookup AI theme index dynamically")), \
         patch("otc_fund_quant.nav_estimator.core.nav_engine.build_confidence_payload", return_value={}), \
         patch.object(engine.equity_estimator, "estimate", side_effect=estimate_success) as mock_equity_estimate, \
         patch.object(engine.index_estimator, "estimate", side_effect=AssertionError("should not use index estimator")), \
         patch.object(engine.qdii_estimator, "estimate", side_effect=AssertionError("should not use qdii estimator")):
        results_df = engine.run(["017811"], strict=True)

    result_map = {row["fund_code"]: row for row in results_df.to_dict(orient="records")}
    assert result_map["017811"]["status"] == "成功"
    mock_equity_estimate.assert_called_once()
    call_kwargs = mock_equity_estimate.call_args.kwargs
    assert call_kwargs["fund_code"] == "017811"
    assert call_kwargs["proxy_components"] == [
        {
            "code": "159819",
            "name": "人工智能ETF",
            "market": "A股",
            "weight": 1.0,
            "target_type": "a_share_etf_proxy",
            "quote_code": "159819",
        }
    ]


def test_resolve_active_proxy_components_filters_non_equity_and_renormalizes_weights():
    fetcher = FundFetcher()

    def lookup_side_effect(index_name):
        mapping = {
            "中证医药卫生": {"code": "930100", "name": "中证医药卫生指数"},
            "中证医药卫生指数": {"code": "930100", "name": "中证医药卫生指数"},
            "中证港股通综合": {"code": "930200", "name": "中证港股通综合指数"},
            "中证港股通综合指数": {"code": "930200", "name": "中证港股通综合指数"},
        }
        return mapping[index_name]

    with patch.object(fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect):
        components = fetcher.resolve_active_proxy_components(
            {
                "code": "015916",
                "benchmark": "中证医药卫生指数收益率×70%+中证港股通综合指数收益率（人民币）×10%+中债-综合指数（全价）收益率×20%",
            }
        )

    assert [item["code"] for item in components] == ["930100", "930200"]
    assert round(components[0]["weight"], 3) == 0.875
    assert round(components[1]["weight"], 3) == 0.125


def test_extract_benchmark_equity_index_components_skips_explanatory_noise():
    fetcher = FundFetcher()
    noisy_texts = [
        "备选成份股来跟踪标的指数",
        "也可以通过买入标的指数",
        "追求跟踪标的指数",
        "获得与指数",
    ]

    with patch.object(fetcher, "_lookup_a_index_code_by_name", side_effect=AssertionError("should not lookup noise")):
        for text in noisy_texts:
            assert fetcher.extract_benchmark_equity_index_components(text, allowed_markets={"A股"}) == []


def test_resolve_active_proxy_components_ignores_explanatory_noise_candidates():
    fetcher = FundFetcher()

    def lookup_side_effect(index_name):
        mapping = {
            "中证医药卫生": {"code": "930100", "name": "中证医药卫生指数"},
            "中证医药卫生指数": {"code": "930100", "name": "中证医药卫生指数"},
            "中证港股通综合": {"code": "930200", "name": "中证港股通综合指数"},
            "中证港股通综合指数": {"code": "930200", "name": "中证港股通综合指数"},
        }
        return mapping[index_name]

    with patch.object(fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect) as mock_lookup:
        components = fetcher.resolve_active_proxy_components(
            {
                "code": "015916",
                "benchmark": "中证医药卫生指数收益率×70%+中证港股通综合指数收益率（人民币）×10%+中债-综合指数（全价）收益率×20%",
                "investment_target": "本基金也可以通过买入标的指数，追求跟踪标的指数，备选成份股来跟踪标的指数，获得与指数相近的收益",
            }
        )

    assert [item["code"] for item in components] == ["930100", "930200"]
    assert all(
        noisy_text not in {call.args[0] for call in mock_lookup.call_args_list}
        for noisy_text in {"备选成份股来跟踪标的指数", "也可以通过买入标的指数", "追求跟踪标的指数", "获得与指数"}
    )


def test_resolve_active_proxy_components_does_not_treat_bond_residue_as_equity_index():
    fetcher = FundFetcher()
    lookup_calls = []

    def lookup_side_effect(index_name):
        lookup_calls.append(index_name)
        mapping = {
            "中证医药卫生": {"code": "930100", "name": "中证医药卫生指数"},
            "中证医药卫生指数": {"code": "930100", "name": "中证医药卫生指数"},
            "中证港股通综合": {"code": "930200", "name": "中证港股通综合指数"},
            "中证港股通综合指数": {"code": "930200", "name": "中证港股通综合指数"},
        }
        return mapping[index_name]

    with patch.object(fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect):
        components = fetcher.resolve_active_proxy_components(
            {
                "code": "015916",
                "benchmark": "中证医药卫生指数收益率×70%+中证港股通综合指数收益率（人民币）×10%+中债-综合指数（全价）收益率×20%",
            }
        )

    assert [item["code"] for item in components] == ["930100", "930200"]
    assert "综合指数" not in lookup_calls


def test_extract_benchmark_equity_index_components_does_not_fallback_to_generic_comprehensive_index():
    fetcher = FundFetcher()
    lookup_calls = []

    def lookup_side_effect(index_name):
        lookup_calls.append(index_name)
        raise ValueError(f"已识别指数名称 {index_name}，但无法映射到A股指数代码")

    with patch.object(fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect):
        components = fetcher.extract_benchmark_equity_index_components(
            "中证医药卫生指数收益率×70%+中证港股通综合指数收益率（人民币）×10%+中债-综合指数（全价）收益率×20%",
            allowed_markets={"A股"},
        )

    assert components == []
    assert "综合指数" not in lookup_calls


def test_resolve_active_proxy_components_rejects_incomplete_weighted_benchmark():
    fetcher = FundFetcher()

    def lookup_side_effect(index_name):
        if index_name in {"中证港股通综合", "中证港股通综合指数"}:
            return {"code": "930200", "name": "中证港股通综合指数"}
        raise ValueError(f"已识别指数名称 {index_name}，但无法映射到A股指数代码")

    with patch.object(fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect):
        with pytest.raises(ValueError, match="未解析指数成分"):
            fetcher.resolve_active_proxy_components(
                {
                    "code": "015916",
                    "benchmark": "中证医药卫生指数收益率×70%+中证港股通综合指数收益率（人民币）×10%+中债-综合指数（全价）收益率×20%",
                }
            )


def test_resolve_active_proxy_components_supports_single_equity_index_with_cash_benchmark():
    fetcher = FundFetcher()

    components = fetcher.resolve_active_proxy_components(
        {
            "code": "018291",
            "benchmark": "中证800指数收益率×65%+一年期人民币定期存款利率（税后）×35%",
        }
    )

    assert len(components) == 1
    assert components[0]["code"] == "000906"
    assert components[0]["weight"] == 1.0


def test_resolve_active_proxy_components_rejects_non_equity_only_benchmark():
    fetcher = FundFetcher()

    with pytest.raises(ValueError, match="未识别到任何权益指数名称"):
        fetcher.resolve_active_proxy_components(
            {
                "code": "099999",
                "benchmark": "中债-综合指数收益率×80%+银行活期存款利率（税后）×20%",
            }
        )


def test_fund_classifier_prefers_a_share_index_when_hengsheng_a_share_present():
    classifier = FundClassifier()

    with patch.object(
        classifier.fund_fetcher,
        "_lookup_a_index_code_by_name",
        return_value={"code": "930999", "name": "恒生A股电网设备指数"},
    ):
        fund_type = classifier.classify(
            "023639",
            fund_info={
                "code": "023639",
                "name": "国泰恒生A股电网设备交易型开放式指数证券投资基金发起式联接基金",
                "type": "股票型-标准指数",
                "benchmark": "恒生A股电网设备指数收益率*95%+银行活期存款利率(税后)*5%",
            },
        )

    assert fund_type == "index_a"


def test_fund_classifier_ignores_explanatory_noise_in_benchmark_lookup():
    classifier = FundClassifier()

    def lookup_side_effect(index_name):
        mapping = {
            "恒生A股电网设备": {"code": "930999", "name": "恒生A股电网设备指数"},
            "恒生A股电网设备指数": {"code": "930999", "name": "恒生A股电网设备指数"},
        }
        return mapping[index_name]

    benchmark_text = "恒生A股电网设备指数收益率*95%+银行活期存款利率(税后)*5%+也可以通过买入标的指数"
    with patch.object(classifier.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect) as mock_lookup:
        fund_type = classifier.classify(
            "023639",
            fund_info={
                "code": "023639",
                "name": "国泰恒生A股电网设备交易型开放式指数证券投资基金发起式联接基金",
                "type": "股票型-标准指数",
                "benchmark": benchmark_text,
            },
        )

    assert fund_type == "index_a"
    lookup_inputs = {call.args[0] for call in mock_lookup.call_args_list}
    assert "也可以通过买入标的指数" not in lookup_inputs


def test_fund_classifier_keeps_qdii_priority():
    classifier = FundClassifier()

    fund_type = classifier.classify(
        "020989",
        fund_info={
            "code": "020989",
            "name": "南方恒生科技交易型开放式指数证券投资基金发起式联接基金（QDII）",
            "type": "QDII-股票",
            "benchmark": "经汇率调整后的恒生科技指数收益率×95%+银行人民币活期存款利率（税后）×5%",
        },
    )

    assert fund_type == "qdii"


def test_nav_engine_strict_run_avoids_prefetch_failures_for_supported_cases():
    engine = NAVEngine(strict=True)
    holdings_df = pd.DataFrame([{"code": "600000", "name": "A", "weight": 10.0, "market": "A股"}])
    fund_info_map = {
        "015916": {
            "code": "015916",
            "name": "永赢医药创新智选混合型发起式证券投资基金",
            "type": "混合型-偏股",
            "benchmark": "中证医药卫生指数收益率×70%+中证港股通综合指数收益率（人民币）×10%+中债-综合指数（全价）收益率×20%",
            "management_fee": 0.015,
            "custody_fee": 0.002,
        },
        "018291": {
            "code": "018291",
            "name": "广发新兴成长灵活配置混合型证券投资基金",
            "type": "混合型-灵活配置",
            "benchmark": "中证800指数收益率×65%+一年期人民币定期存款利率（税后）×35%",
            "management_fee": 0.015,
            "custody_fee": 0.002,
        },
        "018345": {
            "code": "018345",
            "name": "华夏中证机器人交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证机器人指数收益率×95%＋人民币活期存款税后利率×5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
        "020989": {
            "code": "020989",
            "name": "南方恒生科技交易型开放式指数证券投资基金发起式联接基金（QDII）",
            "type": "QDII-股票",
            "benchmark": "经汇率调整后的恒生科技指数收益率×95%+银行人民币活期存款利率（税后）×5%",
            "management_fee": 0.015,
            "custody_fee": 0.003,
            "investment_strategy": "",
            "investment_target": "",
        },
        "023639": {
            "code": "023639",
            "name": "国泰恒生A股电网设备交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "恒生A股电网设备指数收益率*95%+银行活期存款利率(税后)*5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
    }

    def lookup_side_effect(index_name):
        mapping = {
            "中证医药卫生": {"code": "930100", "name": "中证医药卫生指数"},
            "中证医药卫生指数": {"code": "930100", "name": "中证医药卫生指数"},
            "中证港股通综合": {"code": "930200", "name": "中证港股通综合指数"},
            "中证港股通综合指数": {"code": "930200", "name": "中证港股通综合指数"},
            "中证机器人": {"code": "930001", "name": "中证机器人指数"},
            "中证机器人指数": {"code": "930001", "name": "中证机器人指数"},
            "恒生A股电网设备": {"code": "930999", "name": "恒生A股电网设备指数"},
            "恒生A股电网设备指数": {"code": "930999", "name": "恒生A股电网设备指数"},
        }
        return mapping[index_name]

    def estimate_success(**kwargs):
        return {
            "fund_code": kwargs["fund_code"],
            "estimated_nav": 1.01,
            "estimated_return": 0.1,
            "nav_date": kwargs["nav_date"],
            "warnings": [],
        }

    with patch.object(engine.fund_fetcher, "get_fund_info", side_effect=lambda code: fund_info_map[code]), \
         patch.object(engine.fund_fetcher, "get_previous_official_nav", return_value=(1.0, "2026-03-07")), \
         patch.object(engine.fund_fetcher, "get_portfolio_holdings", return_value=holdings_df), \
         patch.object(engine.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect), \
         patch.object(engine.classifier.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect), \
         patch("otc_fund_quant.nav_estimator.core.nav_engine.build_confidence_payload", return_value={}), \
         patch.object(engine.equity_estimator, "estimate", side_effect=estimate_success) as mock_equity_estimate, \
         patch.object(engine.index_estimator, "estimate", side_effect=estimate_success) as mock_index_estimate, \
         patch.object(engine.qdii_estimator, "estimate", side_effect=estimate_success) as mock_qdii_estimate:
        results_df = engine.run(["015916", "018291", "018345", "020989", "023639"], strict=True)

    assert list(results_df["fund_code"]) == ["015916", "018291", "018345", "020989", "023639"]
    assert set(results_df["status"]) == {"成功"}
    result_map = {row["fund_code"]: row["fund_type"] for row in results_df.to_dict(orient="records")}
    assert result_map["015916"] == "active_a"
    assert result_map["018291"] == "active_a"
    assert result_map["018345"] == "index_a"
    assert result_map["020989"] == "qdii"
    assert result_map["023639"] == "index_a"
    assert {call.kwargs["fund_code"] for call in mock_equity_estimate.call_args_list} == {"015916", "018291"}
    assert {call.kwargs["fund_code"] for call in mock_index_estimate.call_args_list} == {"018345", "023639"}
    assert {call.kwargs["fund_code"] for call in mock_qdii_estimate.call_args_list} == {"020989"}
    index_kwargs = {call.kwargs["fund_code"]: call.kwargs for call in mock_index_estimate.call_args_list}
    assert index_kwargs["018345"]["index_code"] == "562500"
    assert index_kwargs["018345"]["tracking_target"] == {
        "target_type": "linked_etf_a_share",
        "security_code": "562500",
        "market": "A股",
        "tracking_name": "中证机器人ETF",
    }
    assert index_kwargs["023639"]["index_code"] == "560880"
    assert index_kwargs["023639"]["tracking_target"] == {
        "target_type": "linked_etf_a_share",
        "security_code": "560880",
        "market": "A股",
        "tracking_name": "恒生A股电网设备ETF",
    }


def test_nav_engine_strict_run_uses_tracking_calibration_without_noise_lookups():
    engine = NAVEngine(strict=True)
    holdings_df = pd.DataFrame([{"code": "600000", "name": "A", "weight": 10.0, "market": "A股"}])
    fund_info_map = {
        "015916": {
            "code": "015916",
            "name": "永赢医药创新智选混合型发起式证券投资基金",
            "type": "混合型-偏股",
            "benchmark": "中证医药卫生指数收益率×70%+中证港股通综合指数收益率（人民币）×10%+中债-综合指数（全价）收益率×20%",
            "investment_target": "本基金也可以通过买入标的指数，追求跟踪标的指数，获得与指数相近的收益",
            "management_fee": 0.015,
            "custody_fee": 0.002,
        },
        "018291": {
            "code": "018291",
            "name": "广发新兴成长灵活配置混合型证券投资基金",
            "type": "混合型-灵活配置",
            "benchmark": "中证800指数收益率×65%+一年期人民币定期存款利率（税后）×35%",
            "management_fee": 0.015,
            "custody_fee": 0.002,
        },
        "018345": {
            "code": "018345",
            "name": "华夏中证机器人交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证机器人指数收益率×95%＋人民币活期存款税后利率×5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
        "020989": {
            "code": "020989",
            "name": "南方恒生科技交易型开放式指数证券投资基金发起式联接基金（QDII）",
            "type": "QDII-股票",
            "benchmark": "经汇率调整后的恒生科技指数收益率×95%+银行人民币活期存款利率（税后）×5%",
            "management_fee": 0.015,
            "custody_fee": 0.003,
            "investment_strategy": "",
            "investment_target": "",
        },
        "023639": {
            "code": "023639",
            "name": "国泰恒生A股电网设备交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "恒生A股电网设备指数收益率*95%+银行活期存款利率(税后)*5%+也可以通过买入标的指数",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
    }
    lookup_calls = []

    def lookup_side_effect(index_name):
        lookup_calls.append(index_name)
        mapping = {
            "中证医药卫生": {"code": "930100", "name": "中证医药卫生指数"},
            "中证医药卫生指数": {"code": "930100", "name": "中证医药卫生指数"},
            "中证港股通综合": {"code": "930200", "name": "中证港股通综合指数"},
            "中证港股通综合指数": {"code": "930200", "name": "中证港股通综合指数"},
            "中证机器人": {"code": "930001", "name": "中证机器人指数"},
            "中证机器人指数": {"code": "930001", "name": "中证机器人指数"},
            "恒生A股电网设备": {"code": "930999", "name": "恒生A股电网设备指数"},
            "恒生A股电网设备指数": {"code": "930999", "name": "恒生A股电网设备指数"},
        }
        return mapping[index_name]

    def estimate_success(**kwargs):
        return {
            "fund_code": kwargs["fund_code"],
            "estimated_nav": 1.01,
            "estimated_return": 0.1,
            "nav_date": kwargs["nav_date"],
            "warnings": [],
        }

    with patch("otc_fund_quant.nav_estimator.core.nav_engine.logger.info") as mock_logger_info, \
         patch.object(engine.fund_fetcher, "get_fund_info", side_effect=lambda code: fund_info_map[code]), \
         patch.object(engine.fund_fetcher, "get_previous_official_nav", return_value=(1.0, "2026-03-07")), \
         patch.object(engine.fund_fetcher, "get_portfolio_holdings", return_value=holdings_df), \
         patch.object(engine.fund_fetcher, "resolve_qdii_proxy_components", return_value=[{"code": "HSTECH", "market": "港股", "weight": 1.0}]), \
         patch.object(engine.fund_fetcher, "is_index_fund", return_value=True), \
         patch.object(engine, "_validate_supported_qdii_profile", return_value="hk"), \
         patch.object(engine.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect), \
         patch.object(engine.classifier.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect), \
         patch("otc_fund_quant.nav_estimator.core.nav_engine.build_confidence_payload", return_value={}), \
         patch.object(engine.equity_estimator, "estimate", side_effect=estimate_success), \
         patch.object(engine.index_estimator, "estimate", side_effect=estimate_success), \
         patch.object(engine.qdii_estimator, "estimate", side_effect=estimate_success):
        results_df = engine.run(["015916", "018291", "018345", "020989", "023639"], strict=True)

    result_map = {row["fund_code"]: row for row in results_df.to_dict(orient="records")}
    assert result_map["015916"]["status"] == "成功"
    assert result_map["018291"]["status"] == "成功"
    assert result_map["018345"]["status"] == "成功"
    assert result_map["020989"]["status"] == "成功"
    assert result_map["023639"]["status"] == "成功"
    assert "中证机器人" not in lookup_calls
    assert "中证机器人指数" not in lookup_calls
    assert "恒生A股电网设备" not in lookup_calls
    assert "恒生A股电网设备指数" not in lookup_calls
    summary_messages = [
        call.args[0]
        for call in mock_logger_info.call_args_list
        if call.args and isinstance(call.args[0], str) and "批量估算完成" in call.args[0]
    ]
    assert summary_messages
    assert any("requested=5" in message and "succeeded=5" in message and "failed=0" in message for message in summary_messages)
    assert any("failure_breakdown={}" in message for message in summary_messages)
    assert all("%s" not in message for message in summary_messages)


def test_nav_engine_classifies_tracking_target_quote_failure():
    assert NAVEngine._classify_failure_reason(
        "A股 562500 实时行情 主备实时源均不可用；主源异常: A股 562500 行情数据结构异常; 备源异常: 新浪实时行情为空: 562500"
    ) == "跟踪标的行情失败"


def test_active_equity_estimator_strict_supports_a_share_etf_proxy_components():
    estimator = ActiveEquityEstimator()
    holdings_df = pd.DataFrame([{"code": "600000", "name": "A", "weight": 10.0, "market": "A股"}])
    proxy_components = [
        {"code": "512480", "quote_code": "512480", "name": "半导体ETF", "market": "A股", "weight": 0.875, "target_type": "a_share_etf_proxy"},
        {"code": "HSI", "quote_code": "HSI", "name": "恒生指数", "market": "港股", "weight": 0.125, "target_type": "index"},
    ]

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", return_value={
        "position": 80.0,
        "report_date": "2025-12-31",
        "raw_report_period": "2025-12-31",
        "snapshot_source": "pingzhongdata_asset_allocation",
    }), \
         patch.object(estimator.fund_fetcher, "extract_holdings_report_metadata", return_value=("2025Q4", "2025-12-31")), \
         patch.object(estimator.stock_fetcher, "_get_a_share_quotes_live_by_codes_partial", return_value=({
             "600000": {
                 "value": 1.0,
                 "source": "eastmoney_single_quote",
                 "source_disagreements": [],
                 "data_as_of_date": "2026-03-12",
                 "raw": {"change_pct": 1.0},
             }
         }, [])), \
         patch.object(estimator.stock_fetcher, "_get_hk_share_quotes_live_by_codes_partial", return_value=({}, [])), \
         patch.object(estimator.stock_fetcher, "get_a_share_quote_live", return_value={
             "value": -2.03,
             "source": "eastmoney_single_quote",
             "source_disagreements": [],
             "data_as_of_date": "2026-03-12",
             "raw": {"change_pct": -2.03},
         }) as mock_etf_quote, \
         patch.object(estimator.index_fetcher, "get_hk_index_return_live", return_value={
             "value": 0.75,
             "source": "eastmoney_index_quote",
             "source_disagreements": [],
             "data_as_of_date": "2026-03-12",
             "raw": {"change_pct": 0.75},
         }), \
         patch.object(estimator.fx_fetcher, "get_hkd_cny_daily_change_live", return_value={
             "value": 0.06,
             "source": "safe_mid_rate",
             "source_disagreements": [],
             "data_as_of_date": "2026-03-12",
             "raw": {"change_pct": 0.06},
         }), \
         patch.object(estimator.index_fetcher, "get_a_index_return_live", side_effect=AssertionError("should not use a-index fetcher for ETF proxy")):
        result = estimator.estimate(
            fund_code="025209",
            last_nav=1.6765,
            nav_date="2026-03-11",
            target_date="2026-03-12",
            holdings=holdings_df,
            proxy_components=proxy_components,
            strict=True,
        )

    mock_etf_quote.assert_called_once_with("512480", strict=True)
    assert result["estimated_nav"] > 0
    assert result["quality_gate_passed"] is True
    assert result["proxy_components"][0]["target_type"] == "a_share_etf_proxy"
    assert result["proxy_components"][0]["quote_code"] == "512480"
    assert result["used_sources"]["proxy_tracking_targets"]["512480"] == "eastmoney_single_quote"


def test_nav_engine_strict_run_supports_current_eleven_fund_batch():
    engine = NAVEngine(strict=True)
    holdings_df = pd.DataFrame([{"code": "600000", "name": "A", "weight": 10.0, "market": "A股"}])
    fund_info_map = {
        "015916": {
            "code": "015916",
            "name": "永赢医药创新智选混合型发起式证券投资基金",
            "type": "混合型-偏股",
            "benchmark": "中证医药卫生指数收益率×70%+中证港股通综合指数收益率（人民币）×10%+中债-综合指数（全价）收益率×20%",
            "management_fee": 0.015,
            "custody_fee": 0.002,
        },
        "018291": {
            "code": "018291",
            "name": "广发新兴成长灵活配置混合型证券投资基金",
            "type": "混合型-灵活配置",
            "benchmark": "中证800指数收益率×65%+一年期人民币定期存款利率（税后）×35%",
            "management_fee": 0.015,
            "custody_fee": 0.002,
        },
        "018345": {
            "code": "018345",
            "name": "华夏中证机器人交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证机器人指数收益率×95%＋人民币活期存款税后利率×5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
        "007467": {
            "code": "007467",
            "name": "华泰柏瑞中证红利低波动交易型开放式指数证券投资基金联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证红利低波动指数收益率×95%+银行活期存款利率(税后)×5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
        "020640": {
            "code": "020640",
            "name": "广发中证半导体材料设备主题交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证半导体材料设备主题指数收益率×95%+人民币活期存款税后利率×5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
        "020989": {
            "code": "020989",
            "name": "南方恒生科技交易型开放式指数证券投资基金发起式联接基金（QDII）",
            "type": "QDII-股票",
            "benchmark": "经汇率调整后的恒生科技指数收益率×95%+银行人民币活期存款利率（税后）×5%",
            "management_fee": 0.015,
            "custody_fee": 0.003,
            "investment_strategy": "",
            "investment_target": "",
        },
        "022464": {
            "code": "022464",
            "name": "富国中证A500交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证A500指数收益率×95%+银行活期存款利率(税后)×5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
        "023639": {
            "code": "023639",
            "name": "国泰恒生A股电网设备交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "恒生A股电网设备指数收益率*95%+银行活期存款利率(税后)*5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
        "023598": {
            "code": "023598",
            "name": "景顺长城中证港股通创新药交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "中证港股通创新药指数收益率×95%+银行活期存款利率(税后)×5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
        "024663": {
            "code": "024663",
            "name": "富国创业板人工智能交易型开放式指数证券投资基金发起式联接基金",
            "type": "股票型-标准指数",
            "benchmark": "创业板人工智能指数收益率×95%+银行活期存款利率(税后)×5%",
            "management_fee": 0.005,
            "custody_fee": 0.001,
        },
        "025209": {
            "code": "025209",
            "name": "永赢先锋半导体智选混合型发起式证券投资基金",
            "type": "混合型-偏股",
            "benchmark": "中证全指半导体产品与设备指数收益率×70%+恒生指数收益率×10%+中债-综合指数（全价）收益率×20%",
            "management_fee": 0.015,
            "custody_fee": 0.002,
        },
    }

    def estimate_success(**kwargs):
        return {
            "fund_code": kwargs["fund_code"],
            "estimated_nav": 1.01,
            "estimated_return": 0.1,
            "nav_date": kwargs["nav_date"],
            "warnings": [],
        }

    def lookup_side_effect(index_name):
        mapping = {
            "中证医药卫生": {"code": "000933", "name": "中证医药卫生"},
            "中证医药卫生指数": {"code": "000933", "name": "中证医药卫生"},
            "中证港股通综合": {"code": "930930", "name": "中证港股通综合"},
            "中证港股通综合指数": {"code": "930930", "name": "中证港股通综合"},
            "中证A500": {"code": "000510", "name": "中证A500"},
            "中证A500指数": {"code": "000510", "name": "中证A500"},
        }
        return mapping[index_name]

    with patch("otc_fund_quant.nav_estimator.core.nav_engine.logger.info") as mock_logger_info, \
         patch.object(engine.fund_fetcher, "get_fund_info", side_effect=lambda code: fund_info_map[code]), \
         patch.object(engine.fund_fetcher, "get_previous_official_nav", return_value=(1.0, "2026-03-11")), \
         patch.object(engine.fund_fetcher, "get_portfolio_holdings", return_value=holdings_df), \
         patch.object(engine.fund_fetcher, "resolve_qdii_proxy_components", return_value=[{"code": "HSTECH", "market": "港股", "weight": 1.0}]), \
         patch.object(engine.fund_fetcher, "is_index_fund", return_value=True), \
         patch.object(engine, "_validate_supported_qdii_profile", return_value="hk"), \
         patch.object(engine.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect), \
         patch.object(engine.classifier.fund_fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect), \
         patch("otc_fund_quant.nav_estimator.core.nav_engine.build_confidence_payload", return_value={}), \
         patch.object(engine.equity_estimator, "estimate", side_effect=estimate_success) as mock_equity_estimate, \
         patch.object(engine.index_estimator, "estimate", side_effect=estimate_success) as mock_index_estimate, \
         patch.object(engine.qdii_estimator, "estimate", side_effect=estimate_success) as mock_qdii_estimate:
        results_df = engine.run(
            ["015916", "018291", "018345", "007467", "020640", "020989", "022464", "023639", "023598", "024663", "025209"],
            strict=True,
        )

    assert list(results_df["fund_code"]) == ["015916", "018291", "018345", "007467", "020640", "020989", "022464", "023639", "023598", "024663", "025209"]
    assert set(results_df["status"]) == {"成功"}
    assert {call.kwargs["fund_code"] for call in mock_equity_estimate.call_args_list} == {"015916", "018291", "025209"}
    assert {call.kwargs["fund_code"] for call in mock_index_estimate.call_args_list} == {"007467", "018345", "020640", "022464", "023598", "023639", "024663"}
    assert {call.kwargs["fund_code"] for call in mock_qdii_estimate.call_args_list} == {"020989"}
    index_kwargs = {call.kwargs["fund_code"]: call.kwargs for call in mock_index_estimate.call_args_list}
    assert index_kwargs["007467"]["tracking_target"] == {
        "target_type": "linked_etf_a_share",
        "security_code": "512890",
        "market": "A股",
        "tracking_name": "红利低波ETF",
    }
    assert index_kwargs["020640"]["tracking_target"] == {
        "target_type": "linked_etf_a_share",
        "security_code": "560780",
        "market": "A股",
        "tracking_name": "广发中证半导体材料设备主题ETF",
    }
    assert index_kwargs["023598"]["tracking_target"] == {
        "target_type": "linked_etf_a_share",
        "security_code": "513780",
        "market": "A股",
        "tracking_name": "港股创新药ETF",
    }
    assert index_kwargs["024663"]["tracking_target"] == {
        "target_type": "linked_etf_a_share",
        "security_code": "159246",
        "market": "A股",
        "tracking_name": "创业板人工智能ETF富国",
    }
    summary_messages = [
        call.args[0]
        for call in mock_logger_info.call_args_list
        if call.args and isinstance(call.args[0], str) and "批量估算完成" in call.args[0]
    ]
    assert any("requested=11" in message and "succeeded=11" in message and "failed=0" in message for message in summary_messages)
    assert any("failure_breakdown={}" in message for message in summary_messages)


def test_qdii_index_strict_still_rejects_zero_disclosed_stock_position():
    estimator = QDIIHKEstimator()

    with patch.object(estimator.fund_fetcher, "get_fund_stock_position_snapshot", return_value={
        "position": 0.0,
        "report_date": "2025-12-31",
        "raw_report_period": "2025-12-31",
        "snapshot_source": "pingzhongdata_asset_allocation",
    }), \
         patch.object(estimator.fx_fetcher, "get_hkd_cny_daily_change_live", return_value={
             "value": 0.0,
             "source": "safe_mid_rate",
             "source_disagreements": [],
             "data_as_of_date": "2026-03-08",
         }):
        with pytest.raises(ValueError, match="QDII股票仓位非法"):
            estimator.estimate(
                fund_code="900999",
                last_nav=1.0,
                nav_date="2026-03-07",
                target_date="2026-03-08",
                is_index_fund=True,
                market_profile="hk",
                proxy_components=[{"code": "HSI", "name": "恒生指数", "market": "港股", "weight": 1.0}],
                strict=True,
            )


def test_nav_estimator_page_default_strict_checked():
    client = _client()
    response = client.get("/nav-estimator")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    strict_index = html.index('id="strict-mode"')
    assert "checked" in html[strict_index:strict_index + 80]


def test_failure_response_preserves_quality_gate_details():
    frame = pd.DataFrame(
        [
            {
                "fund_code": "007343",
                "status": "失败",
                "warnings": [],
                "quality_policy": "quality_first",
                "quality_gate_passed": False,
                "quality_gate_failed_reasons": ["缺少基金披露股票总仓位"],
            }
        ]
    )

    payload = web_app._build_nav_estimator_response(frame, strict=True)
    details = payload["results"][0]["details"]

    assert details["quality_policy"] == "quality_first"
    assert details["quality_gate_failed_reasons"] == ["缺少基金披露股票总仓位"]


def test_analyze_api_smoke_unchanged():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "HOLD",
        "reason": "smoke",
        "buy_score": 1,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.5,
            "is_cheap_zone": False,
            "gold_cross": False,
            "death_cross": False,
            "rsi": 50,
            "macd_turn_positive": False,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 18,
            "market_regime": "RANGE",
        },
    }

    context = _resolved_strategy_context()
    generator = MagicMock(return_value=recommendation)

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=generator)), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(web_app, "_get_backtest_cache", return_value=[]):
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["fund_code"] == "007343"
    assert payload["recommendation"]["action"] == "HOLD"
    assert payload["strategy"] == "v6"
    assert payload["strategy_context"]["profile_id"] == "equity_active_cn"
    generator.assert_called_once()


def test_analyze_api_rejects_invalid_recommendation_payload():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    context = _resolved_strategy_context()

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=None))):
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})

    assert response.status_code == 500
    payload = response.get_json()
    assert "策略 v6 返回了非法推荐结果类型: NoneType" in payload["error"]


def test_analyze_api_normalizes_recommendation_payload_and_indicator_keys():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    context = _resolved_strategy_context()
    recommendation = {
        "action": "hold",
        "reason": "normalized",
        "indicators": {
            "rsi": 52.5,
            "ma20": float("nan"),
        },
    }

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(web_app, "_get_backtest_cache", return_value=[]):
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["recommendation"]["action"] == "HOLD"
    assert payload["recommendation"]["buy_score"] == 0
    assert payload["recommendation"]["sell_signal"] is False
    assert set(payload["recommendation"]["indicators"].keys()) == set(web_app.INDICATOR_RESPONSE_KEYS)
    assert payload["recommendation"]["indicators"]["rsi"] == 52.5
    assert payload["recommendation"]["indicators"]["ma20"] is None
    assert payload["recommendation"]["indicators"]["percentile"] is None


def test_analyze_api_returns_clear_error_when_strategy_context_resolution_fails():
    client = _client()

    with patch.object(web_app, "resolve_strategy_context", side_effect=ValueError("画像 broken 未定义")):
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})

    assert response.status_code == 500
    payload = response.get_json()
    assert payload["error"] == "策略配置解析失败: 画像 broken 未定义"


def test_analyze_api_returns_strategy_context():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "BUY",
        "reason": "profile",
        "buy_score": 3,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.2,
            "is_cheap_zone": True,
            "gold_cross": True,
            "death_cross": False,
            "rsi": 42,
            "macd_turn_positive": True,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 24,
            "market_regime": "BULL",
        },
    }
    context = _resolved_strategy_context(
        effective_strategy="regime_adaptive",
        default_strategy="regime_adaptive",
        strategy_adjusted=True,
        adjustment_reason="当前画像不支持 index_momentum，已切换为默认策略",
        requested_strategy="index_momentum",
        available_ids=["v6", "regime_adaptive"],
    )

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(web_app, "_get_backtest_cache", return_value=[]):
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "index_momentum"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["fund_name"] == "测试基金"
    assert payload["strategy"] == "regime_adaptive"
    assert payload["strategy_context"]["requested_strategy"] == "index_momentum"
    assert payload["strategy_context"]["effective_strategy"] == "regime_adaptive"
    assert payload["strategy_context"]["strategy_adjusted"] is True
    assert payload["strategy_context"]["adjustment_reason"] == "当前画像不支持 index_momentum，已切换为默认策略"


def test_analyze_api_uses_effective_strategy_for_persistence():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "HOLD",
        "reason": "effective strategy",
        "buy_score": 1,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.5,
            "is_cheap_zone": False,
            "gold_cross": False,
            "death_cross": False,
            "rsi": 50,
            "macd_turn_positive": False,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 18,
            "market_regime": "RANGE",
        },
    }
    context = _resolved_strategy_context(
        requested_strategy="index_momentum",
        effective_strategy="regime_adaptive",
        default_strategy="regime_adaptive",
        strategy_adjusted=True,
        adjustment_reason="当前画像不支持 index_momentum，已切换为默认策略",
        available_ids=["v6", "regime_adaptive"],
    )
    mock_latest_signal_date = MagicMock(return_value="")
    mock_generate_signal_points = MagicMock(return_value={"records": []})
    mock_auto_save_reviews = MagicMock()
    mock_get_all_signal_points = MagicMock(return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []})
    mock_get_recommendations = MagicMock(return_value=[])
    mock_get_backtest_cache = MagicMock(return_value=[])

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", mock_latest_signal_date), \
         patch.object(web_app, "generate_signal_points", mock_generate_signal_points), \
         patch.object(web_app, "_auto_save_reviews", mock_auto_save_reviews), \
         patch.object(web_app, "_get_all_signal_points", mock_get_all_signal_points), \
         patch.object(web_app, "_get_recommendations", mock_get_recommendations), \
         patch.object(web_app, "_get_backtest_cache", mock_get_backtest_cache):
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "index_momentum"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["strategy"] == "regime_adaptive"
    mock_latest_signal_date.assert_called_once_with("007343", "regime_adaptive")
    assert mock_generate_signal_points.call_args.kwargs["strategy"] == "regime_adaptive"
    assert mock_generate_signal_points.call_args.kwargs["params"] == context.strategy_params
    assert mock_auto_save_reviews.call_args.kwargs["strategy"] == "regime_adaptive"
    mock_get_all_signal_points.assert_called_once_with("007343", "regime_adaptive")
    mock_get_recommendations.assert_called_once_with("007343", "regime_adaptive")
    mock_get_backtest_cache.assert_called_once()
    assert mock_get_backtest_cache.call_args.args[:2] == ("007343", "regime_adaptive")


def test_analyze_api_marks_period_returns_ready_on_cache_hit():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "HOLD",
        "reason": "cache hit",
        "buy_score": 1,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.5,
            "is_cheap_zone": False,
            "gold_cross": False,
            "death_cross": False,
            "rsi": 50,
            "macd_turn_positive": False,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 18,
            "market_regime": "RANGE",
        },
    }
    context = _resolved_strategy_context()
    cached_period_returns = [{"label": "近1年", "days": 365, "strategy_pct": 12.3, "fund_pct": 8.6}]

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(web_app, "_get_backtest_cache", return_value=cached_period_returns):
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["period_returns_status"] == "ready"
    assert payload["period_returns"] == cached_period_returns
    assert payload["period_returns_error"] is None
    assert payload["period_returns_end_date"] == "2025-01-20"


def test_analyze_api_returns_pending_when_period_returns_cache_misses():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "BUY",
        "reason": "queue",
        "buy_score": 2,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.2,
            "is_cheap_zone": True,
            "gold_cross": True,
            "death_cross": False,
            "rsi": 42,
            "macd_turn_positive": True,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 24,
            "market_regime": "BULL",
        },
    }
    context = _resolved_strategy_context()
    pending_future = Future()
    mock_executor = MagicMock()
    mock_executor.submit.return_value = pending_future

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(web_app, "_get_backtest_cache", return_value=None), \
         patch.object(web_app, "_ensure_backtest_executor", return_value=mock_executor), \
         patch.object(web_app, "calc_period_returns") as mock_calc_period_returns:
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["period_returns_status"] == "pending"
    assert payload["period_returns"] is None
    assert payload["period_returns_error"] is None
    mock_executor.submit.assert_called_once()
    mock_calc_period_returns.assert_not_called()
    task_key = web_app._backtest_task_key("007343", "v6", "2025-01-20")
    task = web_app._backtest_tasks[task_key]
    assert task["status"] == web_app.BACKTEST_STATUS_PENDING
    assert task["created_at"] is not None
    assert task["updated_at"] is not None
    assert task["started_at"] is not None
    assert task["completed_at"] is None
    assert task["task_id"]


def test_analyze_api_dedupes_pending_period_returns_tasks():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "HOLD",
        "reason": "dedupe",
        "buy_score": 1,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.5,
            "is_cheap_zone": False,
            "gold_cross": False,
            "death_cross": False,
            "rsi": 50,
            "macd_turn_positive": False,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 18,
            "market_regime": "RANGE",
        },
    }
    context = _resolved_strategy_context()
    pending_future = Future()
    mock_executor = MagicMock()
    mock_executor.submit.return_value = pending_future

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(web_app, "_get_backtest_cache", return_value=None), \
         patch.object(web_app, "_ensure_backtest_executor", return_value=mock_executor):
        response_one = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})
        response_two = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})

    assert response_one.status_code == 200
    assert response_two.status_code == 200
    assert response_one.get_json()["period_returns_status"] == "pending"
    assert response_two.get_json()["period_returns_status"] == "pending"
    mock_executor.submit.assert_called_once()


def test_period_returns_api_transitions_from_pending_to_ready():
    client = _client()
    task_key = web_app._backtest_task_key("007343", "v6", "2025-01-20")
    with web_app._backtest_tasks_lock:
        web_app._backtest_tasks[task_key] = {
            "status": web_app.BACKTEST_STATUS_PENDING,
            "error": None,
            "period_returns": None,
            "future": None,
        }

    with patch.object(web_app, "_get_backtest_cache", return_value=None):
        pending_response = client.get("/api/period-returns?fund_code=007343&strategy=v6&end_date=2025-01-20")

    assert pending_response.status_code == 200
    assert pending_response.get_json()["status"] == "pending"

    ready_returns = [{"label": "近1年", "days": 365, "strategy_pct": 10.5, "fund_pct": 8.1}]
    with patch.object(web_app, "_get_backtest_cache", return_value=ready_returns):
        ready_response = client.get("/api/period-returns?fund_code=007343&strategy=v6&end_date=2025-01-20")

    assert ready_response.status_code == 200
    payload = ready_response.get_json()
    assert payload["status"] == "ready"
    assert payload["period_returns"] == ready_returns
    assert payload["error"] is None


def test_period_returns_api_returns_error_state():
    client = _client()
    task_key = web_app._backtest_task_key("007343", "v6", "2025-01-20")
    with web_app._backtest_tasks_lock:
        web_app._backtest_tasks[task_key] = {
            "status": web_app.BACKTEST_STATUS_ERROR,
            "error": "区间收益计算失败: boom",
            "period_returns": None,
            "future": None,
        }

    with patch.object(web_app, "_get_backtest_cache", return_value=None):
        response = client.get("/api/period-returns?fund_code=007343&strategy=v6&end_date=2025-01-20")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "error"
    assert payload["period_returns"] is None
    assert payload["error"] == "区间收益计算失败: boom"


def test_period_returns_api_marks_stale_pending_task_as_error():
    client = _client()
    task_key = web_app._backtest_task_key("007343", "v6", "2025-01-20")
    stale_at = web_app._backtest_now() - web_app.BACKTEST_PENDING_TTL - timedelta(seconds=1)
    task = web_app._create_backtest_task(status=web_app.BACKTEST_STATUS_PENDING)
    task["created_at"] = stale_at
    task["updated_at"] = stale_at
    task["started_at"] = stale_at
    with web_app._backtest_tasks_lock:
        web_app._backtest_tasks[task_key] = task

    with patch.object(web_app, "_get_backtest_cache", return_value=None):
        response = client.get("/api/period-returns?fund_code=007343&strategy=v6&end_date=2025-01-20")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "error"
    assert payload["period_returns"] is None
    assert "超时" in payload["error"]
    assert web_app._backtest_tasks[task_key]["status"] == web_app.BACKTEST_STATUS_ERROR


def test_analyze_api_requeues_after_pending_task_becomes_stale():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "HOLD",
        "reason": "requeue stale",
        "buy_score": 1,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.5,
            "is_cheap_zone": False,
            "gold_cross": False,
            "death_cross": False,
            "rsi": 50,
            "macd_turn_positive": False,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 18,
            "market_regime": "RANGE",
        },
    }
    context = _resolved_strategy_context()
    task_key = web_app._backtest_task_key("007343", "v6", "2025-01-20")
    stale_at = web_app._backtest_now() - web_app.BACKTEST_PENDING_TTL - timedelta(seconds=1)
    stale_task = web_app._create_backtest_task(status=web_app.BACKTEST_STATUS_PENDING)
    stale_task["created_at"] = stale_at
    stale_task["updated_at"] = stale_at
    stale_task["started_at"] = stale_at
    with web_app._backtest_tasks_lock:
        web_app._backtest_tasks[task_key] = stale_task

    pending_future = Future()
    mock_executor = MagicMock()
    mock_executor.submit.return_value = pending_future

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(web_app, "_get_backtest_cache", return_value=None), \
         patch.object(web_app, "_ensure_backtest_executor", return_value=mock_executor):
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})

    assert response.status_code == 200
    assert response.get_json()["period_returns_status"] == "pending"
    mock_executor.submit.assert_called_once()
    assert web_app._backtest_tasks[task_key]["status"] == web_app.BACKTEST_STATUS_PENDING


def test_period_returns_payload_prunes_finished_task_after_retention():
    task_key = web_app._backtest_task_key("007343", "v6", "2025-01-20")
    old_completed_at = web_app._backtest_now() - web_app.BACKTEST_FINISHED_TTL - timedelta(seconds=1)
    task = web_app._create_backtest_task(status=web_app.BACKTEST_STATUS_ERROR, error="old error")
    task["created_at"] = old_completed_at
    task["updated_at"] = old_completed_at
    task["completed_at"] = old_completed_at
    with web_app._backtest_tasks_lock:
        web_app._backtest_tasks[task_key] = task

    with patch.object(web_app, "_get_backtest_cache", return_value=None):
        payload = web_app._get_period_returns_payload("007343", "v6", "2025-01-20")

    assert payload["status"] == "error"
    assert payload["error"] == "区间收益任务不存在，请重新分析"
    assert task_key not in web_app._backtest_tasks


def test_cache_hit_prunes_terminal_task_from_memory():
    task_key = web_app._backtest_task_key("007343", "v6", "2025-01-20")
    task = web_app._create_backtest_task(status=web_app.BACKTEST_STATUS_ERROR, error="boom")
    with web_app._backtest_tasks_lock:
        web_app._backtest_tasks[task_key] = task

    ready_returns = [{"label": "近1年", "days": 365, "strategy_pct": 11.1, "fund_pct": 8.2}]
    with patch.object(web_app, "_get_backtest_cache", return_value=ready_returns):
        payload = web_app._get_period_returns_payload("007343", "v6", "2025-01-20")

    assert payload["status"] == "ready"
    assert payload["period_returns"] == ready_returns
    assert task_key not in web_app._backtest_tasks


def test_run_sqlite_write_with_retry_retries_locked_error(monkeypatch):
    logger = MagicMock()
    state = {"count": 0}

    def _action():
        state["count"] += 1
        if state["count"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    monkeypatch.setattr(sqlite_utils.time, "sleep", lambda _seconds: None)
    result = sqlite_utils.run_sqlite_write_with_retry(_action, logger=logger, action_name="retry_test")

    assert result == "ok"
    assert state["count"] == 3
    assert logger.warning.call_count == 2


def test_run_sqlite_write_with_retry_raises_after_retry_limit(monkeypatch):
    logger = MagicMock()
    state = {"count": 0}

    def _action():
        state["count"] += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite_utils.time, "sleep", lambda _seconds: None)
    with pytest.raises(sqlite3.OperationalError):
        sqlite_utils.run_sqlite_write_with_retry(_action, logger=logger, action_name="retry_test")

    assert state["count"] == 4
    assert logger.warning.call_count == 3


def test_analyze_api_keeps_200_when_period_returns_background_state_errors():
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "HOLD",
        "reason": "error state",
        "buy_score": 1,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.5,
            "is_cheap_zone": False,
            "gold_cross": False,
            "death_cross": False,
            "rsi": 50,
            "macd_turn_positive": False,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 18,
            "market_regime": "RANGE",
        },
    }
    context = _resolved_strategy_context()

    with patch.object(web_app, "resolve_strategy_context", return_value=context), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(
             web_app,
             "_ensure_period_returns",
             return_value={"status": "error", "period_returns": None, "error": "区间收益计算失败: boom"},
         ):
        response = client.post("/api/analyze", json={"fund_code": "007343", "strategy": "v6"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["period_returns_status"] == "error"
    assert payload["period_returns"] is None
    assert payload["period_returns_error"] == "区间收益计算失败: boom"


@pytest.mark.parametrize(
    ("fund_type", "requested_strategy", "expected_strategy", "expected_available_ids", "expected_param_subset"),
    [
        ("active_a", "", "regime_adaptive", ["v6", "regime_adaptive"], {"max_position_ratio": 0.8, "bull_dca_boost": 1.5}),
        ("active_hk", "", "regime_adaptive", ["v6", "regime_adaptive"], {"max_position_ratio": 0.7, "bull_dca_boost": 1.3}),
        ("index_a", "", "index_momentum", ["v6", "regime_adaptive", "index_momentum"], {"max_position_ratio": 0.95, "momentum_buy_threshold": 2}),
        ("index_hk", "", "index_momentum", ["v6", "regime_adaptive", "index_momentum"], {"max_position_ratio": 0.85, "momentum_buy_threshold": 3}),
        ("bond_pure", "", "bond_stability", ["bond_stability"], {"max_position_ratio": 0.6, "volatility_guard_window": 60}),
        ("bond_plus", "", "bond_plus_balance", ["v6", "regime_adaptive", "bond_plus_balance"], {"max_position_ratio": 0.68, "drawdown_guard_threshold": 0.05}),
        ("qdii", "", "qdii_trend", ["v6", "regime_adaptive", "qdii_trend"], {"max_position_ratio": 0.72, "trend_window_slow": 120}),
    ],
)
def test_analyze_api_surfaces_category_specific_strategy_context(
    monkeypatch,
    fund_type,
    requested_strategy,
    expected_strategy,
    expected_available_ids,
    expected_param_subset,
):
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "HOLD",
        "reason": "category matrix",
        "buy_score": 1,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.5,
            "is_cheap_zone": False,
            "gold_cross": False,
            "death_cross": False,
            "rsi": 50,
            "macd_turn_positive": False,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 18,
            "market_regime": "RANGE",
            "bb_upper": None,
            "bb_lower": None,
            "bb_position": 0.5,
            "bb_width": 0.1,
            "bb_squeeze": False,
            "bb_touched_lower_3d": False,
            "mom_5d": 0.01,
            "mom_10d": 0.02,
            "mom_20d": 0.03,
            "atr_median": 0.02,
        },
    }

    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试基金"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: fund_type)
    strategy_loader.reload_config()
    monkeypatch.setattr(web_app, "resolve_strategy_context", strategy_loader.resolve_strategy_context)

    with patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(web_app, "_ensure_period_returns", return_value={"status": "ready", "period_returns": [], "error": None}):
        response = client.post("/api/analyze", json={"fund_code": "999999", "strategy": requested_strategy})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["strategy"] == expected_strategy
    assert [item["id"] for item in payload["strategy_context"]["available_strategies"]] == expected_available_ids
    for key, expected_value in expected_param_subset.items():
        assert payload["strategy_params"][key] == expected_value


def test_analyze_api_adjusts_disallowed_strategy_for_bond_pure(monkeypatch):
    client = _client()
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20).date,
            "nav": [1 + idx * 0.01 for idx in range(20)],
        }
    )
    recommendation = {
        "action": "HOLD",
        "reason": "adjust bond pure",
        "buy_score": 1,
        "sell_signal": False,
        "indicators": {
            "current_nav": 1.2,
            "percentile": 0.5,
            "is_cheap_zone": False,
            "gold_cross": False,
            "death_cross": False,
            "rsi": 50,
            "macd_turn_positive": False,
            "macd_5d_negative": False,
            "above_ma20": True,
            "above_ma20_3d": True,
            "ma20": 1.1,
            "ma60": 1.0,
            "macd_hist": 0.01,
            "atr": 0.02,
            "adx": 18,
            "market_regime": "RANGE",
        },
    }

    monkeypatch.setattr(strategy_loader, "_fetch_fund_info", lambda fund_code: {"name": "测试纯债"})
    monkeypatch.setattr(strategy_loader, "_classify_fund", lambda fund_code, fund_info=None: "bond_pure")
    strategy_loader.reload_config()
    monkeypatch.setattr(web_app, "resolve_strategy_context", strategy_loader.resolve_strategy_context)

    with patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "get_strategy_definition", return_value=SimpleNamespace(generator=MagicMock(return_value=recommendation))), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
         patch.object(web_app, "_get_latest_signal_date", return_value=""), \
         patch.object(web_app, "generate_signal_points", return_value={"records": []}), \
         patch.object(web_app, "_auto_save_reviews"), \
         patch.object(web_app, "_get_all_signal_points", return_value={"buy_dates": [], "buy_navs": [], "sell_dates": [], "sell_navs": []}), \
         patch.object(web_app, "_get_recommendations", return_value=[]), \
         patch.object(web_app, "_ensure_period_returns", return_value={"status": "ready", "period_returns": [], "error": None}):
        response = client.post("/api/analyze", json={"fund_code": "999999", "strategy": "regime_adaptive"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["strategy"] == "bond_stability"
    assert payload["strategy_context"]["strategy_adjusted"] is True
    assert payload["strategy_context"]["adjustment_reason"] == "当前画像不支持 regime_adaptive，已切换为默认策略"
