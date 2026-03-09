import os
import sys
import http.client
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
from otc_fund_quant.nav_estimator.data.fetcher.base_fetcher import BaseFetcher
from otc_fund_quant.nav_estimator.data.fetcher.fund_fetcher import FundFetcher
from otc_fund_quant.nav_estimator.estimator.active_equity_estimator import ActiveEquityEstimator
from otc_fund_quant.nav_estimator.estimator.bond_estimator import BondEstimator
from otc_fund_quant.nav_estimator.estimator.index_estimator import IndexEstimator
from otc_fund_quant.nav_estimator.estimator.qdii_hk_estimator import QDIIHKEstimator
from otc_fund_quant.web import app as web_app


def _client():
    web_app.app.config["TESTING"] = True
    return web_app.app.test_client()


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
    fake_df = pd.DataFrame(
        [
            {"代码": "930001", "名称": "中证机器人指数"},
            {"代码": "000906", "名称": "中证800"},
        ]
    )

    with patch("otc_fund_quant.nav_estimator.data.fetcher.fund_fetcher.ak.stock_zh_index_spot_em", return_value=fake_df):
        resolved = fetcher._lookup_a_index_code_by_name("中证机器人")

    assert resolved["code"] == "930001"
    assert resolved["name"] == "中证机器人指数"


def test_resolve_a_index_code_supports_extended_aliases_and_dynamic_lookup():
    fetcher = FundFetcher()

    def lookup_side_effect(index_name):
        mapping = {
            "中证机器人指数": {"code": "930001", "name": "中证机器人指数"},
            "恒生A股电网设备指数": {"code": "930999", "name": "恒生A股电网设备指数"},
        }
        return mapping[index_name]

    with patch.object(fetcher, "_lookup_a_index_code_by_name", side_effect=lookup_side_effect):
        assert fetcher.resolve_a_index_code({"benchmark": "中证800指数收益率×95%+存款利率×5%"}) == "000906"
        assert fetcher.resolve_a_index_code({"benchmark": "中证机器人指数收益率×95%+存款利率×5%"}) == "930001"
        assert fetcher.resolve_a_index_code({"benchmark": "恒生A股电网设备指数收益率×95%+存款利率×5%"}) == "930999"


def test_resolve_active_proxy_components_filters_non_equity_and_renormalizes_weights():
    fetcher = FundFetcher()

    def lookup_side_effect(index_name):
        mapping = {
            "中证医药卫生指数": {"code": "930100", "name": "中证医药卫生指数"},
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
            "中证医药卫生指数": {"code": "930100", "name": "中证医药卫生指数"},
            "中证港股通综合指数": {"code": "930200", "name": "中证港股通综合指数"},
            "中证机器人指数": {"code": "930001", "name": "中证机器人指数"},
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

    with patch.object(web_app, "load_strategy_params", return_value={"max_position_ratio": 1.0}), \
         patch.object(web_app.loader, "update_db"), \
         patch.object(web_app, "_get_fund_history", return_value=history_df), \
         patch.object(web_app, "generate_recommendation", return_value=recommendation), \
         patch.object(web_app, "get_chart_data", return_value={"dates": [], "nav": [], "ma20": [], "ma60": []}), \
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
