"""净值估算引擎。"""

from collections import Counter
from datetime import date, datetime
from typing import Any

import pandas as pd
from loguru import logger

from ..config.settings import BOND_PLUS_MODEL, PROXY_INDEX_MAP, STRICT_MODE
from ..data.fetcher.bond_fetcher import BondFetcher
from ..data.fetcher.base_fetcher import BaseFetcher
from ..data.fetcher.fund_fetcher import FundFetcher
from ..data.fetcher.fx_fetcher import FXFetcher
from ..data.fetcher.index_fetcher import IndexFetcher
from ..data.fetcher.stock_fetcher import StockFetcher
from ..estimator.active_equity_estimator import ActiveEquityEstimator
from ..estimator.base_estimator import BaseEstimator
from ..estimator.bond_estimator import BondEstimator
from ..estimator.index_estimator import IndexEstimator
from ..estimator.qdii_hk_estimator import QDIIHKEstimator
from .batch_context import BatchContext
from .confidence_scorer import build_confidence_payload
from .fund_classifier import FundClassifier
from .quality_gate import build_quality_metadata


class UnsupportedFundError(ValueError):
    """当前版本明确不支持的基金画像。"""

    def __init__(self, message: str, unsupported_reason: str, qdii_market_profile: str | None = None):
        super().__init__(message)
        self.unsupported_reason = unsupported_reason
        self.qdii_market_profile = qdii_market_profile


class NAVEngine:
    """净值估算引擎。"""

    def __init__(self, strict: bool = STRICT_MODE):
        self.classifier = FundClassifier()
        self.fund_fetcher = FundFetcher()
        self.stock_fetcher = StockFetcher()
        self.index_fetcher = IndexFetcher()
        self.bond_fetcher = BondFetcher()
        self.fx_fetcher = FXFetcher()
        self.strict = strict

        self.index_estimator = IndexEstimator()
        self.equity_estimator = ActiveEquityEstimator()
        self.bond_estimator = BondEstimator()
        self.qdii_estimator = QDIIHKEstimator()

        logger.info(f"净值估算引擎初始化完成，strict={self.strict}")

    @staticmethod
    def _is_tracking_target_quote_failure(error_message: str) -> bool:
        return (
            error_message.startswith("A股 ")
            and "实时行情 主备实时源均不可用" in error_message
            and ("行情数据结构异常" in error_message or "新浪实时行情为空" in error_message)
        )

    @staticmethod
    def _log_estimation_failure(fund_code: str, error: Exception, *, stage: str) -> None:
        error_message = str(error)
        if NAVEngine._is_tracking_target_quote_failure(error_message):
            logger.warning(f"基金 {fund_code} {stage}失败: 跟踪标的行情失败, error={error_message}")
            return
        if "实时行情源不支持该指数代码" in error_message:
            logger.warning(f"基金 {fund_code} {stage}失败: 指数行情不支持, error={error_message}")
            return
        if (
            "映射到多个A股指数代码" in error_message
            or "无法映射到A股指数代码" in error_message
            or "未识别到任何权益指数名称" in error_message
            or "未识别到任何A股权益指数名称" in error_message
            or "未识别到任何港股权益指数名称" in error_message
            or "未解析指数成分" in error_message
        ):
            logger.warning(f"基金 {fund_code} {stage}失败: 基准解析失败, error={error_message}")
            return
        logger.error(f"基金 {fund_code} {stage}失败: {error_message}")

    @staticmethod
    def _classify_failure_reason(error_message: str) -> str:
        if NAVEngine._is_tracking_target_quote_failure(error_message):
            return "跟踪标的行情失败"
        if "实时行情源不支持该指数代码" in error_message:
            return "指数行情不支持"
        if (
            "映射到多个A股指数代码" in error_message
            or "无法映射到A股指数代码" in error_message
            or "未识别到任何权益指数名称" in error_message
            or "未识别到任何A股权益指数名称" in error_message
            or "未识别到任何港股权益指数名称" in error_message
            or "未解析指数成分" in error_message
        ):
            return "基准解析失败"
        return "其他"

    def run(self, fund_codes: list[str], strict: bool | None = None, target_date: str | None = None) -> pd.DataFrame:
        if strict is None:
            strict = self.strict

        target_date = self._validate_realtime_target_date(target_date)
        logger.info(f"开始批量估算，共 {len(fund_codes)} 只基金")

        with BaseFetcher.request_scope(strict=strict):
            preloaded_inputs: list[dict[str, Any]] = []
            results: list[dict[str, Any] | None] = [None] * len(fund_codes)

            for index, code in enumerate(fund_codes):
                fund_type: str | None = None
                try:
                    fund_info = self.fund_fetcher.get_fund_info(code)
                    fund_type = self.classifier.classify(code, fund_info=fund_info)
                    last_nav, nav_date = self.fund_fetcher.get_previous_official_nav(code, target_date=target_date)
                    active_proxy_components = None
                    index_tracking_target = None
                    qdii_is_index_fund = None
                    qdii_market_profile = None
                    portfolio_holdings = None
                    if fund_type in {"active_a", "active_hk"}:
                        active_proxy_components = self.fund_fetcher.resolve_active_proxy_components(fund_info)
                        portfolio_holdings = self.fund_fetcher.get_portfolio_holdings(code)
                    elif fund_type == "index_a":
                        index_tracking_target = self.fund_fetcher.resolve_index_tracking_target(fund_info)
                    elif fund_type == "qdii":
                        qdii_market_profile = self._validate_supported_qdii_profile(code, fund_info)
                        qdii_is_index_fund = self.fund_fetcher.is_index_fund(fund_info)
                        active_proxy_components = self.fund_fetcher.resolve_qdii_proxy_components(fund_info)
                        if not qdii_is_index_fund:
                            portfolio_holdings = self.fund_fetcher.get_portfolio_holdings(code)

                    preloaded_inputs.append(
                        {
                            "input_index": index,
                            "fund_code": code,
                            "fund_info": fund_info,
                            "fund_type": fund_type,
                            "last_nav": last_nav,
                            "nav_date": nav_date,
                            "active_proxy_components": active_proxy_components,
                            "index_tracking_target": index_tracking_target,
                            "qdii_is_index_fund": qdii_is_index_fund,
                            "qdii_market_profile": qdii_market_profile,
                            "portfolio_holdings": portfolio_holdings,
                        }
                    )
                except Exception as e:
                    self._log_estimation_failure(code, e, stage="预取阶段")
                    results[index] = self._build_failure_result(
                        fund_code=code,
                        error=str(e),
                        target_date=target_date,
                        strict=strict,
                        fund_type=fund_type,
                        unsupported_reason=getattr(e, "unsupported_reason", None),
                        qdii_market_profile=getattr(e, "qdii_market_profile", None),
                    )

            batch_context = self._build_batch_context(preloaded_inputs, strict=strict)

            for preload_item in preloaded_inputs:
                code = preload_item["fund_code"]
                input_index = preload_item["input_index"]
                try:
                    result = self.estimate_single(
                        code,
                        strict=strict,
                        target_date=target_date,
                        batch_context=batch_context,
                        preloaded_meta=preload_item,
                    )
                    results[input_index] = result
                except Exception as e:
                    self._log_estimation_failure(code, e, stage="估算阶段")
                    results[input_index] = self._build_failure_result(
                        fund_code=code,
                        error=str(e),
                        target_date=target_date,
                        strict=strict,
                        fund_type=preload_item.get("fund_type"),
                        unsupported_reason=getattr(e, "unsupported_reason", None),
                        qdii_market_profile=getattr(e, "qdii_market_profile", None),
                        quality_metadata=getattr(e, "quality_details", None),
                    )

        normalized_results = [item for item in results if item is not None]
        df = pd.DataFrame(normalized_results)
        if not df.empty and "status" in df.columns:
            succeeded = len(df[df["status"] != "失败"])
            failed = len(df[df["status"] == "失败"])
            failure_counter = Counter(
                self._classify_failure_reason(str(row.get("error", "")))
                for row in df.to_dict(orient="records")
                if row.get("status") == "失败"
            )
            logger.info(
                f"批量估算完成 requested={len(fund_codes)} "
                f"succeeded={succeeded} failed={failed} "
                f"failure_breakdown={dict(failure_counter)}"
            )
        else:
            logger.info("批量估算完成，无有效结果")
        return df

    def estimate_single(
        self,
        fund_code: str,
        strict: bool | None = None,
        target_date: str | None = None,
        batch_context: BatchContext | None = None,
        preloaded_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if strict is None:
            strict = self.strict

        target_date = self._validate_realtime_target_date(target_date)
        logger.info("=" * 60)
        logger.info(f"开始估算基金: {fund_code}")
        qdii_is_index_fund: bool | None = None
        fund_type: str | None = None
        qdii_market_profile: str | None = None
        index_tracking_target: dict[str, Any] | None = None

        try:
            if preloaded_meta is None:
                fund_info = self.fund_fetcher.get_fund_info(fund_code)
                fund_type = self.classifier.classify(fund_code, fund_info=fund_info)
                active_proxy_components = None
                portfolio_holdings = None
                if fund_type in {"active_a", "active_hk"}:
                    active_proxy_components = self.fund_fetcher.resolve_active_proxy_components(fund_info)
                elif fund_type == "index_a":
                    index_tracking_target = self.fund_fetcher.resolve_index_tracking_target(fund_info)
                elif fund_type == "qdii":
                    qdii_market_profile = self._validate_supported_qdii_profile(fund_code, fund_info)
                    qdii_is_index_fund = self.fund_fetcher.is_index_fund(fund_info)
                    active_proxy_components = self.fund_fetcher.resolve_qdii_proxy_components(fund_info)
                last_nav, nav_date = self.fund_fetcher.get_previous_official_nav(fund_code, target_date=target_date)
            else:
                fund_info = preloaded_meta["fund_info"]
                fund_type = preloaded_meta["fund_type"]
                last_nav = preloaded_meta["last_nav"]
                nav_date = preloaded_meta["nav_date"]
                active_proxy_components = preloaded_meta.get("active_proxy_components")
                index_tracking_target = preloaded_meta.get("index_tracking_target")
                qdii_is_index_fund = preloaded_meta.get("qdii_is_index_fund")
                qdii_market_profile = preloaded_meta.get("qdii_market_profile")
                portfolio_holdings = preloaded_meta.get("portfolio_holdings")

            logger.info(f"基金类型: {fund_type}")
            logger.info(f"前一交易日净值: {last_nav}, 净值日期: {nav_date}")

            mgmt_rate = self._normalize_fee_value(fund_info.get("management_fee"), "管理费率", fund_code)
            custody_rate = self._normalize_fee_value(fund_info.get("custody_fee"), "托管费率", fund_code)
            fee_kwargs: dict[str, float] = {}
            if mgmt_rate is not None:
                fee_kwargs["mgmt_rate"] = mgmt_rate
            if custody_rate is not None:
                fee_kwargs["custody_rate"] = custody_rate

            estimator = self._get_estimator(fund_type)

            if fund_type == "index_a":
                if index_tracking_target is None:
                    index_tracking_target = self.fund_fetcher.resolve_index_tracking_target(fund_info)
                index_code = str(index_tracking_target.get("code") or index_tracking_target.get("security_code", "")).strip()
                if index_code == "":
                    raise ValueError(f"基金 {fund_code} 缺少有效跟踪标的代码")
                estimate_kwargs = {
                    "fund_code": fund_code,
                    "last_nav": last_nav,
                    "nav_date": nav_date,
                    "index_code": index_code,
                    "tracking_target": index_tracking_target,
                    "target_date": target_date,
                    "fund_info": fund_info,
                    "strict": strict,
                }
                estimate_kwargs.update(fee_kwargs)
                if batch_context is not None:
                    estimate_kwargs["batch_context"] = batch_context
                result = estimator.estimate(**estimate_kwargs)

            elif fund_type == "index_hk":
                hk_index_code = self.fund_fetcher.resolve_hk_index_code(fund_info)
                estimate_kwargs = {
                    "fund_code": fund_code,
                    "last_nav": last_nav,
                    "nav_date": nav_date,
                    "target_date": target_date,
                    "is_index_fund": True,
                    "hk_index_code": hk_index_code,
                    "strict": strict,
                }
                estimate_kwargs.update(fee_kwargs)
                if batch_context is not None:
                    estimate_kwargs["batch_context"] = batch_context
                result = self.qdii_estimator.estimate(**estimate_kwargs)

            elif fund_type == "active_a":
                estimate_kwargs = {
                    "fund_code": fund_code,
                    "last_nav": last_nav,
                    "nav_date": nav_date,
                    "target_date": target_date,
                    "proxy_components": active_proxy_components,
                    "strict": strict,
                }
                if portfolio_holdings is not None:
                    estimate_kwargs["holdings"] = portfolio_holdings
                estimate_kwargs.update(fee_kwargs)
                if batch_context is not None:
                    estimate_kwargs["batch_context"] = batch_context
                result = estimator.estimate(**estimate_kwargs)

            elif fund_type == "active_hk":
                estimate_kwargs = {
                    "fund_code": fund_code,
                    "last_nav": last_nav,
                    "nav_date": nav_date,
                    "target_date": target_date,
                    "is_index_fund": False,
                    "proxy_components": active_proxy_components,
                    "strict": strict,
                }
                if portfolio_holdings is not None:
                    estimate_kwargs["holdings"] = portfolio_holdings
                estimate_kwargs.update(fee_kwargs)
                if batch_context is not None:
                    estimate_kwargs["batch_context"] = batch_context
                result = self.qdii_estimator.estimate(**estimate_kwargs)

            elif fund_type == "bond_pure":
                estimate_kwargs = {
                    "fund_code": fund_code,
                    "last_nav": last_nav,
                    "nav_date": nav_date,
                    "target_date": target_date,
                    "fund_type": "pure",
                    "strict": strict,
                }
                estimate_kwargs.update(fee_kwargs)
                if batch_context is not None:
                    estimate_kwargs["batch_context"] = batch_context
                result = estimator.estimate(**estimate_kwargs)

            elif fund_type == "bond_plus":
                estimate_kwargs = {
                    "fund_code": fund_code,
                    "last_nav": last_nav,
                    "nav_date": nav_date,
                    "target_date": target_date,
                    "fund_type": "plus",
                    "strict": strict,
                }
                estimate_kwargs.update(fee_kwargs)
                if batch_context is not None:
                    estimate_kwargs["batch_context"] = batch_context
                result = estimator.estimate(**estimate_kwargs)

            elif fund_type == "qdii":
                if qdii_market_profile != "hk":
                    qdii_market_profile = self._validate_supported_qdii_profile(fund_code, fund_info)
                if qdii_is_index_fund is None:
                    is_index_fund = self.fund_fetcher.is_index_fund(fund_info)
                else:
                    is_index_fund = qdii_is_index_fund
                qdii_kwargs = {
                    "fund_code": fund_code,
                    "last_nav": last_nav,
                    "nav_date": nav_date,
                    "target_date": target_date,
                    "is_index_fund": is_index_fund,
                    "market_profile": qdii_market_profile,
                    "strict": strict,
                }
                qdii_kwargs.update(fee_kwargs)
                qdii_kwargs["proxy_components"] = active_proxy_components
                if not is_index_fund and portfolio_holdings is not None:
                    qdii_kwargs["holdings"] = portfolio_holdings
                if batch_context is not None:
                    qdii_kwargs["batch_context"] = batch_context
                result = self.qdii_estimator.estimate(**qdii_kwargs)

            else:
                raise ValueError(f"未知的基金类型: {fund_type}")

            result.setdefault("warnings", [])
            confidence_payload = build_confidence_payload(result, fund_type)
            result.update(
                {
                    "status": "成功",
                    "fund_type": fund_type,
                    "timestamp": datetime.now(),
                    "strict_mode": strict,
                }
            )
            result.update(confidence_payload)

            logger.info(f"基金 {fund_code} 估算完成")
            logger.info("=" * 60)
            return result
        except UnsupportedFundError as exc:
            logger.warning(f"基金 {fund_code} 当前版本不支持估值: {exc.unsupported_reason}")
            return self._build_failure_result(
                fund_code=fund_code,
                error=str(exc),
                target_date=target_date,
                strict=strict,
                fund_type=fund_type,
                unsupported_reason=exc.unsupported_reason,
                qdii_market_profile=exc.qdii_market_profile,
                quality_metadata=None,
            )

    def _collect_batch_requirements(self, preloaded_inputs: list[dict[str, Any]]) -> dict[str, Any]:
        requirements: dict[str, Any] = {
            "need_a_prices": False,
            "need_hk_prices": False,
            "need_fx": False,
            "required_a_codes": set(),
            "required_hk_codes": set(),
            "a_index_codes": set(),
            "hk_index_codes": set(),
            "bond_index_types": set(),
        }

        for item in preloaded_inputs:
            fund_type = item["fund_type"]
            fund_info = item["fund_info"]

            if fund_type in {"active_a", "active_hk"}:
                if fund_type == "active_hk":
                    requirements["need_fx"] = True

                try:
                    holdings_df = item.get("portfolio_holdings")
                    if holdings_df is None:
                        holdings_df = self.fund_fetcher.get_portfolio_holdings(item["fund_code"])
                    for _, row in holdings_df.iterrows():
                        stock_code = str(row.get("code", "")).strip()
                        if stock_code == "":
                            continue
                        market = str(row.get("market", "")).strip()
                        if market == "港股" or (len(stock_code) == 5 and stock_code.isdigit()):
                            requirements["required_hk_codes"].add(stock_code.zfill(5))
                            requirements["need_hk_prices"] = True
                            requirements["need_fx"] = True
                        elif market == "A股" or (len(stock_code) == 6 and stock_code.isdigit()):
                            requirements["required_a_codes"].add(stock_code)
                            requirements["need_a_prices"] = True
                except Exception as e:
                    logger.warning(f"基金 {item['fund_code']} 主动持仓预取失败，将回退全市场快照: {e}")
                    requirements["need_a_prices"] = True
                    requirements["need_hk_prices"] = True

                active_proxy_components = item.get("active_proxy_components")
                if active_proxy_components is None:
                    active_proxy_components = self.fund_fetcher.resolve_active_proxy_components(fund_info)

                for component in active_proxy_components:
                    component_code = str(component["code"])
                    component_market = str(component.get("market", "A股"))
                    if component_market == "港股":
                        requirements["hk_index_codes"].add(component_code)
                        requirements["need_fx"] = True
                    else:
                        requirements["a_index_codes"].add(component_code)

            elif fund_type == "index_a":
                try:
                    tracking_target = item.get("index_tracking_target")
                    if tracking_target is None:
                        tracking_target = self.fund_fetcher.resolve_index_tracking_target(fund_info)
                    target_type = str(tracking_target.get("target_type", "a_index"))
                    if target_type == "linked_etf_a_share":
                        security_code = str(tracking_target.get("security_code", "")).strip()
                        if security_code != "":
                            requirements["required_a_codes"].add(security_code)
                            requirements["need_a_prices"] = True
                    else:
                        code = str(tracking_target.get("code", "")).strip()
                        if code != "":
                            requirements["a_index_codes"].add(code)
                except Exception as e:
                    logger.warning(f"基金 {item['fund_code']} A股指数预解析失败，将在单基金阶段抛错: {e}")

            elif fund_type == "index_hk":
                requirements["need_fx"] = True
                try:
                    code = self.fund_fetcher.resolve_hk_index_code(fund_info)
                    requirements["hk_index_codes"].add(code)
                except Exception as e:
                    logger.warning(f"基金 {item['fund_code']} 港股指数预解析失败，将在单基金阶段抛错: {e}")

            elif fund_type == "bond_pure":
                requirements["bond_index_types"].add("comprehensive")

            elif fund_type == "bond_plus":
                requirements["bond_index_types"].add("comprehensive")
                requirements["a_index_codes"].add(PROXY_INDEX_MAP["沪深300"])
                requirements["a_index_codes"].add(BOND_PLUS_MODEL["convertible_index_code"])

            elif fund_type == "qdii":
                requirements["need_fx"] = True
                preloaded_qdii_flag = item.get("qdii_is_index_fund")
                if preloaded_qdii_flag is None:
                    try:
                        is_index_fund = self.fund_fetcher.is_index_fund(fund_info)
                    except Exception as e:
                        logger.warning(f"基金 {item['fund_code']} 指数属性预判失败，按主动基金预拉取港股代理指数: {e}")
                        is_index_fund = False
                else:
                    is_index_fund = bool(preloaded_qdii_flag)

                if is_index_fund:
                    proxy_components = item.get("active_proxy_components") or self.fund_fetcher.resolve_qdii_proxy_components(fund_info)
                    for component in proxy_components:
                        component_code = str(component["code"])
                        component_market = str(component.get("market", "港股"))
                        if component_market == "港股":
                            requirements["hk_index_codes"].add(component_code)
                        elif component_market == "A股":
                            requirements["a_index_codes"].add(component_code)
                else:
                    try:
                        holdings_df = item.get("portfolio_holdings")
                        if holdings_df is None:
                            holdings_df = self.fund_fetcher.get_portfolio_holdings(item["fund_code"])
                        for _, row in holdings_df.iterrows():
                            stock_code = str(row.get("code", "")).strip()
                            if stock_code == "":
                                continue
                            market = str(row.get("market", "")).strip()
                            if market == "港股" or (len(stock_code) == 5 and stock_code.isdigit()):
                                requirements["required_hk_codes"].add(stock_code.zfill(5))
                                requirements["need_hk_prices"] = True
                            elif market == "A股" or (len(stock_code) == 6 and stock_code.isdigit()):
                                requirements["required_a_codes"].add(stock_code)
                                requirements["need_a_prices"] = True
                    except Exception as e:
                        logger.warning(f"基金 {item['fund_code']} QDII主动持仓预取失败，将回退全市场快照: {e}")
                        requirements["need_a_prices"] = True
                        requirements["need_hk_prices"] = True

                    active_proxy_components = item.get("active_proxy_components")
                    if active_proxy_components is None:
                        active_proxy_components = self.fund_fetcher.resolve_qdii_proxy_components(fund_info)
                    for component in active_proxy_components:
                        component_code = str(component["code"])
                        component_market = str(component.get("market", "A股"))
                        if component_market == "港股":
                            requirements["hk_index_codes"].add(component_code)
                        elif component_market == "A股":
                            requirements["a_index_codes"].add(component_code)

        return requirements

    def _build_batch_context(self, preloaded_inputs: list[dict[str, Any]], strict: bool) -> BatchContext:
        batch_context = BatchContext()
        if strict:
            return batch_context

        requirements = self._collect_batch_requirements(preloaded_inputs)
        batch_context.required_a_codes = set(requirements["required_a_codes"])
        batch_context.required_hk_codes = set(requirements["required_hk_codes"])

        if not strict:
            if requirements["need_fx"]:
                try:
                    batch_context.hkd_cny_daily_change = self.fx_fetcher.get_hkd_cny_daily_change(strict=strict)
                    batch_context.fx_data_as_of_date = batch_context.data_as_of_date
                except Exception as e:
                    batch_context.fx_error = str(e)
                    logger.warning(f"批次汇率变动预取失败: {e}")
            return batch_context

        return batch_context

    def _get_estimator(self, fund_type: str):
        mapping = {
            "index_a": self.index_estimator,
            "index_hk": self.qdii_estimator,
            "active_a": self.equity_estimator,
            "active_hk": self.qdii_estimator,
            "bond_pure": self.bond_estimator,
            "bond_plus": self.bond_estimator,
            "qdii": self.qdii_estimator,
        }
        return mapping.get(fund_type, self.equity_estimator)

    @staticmethod
    def _normalize_fee_value(fee_value: Any, fee_field_name: str, fund_code: str) -> float | None:
        if fee_value is None:
            return None
        if isinstance(fee_value, str):
            normalized_value = fee_value.strip()
            if normalized_value == "":
                return None
            try:
                fee_float = float(normalized_value)
            except ValueError as exc:
                raise ValueError(f"基金 {fund_code} 的{fee_field_name}无法解析为数值: {fee_value}") from exc
        elif isinstance(fee_value, (int, float)):
            fee_float = float(fee_value)
        else:
            raise TypeError(f"基金 {fund_code} 的{fee_field_name}类型非法: {type(fee_value)}")

        if fee_float < 0 or fee_float > 1:
            raise ValueError(f"基金 {fund_code} 的{fee_field_name}超出合理范围[0,1]: {fee_float}")
        return fee_float

    @staticmethod
    def _validate_realtime_target_date(target_date: str | None) -> str | None:
        if target_date is None:
            return None
        target_dt = BaseEstimator.normalize_date(target_date, "target_date")
        today = date.today()
        if target_dt != today:
            raise ValueError(f"当前版本仅支持实时估值，target_date 仅可为 None 或当天 {today.isoformat()}；不支持历史/未来行情回放")
        return target_dt.isoformat()

    def _validate_supported_qdii_profile(self, fund_code: str, fund_info: dict[str, Any]) -> str:
        qdii_market_profile = self.fund_fetcher.resolve_qdii_market_profile(fund_info)
        if qdii_market_profile in {"hk", "us", "hk_us_mixed"}:
            return qdii_market_profile
        unsupported_reason = f"基金 {fund_code} 的QDII市场画像为 {qdii_market_profile}，当前版本仅支持港股、美股、港美混合质量优先 QDII"
        raise UnsupportedFundError(unsupported_reason, unsupported_reason=unsupported_reason, qdii_market_profile=qdii_market_profile)

    @staticmethod
    def _build_failure_result(
        fund_code: str,
        error: str,
        target_date: str | None,
        strict: bool,
        fund_type: str | None = None,
        unsupported_reason: str | None = None,
        qdii_market_profile: str | None = None,
        quality_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "fund_code": fund_code,
            "status": "失败",
            "error": error,
            "target_date": target_date,
            "timestamp": datetime.now(),
            "warnings": [],
            "strict_mode": strict,
        }
        if fund_type is not None:
            result["fund_type"] = fund_type
        if unsupported_reason is not None:
            result["unsupported_reason"] = unsupported_reason
        if qdii_market_profile is not None:
            result["qdii_market_profile"] = qdii_market_profile
        result.update(
            quality_metadata
            or build_quality_metadata(
                strict,
                passed=False,
                reasons=[error] if strict else [],
            )
        )
        return result
