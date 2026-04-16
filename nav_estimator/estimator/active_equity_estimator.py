"""主动偏股型基金估算器。"""

from datetime import date
from typing import Any

from loguru import logger

from ..config.settings import DEFAULT_FEES, DEFAULT_POSITION, PROXY_INDEX_MAP, STRICT_MODE
from ..core.batch_context import BatchContext
from ..core.quality_gate import QualityGateError, build_quality_metadata
from ..data.fetcher.fund_fetcher import FundFetcher
from ..data.fetcher.fx_fetcher import FXFetcher
from ..data.fetcher.index_fetcher import IndexFetcher
from ..data.fetcher.stock_fetcher import StockFetcher
from .base_estimator import BaseEstimator


class ActiveEquityEstimator(BaseEstimator):
    """主动管理偏股型基金净值估算器。"""

    def __init__(self):
        self.fund_fetcher = FundFetcher()
        self.stock_fetcher = StockFetcher()
        self.index_fetcher = IndexFetcher()
        self.fx_fetcher = FXFetcher()

    @staticmethod
    def _normalize_market_from_code(code: str, raw_market: str | None) -> str:
        market = str(raw_market or "").strip()
        if market in {"A股", "港股"}:
            return market
        normalized_code = str(code).strip()
        if len(normalized_code) == 5 and normalized_code.isdigit():
            return "港股"
        if len(normalized_code) == 6 and normalized_code.isdigit():
            return "A股"
        return "其他"

    @staticmethod
    def _normalize_proxy_components(
        proxy_components: list[dict[str, Any]] | None,
        proxy_index_code: str | None,
    ) -> list[dict[str, Any]]:
        if proxy_components is None or len(proxy_components) == 0:
            fallback_code = proxy_index_code or PROXY_INDEX_MAP["沪深300"]
            proxy_components = [{"code": str(fallback_code), "name": str(fallback_code), "market": "A股", "weight": 1.0}]

        normalized_components: list[dict[str, Any]] = []
        total_weight = 0.0
        for component in proxy_components:
            code = str(component.get("code", "")).strip()
            if code == "":
                raise ValueError("代理指数缺少code字段")
            raw_weight = component.get("weight", None)
            if raw_weight is None:
                raise ValueError(f"代理指数 {code} 缺少weight字段")
            weight = float(raw_weight)
            if weight <= 0:
                raise ValueError(f"代理指数 {code} 权重非法(<=0): {weight}")
            market = str(component.get("market", "")).strip()
            if market not in {"A股", "港股"}:
                market = "港股" if code in {"HSI", "HSTECH"} else "A股"
            normalized_components.append(
                {
                    "code": code,
                    "name": str(component.get("name", code)),
                    "market": market,
                    "weight": weight,
                    "target_type": str(component.get("target_type", "a_index")),
                    "quote_code": str(component.get("quote_code", code)),
                }
            )
            total_weight += weight

        if total_weight <= 0:
            raise ValueError("代理指数权重和非法(<=0)")
        for component in normalized_components:
            component["weight"] = component["weight"] / total_weight
        return normalized_components

    @staticmethod
    def _append_used_source(used_sources: dict[str, Any], category: str, source: str | None, item_code: str | None = None):
        if source is None:
            return
        bucket = used_sources.setdefault(category, {})
        if item_code is not None:
            bucket[str(item_code)] = source
            return
        bucket[source] = bucket.get(source, 0) + 1

    @staticmethod
    def _merge_live_source_meta(
        *,
        used_sources: dict[str, Any],
        source_disagreements: list[str],
        category: str,
        payload: dict[str, Any],
        item_code: str | None = None,
    ):
        ActiveEquityEstimator._append_used_source(used_sources, category, payload.get("source"), item_code=item_code)
        source_disagreements.extend(payload.get("source_disagreements", []))

    def _get_fx_payload(self, batch_context: BatchContext | None, strict: bool) -> dict[str, Any]:
        if strict:
            return self.fx_fetcher.get_hkd_cny_daily_change_live(strict=True)
        if batch_context is not None:
            if batch_context.hkd_cny_daily_change is not None:
                return {
                    "value": batch_context.hkd_cny_daily_change,
                    "data_as_of_date": batch_context.fx_data_as_of_date,
                    "source": "batch_prefetch",
                    "source_priority": 1,
                    "raw": {"change_pct": batch_context.hkd_cny_daily_change},
                    "warnings": [],
                    "source_disagreements": [],
                }
            if batch_context.fx_error is not None:
                logger.warning(f"批次汇率数据不可用，非严格模式下转为即时获取: {batch_context.fx_error}")
        fx_change = self.fx_fetcher.get_hkd_cny_daily_change(strict=False)
        return {
            "value": fx_change,
            "data_as_of_date": batch_context.fx_data_as_of_date if batch_context is not None else None,
            "source": "best_effort",
            "source_priority": 1,
            "raw": {"change_pct": fx_change},
            "warnings": [],
            "source_disagreements": [],
        }

    def _get_fx_return(self, batch_context: BatchContext | None, strict: bool) -> float:
        if batch_context is not None:
            if batch_context.hkd_cny_daily_change is not None:
                return batch_context.hkd_cny_daily_change / 100
            if batch_context.fx_error is not None:
                if not strict:
                    logger.warning(f"批次汇率数据不可用，非严格模式下转为即时获取: {batch_context.fx_error}")
                    return self.fx_fetcher.get_hkd_cny_daily_change(strict=strict) / 100
                raise ValueError(f"批次汇率数据不可用: {batch_context.fx_error}")
        return self.fx_fetcher.get_hkd_cny_daily_change(strict=strict) / 100

    def _get_index_return_by_market(
        self,
        index_code: str,
        market: str,
        batch_context: BatchContext | None,
        strict: bool,
    ) -> float:
        if market == "港股":
            if batch_context is not None:
                if index_code in batch_context.hk_index_returns:
                    return batch_context.hk_index_returns[index_code]
                if index_code in batch_context.hk_index_errors:
                    if not strict:
                        logger.warning(f"批次港股指数 {index_code} 不可用，非严格模式下转为即时获取: {batch_context.hk_index_errors[index_code]}")
                        return self.index_fetcher.get_hk_index_return(index_code)
                    raise ValueError(f"批次港股指数 {index_code} 不可用: {batch_context.hk_index_errors[index_code]}")
            return self.index_fetcher.get_hk_index_return(index_code)

        if batch_context is not None:
            if index_code in batch_context.a_index_returns:
                return batch_context.a_index_returns[index_code]
            if index_code in batch_context.a_index_errors:
                if not strict:
                    logger.warning(f"批次A股指数 {index_code} 不可用，非严格模式下转为即时获取: {batch_context.a_index_errors[index_code]}")
                    return self.index_fetcher.get_a_index_return(index_code)
                raise ValueError(f"批次A股指数 {index_code} 不可用: {batch_context.a_index_errors[index_code]}")
        return self.index_fetcher.get_a_index_return(index_code)

    def estimate(
        self,
        fund_code: str,
        last_nav: float,
        nav_date: str,
        target_date: str | None = None,
        report_date: str | None = None,
        holdings: Any | None = None,
        total_position: float | None = None,
        position_source_override: str | None = None,
        proxy_index_code: str | None = None,
        proxy_components: list[dict[str, Any]] | None = None,
        mgmt_rate: float | None = None,
        custody_rate: float | None = None,
        strict: bool | None = None,
        batch_context: BatchContext | None = None,
    ) -> dict[str, Any]:
        logger.info(f"开始估算主动偏股基金 {fund_code}")

        if strict is None:
            strict = STRICT_MODE

        normalized_proxy_components = self._normalize_proxy_components(proxy_components=proxy_components, proxy_index_code=proxy_index_code)
        if proxy_index_code is None:
            proxy_index_code = normalized_proxy_components[0]["code"] if len(normalized_proxy_components) == 1 else "MULTI"
        if mgmt_rate is None:
            mgmt_rate = DEFAULT_FEES["equity"]["management"]
        if custody_rate is None:
            custody_rate = DEFAULT_FEES["equity"]["custody"]

        nav_date_resolved, target_date_resolved, fee_days = self.resolve_fee_days(nav_date, target_date)
        data_as_of_date = batch_context.data_as_of_date if batch_context is not None else date.today().isoformat()
        warnings: list[str] = []
        used_sources: dict[str, Any] = {}
        source_disagreements: list[str] = []
        position_report_date: str | None = None
        position_raw_report_period: str | None = None
        position_snapshot_source: str | None = None
        estimated_position_series_latest: float | None = None

        try:
            holdings_df = holdings.copy() if holdings is not None else self.fund_fetcher.get_portfolio_holdings(fund_code, report_date)
            if holdings_df.empty:
                raise ValueError(f"未获取到基金 {fund_code} 的持仓数据")
        except Exception as e:
            logger.error(f"获取持仓数据失败: {e}")
            raise

        if total_position is None:
            try:
                stock_position_snapshot = self.fund_fetcher.get_fund_stock_position_snapshot(fund_code)
                total_position = float(stock_position_snapshot["position"])
                position_source = "fund_disclosure"
                position_report_date = stock_position_snapshot.get("report_date")
                position_raw_report_period = stock_position_snapshot.get("raw_report_period")
                position_snapshot_source = stock_position_snapshot.get("snapshot_source")
            except Exception as e:
                if strict:
                    raise QualityGateError(
                        ["缺少基金披露股票总仓位"],
                        used_sources={"stock_position": {"source": "fund_disclosure", "status": "missing"}},
                    ) from e
                try:
                    estimated_position_series_latest = self.fund_fetcher.get_estimated_stock_position_series_latest(fund_code)
                except Exception:
                    estimated_position_series_latest = None
                fallback_position, top10_weight_sum = self.fund_fetcher.build_position_fallback_from_holdings(
                    holdings_df=holdings_df,
                    default_position=DEFAULT_POSITION["equity_active"],
                )
                total_position = fallback_position
                position_source = "default_assumption_fallback"
                warning_msg = (
                    f"获取基金 {fund_code} 股票总仓位失败，已按前十大权重和 {top10_weight_sum:.2f}% "
                    f"与默认仓位 {DEFAULT_POSITION['equity_active']:.2f}% 回退为 {fallback_position:.2f}%: {e}"
                )
                logger.warning(warning_msg)
                warnings.append(warning_msg)
        else:
            position_source = position_source_override or "input_parameter"
        total_position = self.validate_percentage(total_position, "主动偏股基金总股票仓位")

        holdings_report_period, holdings_report_date = self.fund_fetcher.extract_holdings_report_metadata(holdings_df)
        holdings_df = holdings_df.copy()
        holdings_df["market"] = holdings_df.apply(lambda row: self._normalize_market_from_code(row["code"], row.get("market")), axis=1)

        required_a_codes = {str(code).strip() for code in holdings_df.loc[holdings_df["market"] == "A股", "code"].tolist() if str(code).strip() != ""}
        required_hk_codes = {str(code).strip().zfill(5) for code in holdings_df.loc[holdings_df["market"] == "港股", "code"].tolist() if str(code).strip() != ""}

        if strict:
            a_prices, failed_a_codes = self.stock_fetcher._get_a_share_quotes_live_by_codes_partial(required_a_codes, strict=True)
            hk_prices, failed_hk_codes = self.stock_fetcher._get_hk_share_quotes_live_by_codes_partial(required_hk_codes, strict=True)
            all_prices = {**a_prices, **hk_prices}
            for code, payload in a_prices.items():
                self._merge_live_source_meta(
                    used_sources=used_sources,
                    source_disagreements=source_disagreements,
                    category="a_share_quotes",
                    payload=payload,
                    item_code=code,
                )
            for code, payload in hk_prices.items():
                self._merge_live_source_meta(
                    used_sources=used_sources,
                    source_disagreements=source_disagreements,
                    category="hk_share_quotes",
                    payload=payload,
                    item_code=code,
                )
            if failed_a_codes:
                logger.warning(f"即时获取A股持仓行情失败，以下代码将按缺失行情处理: {failed_a_codes}")
            if failed_hk_codes:
                logger.warning(f"即时获取港股持仓行情失败，以下代码将按缺失行情处理: {failed_hk_codes}")
        elif batch_context is not None:
            a_prices = {
                code: batch_context.all_prices[code]
                for code in required_a_codes
                if code in batch_context.all_prices
            }
            hk_prices = {
                code: batch_context.all_prices[code]
                for code in required_hk_codes
                if code in batch_context.all_prices
            }

            missing_a_codes = required_a_codes - set(a_prices.keys())
            missing_hk_codes = required_hk_codes - set(hk_prices.keys())

            if required_a_codes and batch_context.a_prices_error is not None:
                logger.warning(f"批次A股快照不可用，非严格模式下转为即时获取: {batch_context.a_prices_error}")
            if required_hk_codes and batch_context.hk_prices_error is not None:
                logger.warning(f"批次港股快照不可用，非严格模式下转为即时获取: {batch_context.hk_prices_error}")

            if missing_a_codes:
                fresh_a_prices, failed_a_codes = self.stock_fetcher._get_a_share_prices_by_codes_partial(missing_a_codes)
                a_prices.update(fresh_a_prices)
                if failed_a_codes:
                    logger.warning(f"即时获取A股持仓行情失败，以下代码将按缺失行情处理: {failed_a_codes}")

            if missing_hk_codes:
                fresh_hk_prices, failed_hk_codes = self.stock_fetcher._get_hk_prices_by_codes_partial(missing_hk_codes)
                hk_prices.update(fresh_hk_prices)
                if failed_hk_codes:
                    logger.warning(f"即时获取港股持仓行情失败，以下代码将按缺失行情处理: {failed_hk_codes}")

            all_prices = {**a_prices, **hk_prices}
        else:
            if required_a_codes:
                try:
                    a_prices = self.stock_fetcher.get_a_share_prices_by_codes(required_a_codes)
                except Exception as e:
                    logger.warning(f"按代码获取A股行情失败，非严格模式下继续使用部分可用行情: {e}")
                    a_prices, failed_a_codes = self.stock_fetcher._get_a_share_prices_by_codes_partial(required_a_codes)
                    if failed_a_codes:
                        logger.warning(f"A股持仓行情缺失代码: {failed_a_codes}")
            else:
                a_prices = {}
            if required_hk_codes:
                try:
                    hk_prices = self.stock_fetcher.get_hk_prices_by_codes(required_hk_codes)
                except Exception as e:
                    logger.warning(f"按代码获取港股行情失败，非严格模式下继续使用部分可用行情: {e}")
                    hk_prices, failed_hk_codes = self.stock_fetcher._get_hk_prices_by_codes_partial(required_hk_codes)
                    if failed_hk_codes:
                        logger.warning(f"港股持仓行情缺失代码: {failed_hk_codes}")
            else:
                hk_prices = {}
            all_prices = {**a_prices, **hk_prices}

        has_hk_holdings = len(required_hk_codes) > 0
        has_hk_proxy = any(item["market"] == "港股" for item in normalized_proxy_components)
        need_fx = has_hk_holdings or has_hk_proxy
        fx_payload = self._get_fx_payload(batch_context=batch_context, strict=strict) if need_fx else None
        if fx_payload is not None and strict:
            self._merge_live_source_meta(
                used_sources=used_sources,
                source_disagreements=source_disagreements,
                category="fx",
                payload=fx_payload,
            )
        r_fx = (float(fx_payload["value"]) / 100) if fx_payload is not None else 0.0

        top10_weighted_return = 0.0
        top10_total_weight = 0.0
        top10_weight_known = 0.0
        holdings_details: list[dict[str, Any]] = []
        missing_quotes: list[str] = []
        stock_data_dates: list[str] = []

        for _, stock in holdings_df.iterrows():
            code = str(stock["code"]).strip()
            market = str(stock["market"])
            if market == "港股":
                code = code.zfill(5)

            weight = self.validate_holding_weight_percentage(stock["weight"], f"持仓权重-{code}")
            name = stock["name"]
            top10_total_weight += weight

            if code in all_prices:
                price_data = all_prices[code]
                quote_raw = price_data["raw"] if strict else price_data
                change_pct = float(quote_raw["change_pct"])
                top10_weight_known += weight
                if strict:
                    if price_data.get("data_as_of_date") is not None:
                        stock_data_dates.append(str(price_data["data_as_of_date"]))
                elif "data_as_of_date" in price_data:
                    stock_data_dates.append(str(price_data["data_as_of_date"]))
            else:
                missing_quotes.append(code)
                warning_msg = f"持仓股票 {code} 缺少实时行情"
                warnings.append(warning_msg)
                change_pct = 0.0

            local_return = change_pct / 100
            if market == "港股":
                stock_rmb_return = (1 + local_return) * (1 + r_fx) - 1
                fx_return_pct = r_fx * 100
            else:
                stock_rmb_return = local_return
                fx_return_pct = 0.0

            contribution = (weight / 100) * stock_rmb_return
            top10_weighted_return += contribution
            holdings_details.append(
                {
                    "code": code,
                    "name": name,
                    "market": market,
                    "weight": weight,
                    "change_pct": change_pct,
                    "local_return_pct": change_pct,
                    "fx_return_pct": fx_return_pct,
                    "rmb_return_pct": stock_rmb_return * 100,
                    "contribution": contribution * 100,
                }
            )

        known_weight_sum = top10_total_weight
        if known_weight_sum > total_position + 1e-6:
            if strict:
                raise ValueError(f"基金 {fund_code} 持仓权重之和({known_weight_sum:.2f}%)超过总股票仓位({total_position:.2f}%)")
            warning_msg = f"基金 {fund_code} 持仓权重之和({known_weight_sum:.2f}%)超过总股票仓位({total_position:.2f}%)，非严格模式下将总仓位提升至 {known_weight_sum:.2f}%"
            logger.warning(warning_msg)
            warnings.append(warning_msg)
            total_position = known_weight_sum
            position_source = "adjusted_by_top10_weight"

        non_top10_weight = max(total_position - top10_total_weight, 0.0)
        missing_quotes_weight = top10_total_weight - top10_weight_known
        proxy_weight = non_top10_weight + missing_quotes_weight
        proxy_components_details: list[dict[str, Any]] = []
        index_data_dates: list[str] = []
        proxy_data_available = True
        proxy_error_message: str | None = None

        if proxy_weight > 0:
            try:
                proxy_contribution = 0.0
                for component in normalized_proxy_components:
                    component_code = component["code"]
                    component_quote_code = str(component.get("quote_code", component_code))
                    component_name = component["name"]
                    component_market = component["market"]
                    component_target_type = str(component.get("target_type", "a_index"))
                    component_ratio = float(component["weight"])
                    component_weight = proxy_weight * component_ratio
                    if strict:
                        if component_market == "港股":
                            component_payload = self.index_fetcher.get_hk_index_return_live(component_code, strict=True)
                            self._merge_live_source_meta(
                                used_sources=used_sources,
                                source_disagreements=source_disagreements,
                                category="proxy_indices",
                                payload=component_payload,
                                item_code=component_code,
                            )
                        elif component_target_type == "a_share_etf_proxy":
                            component_payload = self.stock_fetcher.get_a_share_quote_live(component_quote_code, strict=True)
                            self._merge_live_source_meta(
                                used_sources=used_sources,
                                source_disagreements=source_disagreements,
                                category="proxy_tracking_targets",
                                payload=component_payload,
                                item_code=component_quote_code,
                            )
                        else:
                            component_payload = self.index_fetcher.get_a_index_return_live(component_code, strict=True)
                            self._merge_live_source_meta(
                                used_sources=used_sources,
                                source_disagreements=source_disagreements,
                                category="proxy_indices",
                                payload=component_payload,
                                item_code=component_code,
                            )
                        component_local_return_pct = float(component_payload["value"])
                        if component_payload.get("data_as_of_date") is not None:
                            index_data_dates.append(str(component_payload["data_as_of_date"]))
                    else:
                        if component_market == "港股":
                            component_local_return_pct = self._get_index_return_by_market(
                                index_code=component_code,
                                market=component_market,
                                batch_context=batch_context,
                                strict=strict,
                            )
                            if batch_context is not None:
                                component_as_of_date = batch_context.hk_index_as_of_dates.get(component_code)
                                if component_as_of_date is not None:
                                    index_data_dates.append(str(component_as_of_date))
                        elif component_target_type == "a_share_etf_proxy":
                            if batch_context is not None and component_quote_code in batch_context.all_prices:
                                component_local_return_pct = float(batch_context.all_prices[component_quote_code]["change_pct"])
                                component_as_of_date = batch_context.a_price_as_of_dates.get(component_quote_code)
                                if component_as_of_date is not None:
                                    index_data_dates.append(str(component_as_of_date))
                            else:
                                component_payload = self.stock_fetcher.get_a_share_quote_live(component_quote_code, strict=False)
                                component_local_return_pct = float(component_payload["value"])
                                if component_payload.get("data_as_of_date") is not None:
                                    index_data_dates.append(str(component_payload["data_as_of_date"]))
                        else:
                            component_local_return_pct = self._get_index_return_by_market(
                                index_code=component_code,
                                market=component_market,
                                batch_context=batch_context,
                                strict=strict,
                            )
                            if batch_context is not None:
                                component_as_of_date = batch_context.a_index_as_of_dates.get(component_code)
                                if component_as_of_date is not None:
                                    index_data_dates.append(str(component_as_of_date))
                    component_local_return = component_local_return_pct / 100
                    if component_market == "港股":
                        component_rmb_return = (1 + component_local_return) * (1 + r_fx) - 1
                        component_fx_return_pct = r_fx * 100
                    else:
                        component_rmb_return = component_local_return
                        component_fx_return_pct = 0.0

                    component_contribution = (component_weight / 100) * component_rmb_return
                    proxy_contribution += component_contribution
                    proxy_components_details.append(
                        {
                            "code": component_code,
                            "quote_code": component_quote_code,
                            "name": component_name,
                            "market": component_market,
                            "target_type": component_target_type,
                            "weight": component_ratio,
                            "proxy_weight": component_weight,
                            "local_return_pct": component_local_return_pct,
                            "fx_return_pct": component_fx_return_pct,
                            "rmb_return_pct": component_rmb_return * 100,
                            "contribution": component_contribution * 100,
                        }
                    )

                proxy_return = proxy_contribution / (proxy_weight / 100) * 100 if proxy_weight > 1e-12 else 0.0
            except Exception as e:
                proxy_data_available = False
                proxy_error_message = str(e)
                if strict:
                    warnings.append(f"获取代理指数失败: {e}")
                    proxy_return = 0.0
                    proxy_contribution = 0.0
                    proxy_components_details = []
                else:
                    warning_msg = f"获取代理指数失败，剩余仓位收益设为0: {e}"
                    logger.warning(warning_msg)
                    warnings.append(warning_msg)
                    proxy_return = 0.0
                    proxy_contribution = 0.0
                    proxy_components_details = []
        else:
            proxy_return = 0.0
            proxy_contribution = 0.0

        f_t = self.daily_fee_drag(mgmt_rate, custody_rate, nav_date=nav_date_resolved, target_date=target_date_resolved)
        r_t = top10_weighted_return + proxy_contribution - f_t
        top10_quote_coverage = 0.0 if top10_total_weight <= 1e-12 else top10_weight_known / top10_total_weight
        live_quote_weight_pct = 0.0 if total_position <= 1e-12 else top10_weight_known / total_position * 100
        stock_data_as_of_date = min(stock_data_dates) if stock_data_dates else data_as_of_date
        index_data_as_of_date = min(index_data_dates) if index_data_dates else (data_as_of_date if proxy_weight > 0 else None)
        est_nav = self.estimate_nav(last_nav, r_t)

        if strict:
            quality_gate_failed_reasons: list[str] = []
            if position_source != "fund_disclosure":
                quality_gate_failed_reasons.append("缺少基金披露股票总仓位")
            if top10_total_weight > 0 and top10_quote_coverage < 0.5 and not proxy_data_available:
                quality_gate_failed_reasons.append("前十大实时行情覆盖不足且代理指数不可用")
            if proxy_weight > 20 and not proxy_data_available:
                quality_gate_failed_reasons.append("未覆盖仓位超过20%且代理指数不可用")
            if quality_gate_failed_reasons:
                raise QualityGateError(
                    quality_gate_failed_reasons,
                    used_sources=used_sources,
                    source_disagreements=source_disagreements,
                    live_quote_coverage_pct=top10_quote_coverage * 100,
                    live_quote_weight_pct=live_quote_weight_pct,
                )
        quality_metadata = build_quality_metadata(
            strict,
            passed=True,
            used_sources=used_sources,
            source_disagreements=source_disagreements,
            live_quote_coverage_pct=top10_quote_coverage * 100,
            live_quote_weight_pct=live_quote_weight_pct,
        )

        result = {
            "fund_code": fund_code,
            "last_nav": last_nav,
            "nav_date": nav_date_resolved,
            "target_date": target_date_resolved,
            "fee_days": fee_days,
            "estimated_nav": est_nav,
            "estimated_return": r_t * 100,
            "top10_weight": top10_total_weight,
            "top10_weight_known": top10_weight_known,
            "top10_quote_coverage": top10_quote_coverage,
            "total_position": total_position,
            "remaining_weight": non_top10_weight,
            "proxy_weight": proxy_weight,
            "proxy_index_code": proxy_index_code,
            "proxy_return": proxy_return,
            "proxy_components": proxy_components_details,
            "position_source": position_source,
            "position_snapshot_source": position_snapshot_source,
            "position_report_date": position_report_date,
            "position_raw_report_period": position_raw_report_period,
            "estimated_position_series_latest": estimated_position_series_latest,
            "stock_data_as_of_date": stock_data_as_of_date,
            "index_data_as_of_date": index_data_as_of_date,
            "fx_return": r_fx * 100 if need_fx else 0.0,
            "holdings_report_period": holdings_report_period,
            "holdings_report_date": holdings_report_date,
            "fx_data_as_of_date": fx_payload.get("data_as_of_date") if fx_payload is not None else None,
            "holdings_details": holdings_details,
            "fee_drag": f_t * 100,
            "missing_quotes": missing_quotes,
            "warnings": warnings,
            "strict_mode": strict,
        }
        result.update(quality_metadata)

        logger.info(
            f"估算完成: {fund_code}, 净值日期: {nav_date_resolved}, 估算目标日: {target_date_resolved}, 费率天数: {fee_days}, 前十大权重: {top10_total_weight:.2f}%, 估算涨跌: {r_t*100:+.2f}%"
        )
        return result
