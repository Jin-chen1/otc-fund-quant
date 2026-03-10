"""指数型基金估算器。"""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..config.settings import DEFAULT_FEES, DEFAULT_POSITION, INDEX_ENHANCED_REGRESSION, STRICT_MODE
from ..core.batch_context import BatchContext
from ..core.quality_gate import build_quality_metadata
from ..data.fetcher.fund_fetcher import FundFetcher
from ..data.fetcher.index_fetcher import IndexFetcher
from ..data.fetcher.stock_fetcher import StockFetcher
from .base_estimator import BaseEstimator


class IndexEstimator(BaseEstimator):
    """指数型基金净值估算器。"""

    def __init__(self):
        self.index_fetcher = IndexFetcher()
        self.fund_fetcher = FundFetcher()
        self.stock_fetcher = StockFetcher()

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
        IndexEstimator._append_used_source(used_sources, category, payload.get("source"), item_code=item_code)
        source_disagreements.extend(payload.get("source_disagreements", []))

    @staticmethod
    def _build_batch_tracking_payload(change_pct: float, data_as_of_date: str | None) -> dict[str, Any]:
        return {
            "value": float(change_pct),
            "source": "batch_prefetch",
            "source_priority": 1,
            "raw": {"change_pct": float(change_pct)},
            "warnings": [],
            "source_disagreements": [],
            "data_as_of_date": data_as_of_date or date.today().isoformat(),
        }

    @staticmethod
    def _resolve_tracking_target_payload(index_code: str | None, tracking_target: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(tracking_target or {})
        target_type = str(payload.get("target_type", "a_index")).strip() or "a_index"
        resolved_code = str(payload.get("code") or payload.get("security_code") or index_code or "").strip()
        if resolved_code == "":
            raise ValueError("指数基金缺少有效跟踪标的代码")
        payload["target_type"] = target_type
        payload["resolved_code"] = resolved_code
        return payload

    def _resolve_default_position(self, fund_code: str, fund_info: dict[str, Any] | None) -> tuple[float, str, dict[str, Any]]:
        resolved_fund_info = fund_info
        if resolved_fund_info is None:
            resolved_fund_info = self.fund_fetcher.get_fund_info(fund_code)
        if self.fund_fetcher.is_etf_or_linked_fund(resolved_fund_info):
            return DEFAULT_POSITION["index_etf"], "default_assumption_etf", resolved_fund_info
        return DEFAULT_POSITION["index_regular"], "default_assumption_regular", resolved_fund_info

    def _build_regression_dataset(
        self,
        fund_code: str,
        index_code: str,
        lookback_days: int,
        mgmt_rate: float,
        custody_rate: float,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        nav_series_df = self.fund_fetcher.get_fund_nav_series(
            fund_code=fund_code,
            lookback_days=lookback_days,
            end_date=end_date,
        )
        if len(nav_series_df) < 2:
            raise ValueError(f"基金 {fund_code} 净值样本不足，无法构建回归数据集")

        interval_fee_drag: list[float] = [np.nan]
        nav_dates = nav_series_df["date"].tolist()
        for prev_date, curr_date in zip(nav_dates[:-1], nav_dates[1:]):
            fee_drag_pct = self.daily_fee_drag(mgmt_rate, custody_rate, nav_date=prev_date, target_date=curr_date) * 100
            interval_fee_drag.append(fee_drag_pct)

        nav_series_df["fund_return_net"] = nav_series_df["nav"].pct_change() * 100
        nav_series_df["interval_fee_drag"] = interval_fee_drag
        nav_series_df["fund_return"] = nav_series_df["fund_return_net"] + nav_series_df["interval_fee_drag"]
        fund_returns_df = nav_series_df.dropna(subset=["fund_return"])[["date", "fund_return"]].copy()
        index_returns_df = self.index_fetcher.get_a_index_return_series(index_code=index_code, lookback_days=lookback_days, end_date=end_date)

        merged_df = pd.merge(fund_returns_df, index_returns_df, on="date", how="inner")
        if merged_df.empty:
            raise ValueError(f"基金 {fund_code} 与指数 {index_code} 的收益序列无重叠交易日，无法回归")

        merged_df["fund_return"] = pd.to_numeric(merged_df["fund_return"], errors="coerce")
        merged_df["index_return"] = pd.to_numeric(merged_df["index_return"], errors="coerce")
        merged_df = merged_df.dropna(subset=["fund_return", "index_return"])
        if merged_df.empty:
            raise ValueError(f"基金 {fund_code} 与指数 {index_code} 的收益序列存在非法值，无法回归")

        return merged_df.reset_index(drop=True)

    @staticmethod
    def _fit_alpha_beta(merged_df: pd.DataFrame) -> tuple[float, float]:
        x = merged_df["index_return"].to_numpy(dtype=float)
        y = merged_df["fund_return"].to_numpy(dtype=float)
        if len(x) < 2:
            raise ValueError("回归样本数不足，至少需要2个样本")
        if float(np.var(x)) <= 1e-12:
            raise ValueError("指数收益序列方差过小，无法进行稳定回归")

        design_matrix = np.column_stack((np.ones(len(x)), x))
        coefficients, *_ = np.linalg.lstsq(design_matrix, y, rcond=None)
        return float(coefficients[0]), float(coefficients[1])

    def estimate(
        self,
        fund_code: str,
        last_nav: float,
        nav_date: str,
        index_code: str,
        target_date: str | None = None,
        fund_info: dict[str, Any] | None = None,
        tracking_target: dict[str, Any] | None = None,
        position: float | None = None,
        mgmt_rate: float | None = None,
        custody_rate: float | None = None,
        strict: bool | None = None,
        batch_context: BatchContext | None = None,
    ) -> dict[str, Any]:
        logger.info(f"开始估算指数基金 {fund_code}")
        if strict is None:
            strict = STRICT_MODE

        warnings: list[str] = []
        used_sources: dict[str, Any] = {}
        source_disagreements: list[str] = []
        resolved_tracking_target = self._resolve_tracking_target_payload(index_code, tracking_target)
        tracking_target_type = str(resolved_tracking_target["target_type"])
        tracking_code = str(resolved_tracking_target["resolved_code"])
        if mgmt_rate is None:
            mgmt_rate = DEFAULT_FEES["index"]["management"]
        if custody_rate is None:
            custody_rate = DEFAULT_FEES["index"]["custody"]

        input_position_provided = position is not None
        position_source = "input_parameter"
        if not input_position_provided:
            try:
                position, position_source, fund_info = self._resolve_default_position(fund_code=fund_code, fund_info=fund_info)
            except Exception as exc:
                raise ValueError(f"基金 {fund_code} 默认仓位判别失败: {exc}") from exc

        is_index_enhanced_fund = False
        regression_applied = False
        regression_samples: int | None = None
        regression_alpha: float | None = None
        regression_beta: float | None = None
        alpha_adjustment_pct = 0.0
        position_before_regression: float | None = None

        if not input_position_provided:
            if fund_info is None:
                try:
                    fund_info = self.fund_fetcher.get_fund_info(fund_code)
                except Exception as exc:
                    if strict:
                        raise ValueError(f"基金 {fund_code} 获取基础信息失败，无法判断指数增强属性: {exc}") from exc
                    warning_msg = f"基金 {fund_code} 获取基础信息失败，跳过指数增强校准: {exc}"
                    logger.warning(warning_msg)
                    warnings.append(warning_msg)

            if fund_info is not None:
                is_index_enhanced_fund = self.fund_fetcher.is_index_enhanced_fund(fund_info)

            if is_index_enhanced_fund:
                if tracking_target_type != "a_index":
                    raise ValueError(f"基金 {fund_code} 的联接ETF跟踪标的不支持指数增强回归校准")
                lookback_days = int(INDEX_ENHANCED_REGRESSION["lookback_days"])
                min_samples = int(INDEX_ENHANCED_REGRESSION["min_samples"])
                beta_min = float(INDEX_ENHANCED_REGRESSION["beta_min"])
                beta_max = float(INDEX_ENHANCED_REGRESSION["beta_max"])

                try:
                    regression_df = self._build_regression_dataset(
                        fund_code=fund_code,
                        index_code=tracking_code,
                        lookback_days=lookback_days,
                        mgmt_rate=mgmt_rate,
                        custody_rate=custody_rate,
                        end_date=nav_date,
                    )
                    regression_samples = len(regression_df)
                    if regression_samples < min_samples:
                        raise ValueError(f"回归样本数不足: {regression_samples} < {min_samples}")

                    raw_alpha, raw_beta = self._fit_alpha_beta(regression_df)
                    clipped_beta = min(max(raw_beta, beta_min), beta_max)
                    if abs(clipped_beta - raw_beta) > 1e-9:
                        warning_msg = f"基金 {fund_code} 回归beta={raw_beta:.4f} 超出区间[{beta_min:.2f}, {beta_max:.2f}]，已截断为 {clipped_beta:.4f}"
                        logger.warning(warning_msg)
                        warnings.append(warning_msg)

                    position_before_regression = position
                    position = clipped_beta * 100
                    position_source = "regression_beta_calibrated"
                    regression_applied = True
                    regression_alpha = raw_alpha
                    regression_beta = clipped_beta
                    alpha_adjustment_pct = raw_alpha
                except Exception as exc:
                    if strict:
                        raise ValueError(f"基金 {fund_code} 指数增强回归校准失败: {exc}") from exc
                    warning_msg = f"基金 {fund_code} 指数增强回归校准失败，沿用默认仓位: {exc}"
                    logger.warning(warning_msg)
                    warnings.append(warning_msg)

        position = self.validate_percentage(position, "指数基金仓位")
        if position < 1:
            raise ValueError(f"指数基金仓位非法: {position}，应在[1, 100]区间内")

        nav_date_resolved, target_date_resolved, fee_days = self.resolve_fee_days(nav_date, target_date)
        index_data_as_of_date = (
            (
                (
                    batch_context.a_index_as_of_dates.get(tracking_code)
                    if tracking_target_type == "a_index" and tracking_code in batch_context.a_index_as_of_dates
                    else batch_context.a_price_as_of_dates.get(tracking_code)
                    if tracking_target_type == "linked_etf_a_share" and tracking_code in batch_context.a_price_as_of_dates
                    else batch_context.data_as_of_date
                )
                if batch_context is not None
                else None
            )
            if batch_context is not None
            else date.today().isoformat()
        )

        try:
            if tracking_target_type == "linked_etf_a_share":
                if strict:
                    tracking_payload = self.stock_fetcher.get_a_share_quote_live(tracking_code, strict=True)
                elif batch_context is not None and tracking_code in batch_context.a_prices:
                    tracking_payload = self._build_batch_tracking_payload(
                        batch_context.a_prices[tracking_code]["change_pct"],
                        batch_context.a_price_as_of_dates.get(tracking_code) or batch_context.data_as_of_date,
                    )
                else:
                    if batch_context is not None and batch_context.a_prices_error is not None:
                        logger.warning(
                            f"批次联接ETF {tracking_code} 行情不可用，非严格模式下转为即时获取: {batch_context.a_prices_error}"
                        )
                    tracking_payload = self.stock_fetcher.get_a_share_quote_live(tracking_code, strict=False)
                self._merge_live_source_meta(
                    used_sources=used_sources,
                    source_disagreements=source_disagreements,
                    category="tracking_targets",
                    payload=tracking_payload,
                    item_code=tracking_code,
                )
                r_index = float(tracking_payload["value"])
                index_data_as_of_date = tracking_payload.get("data_as_of_date") or index_data_as_of_date
            else:
                if strict:
                    index_payload = self.index_fetcher.get_a_index_return_live(tracking_code, strict=True)
                elif batch_context is not None and tracking_code in batch_context.a_index_returns:
                    index_payload = self._build_batch_tracking_payload(
                        batch_context.a_index_returns[tracking_code],
                        batch_context.a_index_as_of_dates.get(tracking_code) or batch_context.data_as_of_date,
                    )
                else:
                    if batch_context is not None and tracking_code in batch_context.a_index_errors:
                        logger.warning(
                            f"批次A股指数 {tracking_code} 不可用，非严格模式下转为即时获取: {batch_context.a_index_errors[tracking_code]}"
                        )
                    best_effort_return = self.index_fetcher.get_a_index_return(tracking_code)
                    index_payload = {
                        "value": best_effort_return,
                        "source": "best_effort",
                        "source_priority": 1,
                        "raw": {"change_pct": best_effort_return},
                        "warnings": [],
                        "source_disagreements": [],
                        "data_as_of_date": index_data_as_of_date,
                    }
                self._merge_live_source_meta(
                    used_sources=used_sources,
                    source_disagreements=source_disagreements,
                    category="tracking_targets",
                    payload=index_payload,
                    item_code=tracking_code,
                )
                r_index = float(index_payload["value"])
                index_data_as_of_date = index_payload.get("data_as_of_date") or index_data_as_of_date
        except Exception as e:
            logger.error(f"获取指数行情失败: {e}")
            raise

        f_t = self.daily_fee_drag(mgmt_rate, custody_rate, nav_date=nav_date_resolved, target_date=target_date_resolved)
        r_t = (r_index / 100) * (position / 100) + (alpha_adjustment_pct / 100) - f_t
        est_nav = self.estimate_nav(last_nav, r_t)

        result = {
            "fund_code": fund_code,
            "last_nav": last_nav,
            "nav_date": nav_date_resolved,
            "target_date": target_date_resolved,
            "fee_days": fee_days,
            "estimated_nav": est_nav,
            "estimated_return": r_t * 100,
            "index_code": tracking_code,
            "index_return": r_index,
            "position": position,
            "position_source": position_source,
            "is_index_enhanced_fund": is_index_enhanced_fund,
            "regression_applied": regression_applied,
            "regression_samples": regression_samples,
            "regression_alpha": regression_alpha,
            "regression_alpha_basis": "pre_fee" if regression_alpha is not None else None,
            "regression_beta": regression_beta,
            "position_before_regression": position_before_regression,
            "alpha_adjustment": alpha_adjustment_pct,
            "alpha_adjustment_basis": "pre_fee" if regression_applied else None,
            "index_data_as_of_date": index_data_as_of_date,
            "fee_drag": f_t * 100,
            "warnings": warnings,
            "strict_mode": strict,
        }
        result.update(
            build_quality_metadata(
                strict,
                passed=True,
                used_sources=used_sources,
                source_disagreements=source_disagreements,
            )
        )

        logger.info(
            f"估算完成: {fund_code}, 昨日净值: {last_nav:.4f}({nav_date_resolved}), 估算目标日: {target_date_resolved}, 费率天数: {fee_days}, 估算净值: {est_nav:.4f}, 估算涨跌: {r_t*100:+.2f}%"
        )
        return result
