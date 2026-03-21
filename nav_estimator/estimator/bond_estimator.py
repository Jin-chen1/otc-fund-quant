"""债券型基金估算器。"""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..config.settings import (
    BOND_PLUS_MODEL,
    BOND_STALE_FALLBACK,
    DEFAULT_FEES,
    DEFAULT_POSITION,
    PROXY_INDEX_MAP,
    STRICT_MODE,
)
from ..core.batch_context import BatchContext
from ..core.quality_gate import build_quality_metadata
from ..data.fetcher.bond_fetcher import BondFetcher
from ..data.fetcher.fund_fetcher import FundFetcher
from ..data.fetcher.index_fetcher import IndexFetcher
from .base_estimator import BaseEstimator


class BondEstimator(BaseEstimator):
    """债券型基金净值估算器。"""

    def __init__(self):
        self.bond_fetcher = BondFetcher()
        self.index_fetcher = IndexFetcher()
        self.fund_fetcher = FundFetcher()

    def _get_bond_snapshot(self, bond_index_type: str, batch_context: BatchContext | None, strict: bool) -> dict[str, Any]:
        if batch_context is not None:
            if bond_index_type in batch_context.bond_index_snapshots:
                return batch_context.bond_index_snapshots[bond_index_type]
            if bond_index_type in batch_context.bond_index_errors:
                if not strict:
                    logger.warning(f"批次债券指数 {bond_index_type} 不可用，非严格模式下转为即时获取: {batch_context.bond_index_errors[bond_index_type]}")
                    return self.bond_fetcher.get_bond_index_snapshot(index_type=bond_index_type)
                raise ValueError(f"批次债券指数 {bond_index_type} 不可用: {batch_context.bond_index_errors[bond_index_type]}")
            if bond_index_type in batch_context.bond_index_returns:
                return {
                    "index_type": bond_index_type,
                    "index_name": bond_index_type,
                    "prev_date": None,
                    "curr_date": None,
                    "prev_close": None,
                    "curr_close": None,
                    "change_pct": float(batch_context.bond_index_returns[bond_index_type]),
                }
        return self.bond_fetcher.get_bond_index_snapshot(index_type=bond_index_type)

    def _get_a_index_return(self, index_code: str, batch_context: BatchContext | None, strict: bool) -> float:
        if batch_context is not None:
            if index_code in batch_context.a_index_returns:
                return float(batch_context.a_index_returns[index_code])
            if index_code in batch_context.a_index_errors:
                if not strict:
                    logger.warning(f"批次A股指数 {index_code} 不可用，非严格模式下转为即时获取: {batch_context.a_index_errors[index_code]}")
                    return float(self.index_fetcher.get_a_index_return(index_code))
                raise ValueError(f"批次A股指数 {index_code} 不可用: {batch_context.a_index_errors[index_code]}")
        return float(self.index_fetcher.get_a_index_return(index_code))

    def _build_stale_fallback_bond_return(self, warnings: list[str]) -> tuple[float, dict[str, Any]]:
        duration = float(BOND_STALE_FALLBACK["duration"])
        tenor = str(BOND_STALE_FALLBACK["yield_tenor"])
        default_ytm = float(BOND_STALE_FALLBACK["default_ytm"])
        try:
            ytm_snapshot = self.bond_fetcher.get_treasury_yield_snapshot(tenor=tenor)
            curr_ytm = float(ytm_snapshot["curr_yield"])
            delta_yield = float(ytm_snapshot["delta_yield"])
            ytm_as_of_date = str(ytm_snapshot["curr_date"])
        except Exception as exc:
            warning_msg = f"国债收益率代理获取失败，非严格模式下使用默认到期收益率计算票息应计，不叠加利率代理: {exc}"
            logger.warning(warning_msg)
            warnings.append(warning_msg)
            curr_ytm = default_ytm
            delta_yield = 0.0
            ytm_as_of_date = None

        accrual_pct = self.bond_fetcher.get_daily_accrual(curr_ytm, duration=duration)
        rate_proxy_pct = -duration * delta_yield
        fallback_return = accrual_pct + rate_proxy_pct
        detail = {
            "tenor": tenor,
            "duration": duration,
            "ytm": curr_ytm,
            "ytm_as_of_date": ytm_as_of_date,
            "delta_yield": delta_yield,
            "accrual_pct": accrual_pct,
            "rate_proxy_pct": rate_proxy_pct,
            "fallback_return_pct": fallback_return,
        }
        return fallback_return, detail

    def _fit_bond_plus_three_factor_weights(
        self,
        fund_code: str,
        equity_index_code: str,
        convertible_index_code: str,
        bond_index_type: str,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        regression_cfg = BOND_PLUS_MODEL["regression"]
        lookback_days = int(regression_cfg["lookback_days"])
        min_samples = int(regression_cfg["min_samples"])
        stock_weight_max = float(regression_cfg["stock_weight_max"])
        convertible_weight_max = float(regression_cfg["convertible_weight_max"])

        fund_returns_df = self.fund_fetcher.get_fund_nav_return_series(fund_code=fund_code, lookback_days=lookback_days, end_date=end_date)
        equity_returns_df = self.index_fetcher.get_a_index_return_series(index_code=equity_index_code, lookback_days=lookback_days, end_date=end_date).rename(columns={"index_return": "equity_return"})
        convertible_returns_df = self.index_fetcher.get_a_index_return_series(index_code=convertible_index_code, lookback_days=lookback_days, end_date=end_date).rename(columns={"index_return": "convertible_return"})
        bond_returns_df = self.bond_fetcher.get_bond_index_return_series(index_type=bond_index_type, lookback_days=lookback_days, end_date=end_date)

        merged_df = pd.merge(fund_returns_df, equity_returns_df, on="date", how="inner")
        merged_df = pd.merge(merged_df, convertible_returns_df, on="date", how="inner")
        merged_df = pd.merge(merged_df, bond_returns_df, on="date", how="inner")
        merged_df["fund_return"] = pd.to_numeric(merged_df["fund_return"], errors="coerce")
        merged_df["equity_return"] = pd.to_numeric(merged_df["equity_return"], errors="coerce")
        merged_df["convertible_return"] = pd.to_numeric(merged_df["convertible_return"], errors="coerce")
        merged_df["bond_return"] = pd.to_numeric(merged_df["bond_return"], errors="coerce")
        merged_df = merged_df.dropna(subset=["fund_return", "equity_return", "convertible_return", "bond_return"])
        if len(merged_df) < min_samples:
            raise ValueError(f"固收+三因子回归样本不足: {len(merged_df)} < {min_samples}")

        x_matrix = merged_df[["equity_return", "convertible_return", "bond_return"]].to_numpy(dtype=float)
        y_vector = merged_df["fund_return"].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(x_matrix, y_vector, rcond=None)

        stock_weight = max(float(coefficients[0]), 0.0)
        convertible_weight = max(float(coefficients[1]), 0.0)
        bond_weight = max(float(coefficients[2]), 0.0)
        weight_sum = stock_weight + convertible_weight + bond_weight
        if weight_sum <= 1e-12:
            raise ValueError("固收+三因子回归得到的权重和为0")

        stock_weight /= weight_sum
        convertible_weight /= weight_sum
        bond_weight /= weight_sum

        if stock_weight > stock_weight_max:
            stock_weight = stock_weight_max
        if convertible_weight > convertible_weight_max:
            convertible_weight = convertible_weight_max

        bond_weight = max(1.0 - stock_weight - convertible_weight, 0.0)
        renorm_sum = stock_weight + convertible_weight + bond_weight
        if renorm_sum <= 1e-12:
            raise ValueError("固收+三因子回归权重归一化失败")

        stock_weight /= renorm_sum
        convertible_weight /= renorm_sum
        bond_weight /= renorm_sum
        return {"samples": len(merged_df), "w_stock": stock_weight, "w_cb": convertible_weight, "w_bond": bond_weight}

    def estimate(
        self,
        fund_code: str,
        last_nav: float,
        nav_date: str,
        target_date: str | None = None,
        fund_type: str = "pure",
        stock_position: float | None = None,
        stock_position_snapshot: dict[str, Any] | None = None,
        position_source_override: str | None = None,
        convertible_position: float | None = None,
        equity_index_code: str | None = None,
        convertible_index_code: str | None = None,
        bond_index_type: str = "comprehensive",
        mgmt_rate: float | None = None,
        custody_rate: float | None = None,
        strict: bool | None = None,
        batch_context: BatchContext | None = None,
    ) -> dict[str, Any]:
        logger.info(f"开始估算债券基金 {fund_code}, 类型: {fund_type}")
        if strict is None:
            strict = STRICT_MODE
        warnings: list[str] = []
        used_sources: dict[str, Any] = {"bond_index": {bond_index_type: "bond_index_snapshot"}}

        nav_date_resolved, target_date_resolved, fee_days = self.resolve_fee_days(nav_date, target_date)
        data_as_of_date = batch_context.data_as_of_date if batch_context is not None else date.today().isoformat()
        nav_dt = self.normalize_date(nav_date_resolved, "nav_date")
        target_dt = self.normalize_date(target_date_resolved, "target_date")
        data_as_of_dt = self.normalize_date(data_as_of_date, "data_as_of_date")

        if equity_index_code is None:
            equity_index_code = PROXY_INDEX_MAP["沪深300"]
        if convertible_index_code is None:
            convertible_index_code = str(BOND_PLUS_MODEL["convertible_index_code"])

        if fund_type == "pure":
            if mgmt_rate is None:
                mgmt_rate = DEFAULT_FEES["bond_pure"]["management"]
            if custody_rate is None:
                custody_rate = DEFAULT_FEES["bond_pure"]["custody"]
        elif fund_type == "plus":
            if mgmt_rate is None:
                mgmt_rate = DEFAULT_FEES["bond_plus"]["management"]
            if custody_rate is None:
                custody_rate = DEFAULT_FEES["bond_plus"]["custody"]
        else:
            raise ValueError(f"未知债券基金类型: {fund_type}")

        bond_snapshot = self._get_bond_snapshot(bond_index_type=bond_index_type, batch_context=batch_context, strict=strict)
        bond_return = float(bond_snapshot["change_pct"])
        bond_prev_date = bond_snapshot.get("prev_date")
        bond_curr_date = bond_snapshot.get("curr_date")
        bond_return_source = "bond_index_snapshot"
        bond_stale_fallback: dict[str, Any] | None = None

        if bond_curr_date is not None:
            bond_curr_dt = self.normalize_date(bond_curr_date, "bond_curr_date")
            if bond_curr_dt < target_dt and not (batch_context is not None and batch_context.is_historical):
                is_same_day_estimate = target_dt == data_as_of_dt
                latest_published_snapshot_is_valid = is_same_day_estimate and bond_curr_dt == nav_dt
                stale_msg = f"债券指数数据滞后: curr_date={bond_curr_date}, target_date={target_date_resolved}"
                if strict and latest_published_snapshot_is_valid:
                    logger.info(
                        f"债券指数快照沿用最新已发布交易日: curr_date={bond_curr_date}, nav_date={nav_date_resolved}, target_date={target_date_resolved}"
                    )
                elif strict:
                    raise ValueError(stale_msg)
                else:
                    warnings.append(stale_msg)
                    logger.warning(stale_msg)
                    bond_return, bond_stale_fallback = self._build_stale_fallback_bond_return(warnings)
                    bond_return_source = "stale_fallback_accrual_rate_proxy"

        fee_drag = self.daily_fee_drag(mgmt_rate, custody_rate, nav_date=nav_date_resolved, target_date=target_date_resolved)

        if fund_type == "pure":
            r_bond = bond_return / 100
            r_t = r_bond - fee_drag
            result = {
                "fund_code": fund_code,
                "fund_type": "纯债",
                "last_nav": last_nav,
                "nav_date": nav_date_resolved,
                "target_date": target_date_resolved,
                "fee_days": fee_days,
                "estimated_nav": self.estimate_nav(last_nav, r_t),
                "estimated_return": r_t * 100,
                "bond_return": bond_return,
                "bond_prev_date": bond_prev_date,
                "bond_curr_date": bond_curr_date,
                "bond_return_source": bond_return_source,
                "bond_stale_fallback": bond_stale_fallback,
                "position_source": "not_applicable",
                "bond_data_as_of_date": batch_context.bond_index_as_of_dates.get(bond_index_type) if batch_context is not None and batch_context.bond_index_as_of_dates.get(bond_index_type) is not None else (bond_curr_date or data_as_of_date),
                "fee_drag": fee_drag * 100,
                "warnings": warnings,
                "strict_mode": strict,
            }
            result.update(build_quality_metadata(strict, passed=True, used_sources=used_sources))
        else:
            position_source = "input_parameter"
            weight_source = "position_assumption"
            stock_position_report_date = None
            stock_position_raw_report_period = None
            stock_position_snapshot_source = None

            if stock_position is None:
                if stock_position_snapshot is not None:
                    stock_position = float(stock_position_snapshot["position"])
                    stock_position_report_date = stock_position_snapshot.get("report_date")
                    stock_position_raw_report_period = stock_position_snapshot.get("raw_report_period")
                    stock_position_snapshot_source = stock_position_snapshot.get("snapshot_source")
                    position_source = "fund_disclosure_latest_report"
                else:
                    stock_snapshot = self.fund_fetcher.get_fund_stock_position_snapshot(fund_code)
                    stock_position = float(stock_snapshot["position"])
                    stock_position_report_date = stock_snapshot.get("report_date")
                    stock_position_raw_report_period = stock_snapshot.get("raw_report_period")
                    stock_position_snapshot_source = stock_snapshot.get("snapshot_source")
                    position_source = "fund_disclosure_latest_report"
            elif position_source_override is not None:
                position_source = position_source_override

            stock_position = self.validate_percentage(stock_position, "固收+股票仓位")
            if stock_position > 50:
                if strict:
                    raise ValueError(f"基金 {fund_code} 固收+股票仓位 {stock_position:.2f}% 异常偏高，默认值参考 {DEFAULT_POSITION['bond_plus_stock']:.2f}%")
                fallback_position = DEFAULT_POSITION["bond_plus_stock"]
                warning_msg = f"基金 {fund_code} 固收+股票仓位 {stock_position:.2f}% 异常偏高，非严格模式下回退为默认值 {fallback_position:.2f}%"
                logger.warning(warning_msg)
                warnings.append(warning_msg)
                stock_position = fallback_position
                position_source = "default_assumption_fallback"

            if convertible_position is None:
                convertible_position = float(BOND_PLUS_MODEL["default_convertible_position"])
            convertible_position = self.validate_percentage(convertible_position, "固收+可转债仓位")

            if stock_position + convertible_position > 100:
                if strict:
                    raise ValueError(f"基金 {fund_code} 固收+仓位和超过100%: stock={stock_position:.2f}%, convertible={convertible_position:.2f}%")
                adjusted_convertible_position = max(100.0 - stock_position, 0.0)
                warning_msg = f"基金 {fund_code} 固收+仓位和超过100%，非严格模式下将可转债仓位从 {convertible_position:.2f}% 调整为 {adjusted_convertible_position:.2f}%"
                logger.warning(warning_msg)
                warnings.append(warning_msg)
                convertible_position = adjusted_convertible_position

            w_stock = stock_position / 100
            w_cb = convertible_position / 100
            w_bond = max(1 - w_stock - w_cb, 0.0)
            regression_applied = False
            regression_samples = None
            if bool(BOND_PLUS_MODEL["regression"].get("enabled", False)):
                try:
                    regression_result = self._fit_bond_plus_three_factor_weights(
                        fund_code=fund_code,
                        equity_index_code=equity_index_code,
                        convertible_index_code=convertible_index_code,
                        bond_index_type=bond_index_type,
                        end_date=nav_date_resolved,
                    )
                    w_stock = float(regression_result["w_stock"])
                    w_cb = float(regression_result["w_cb"])
                    w_bond = float(regression_result["w_bond"])
                    regression_samples = int(regression_result["samples"])
                    regression_applied = True
                    weight_source = "regression_three_factor"
                    position_source = "regression_three_factor"
                except Exception as exc:
                    if strict:
                        raise ValueError(f"基金 {fund_code} 固收+三因子回归失败: {exc}") from exc
                    warning_msg = f"基金 {fund_code} 固收+三因子回归失败，沿用仓位假设: {exc}"
                    logger.warning(warning_msg)
                    warnings.append(warning_msg)

            stock_position = w_stock * 100
            convertible_position = w_cb * 100
            bond_position = w_bond * 100
            equity_return = self._get_a_index_return(index_code=equity_index_code, batch_context=batch_context, strict=strict)
            convertible_return = self._get_a_index_return(index_code=convertible_index_code, batch_context=batch_context, strict=strict)
            r_equity = equity_return / 100
            r_cb = convertible_return / 100
            r_bond = bond_return / 100
            r_t = (w_stock * r_equity + w_cb * r_cb + w_bond * r_bond) - fee_drag

            result = {
                "fund_code": fund_code,
                "fund_type": "固收+",
                "last_nav": last_nav,
                "nav_date": nav_date_resolved,
                "target_date": target_date_resolved,
                "fee_days": fee_days,
                "estimated_nav": self.estimate_nav(last_nav, r_t),
                "estimated_return": r_t * 100,
                "stock_position": stock_position,
                "convertible_position": convertible_position,
                "bond_position": bond_position,
                "w_stock": w_stock,
                "w_cb": w_cb,
                "w_bond": w_bond,
                "equity_return": equity_return,
                "convertible_return": convertible_return,
                "bond_return": bond_return,
                "equity_index_code": equity_index_code,
                "convertible_index_code": convertible_index_code,
                "bond_prev_date": bond_prev_date,
                "bond_curr_date": bond_curr_date,
                "bond_return_source": bond_return_source,
                "bond_stale_fallback": bond_stale_fallback,
                "position_source": position_source,
                "weight_source": weight_source,
                "regression_applied": regression_applied,
                "regression_samples": regression_samples,
                "stock_position_report_date": stock_position_report_date,
                "stock_position_raw_report_period": stock_position_raw_report_period,
                "stock_position_snapshot_source": stock_position_snapshot_source,
                "index_data_as_of_date": (
                    min(
                        [
                            value
                            for value in [
                                batch_context.a_index_as_of_dates.get(equity_index_code) if batch_context is not None else None,
                                batch_context.a_index_as_of_dates.get(convertible_index_code) if batch_context is not None else None,
                            ]
                            if value is not None
                        ]
                    )
                    if batch_context is not None and (
                        batch_context.a_index_as_of_dates.get(equity_index_code) is not None
                        or batch_context.a_index_as_of_dates.get(convertible_index_code) is not None
                    )
                    else data_as_of_date
                ),
                "bond_data_as_of_date": batch_context.bond_index_as_of_dates.get(bond_index_type) if batch_context is not None and batch_context.bond_index_as_of_dates.get(bond_index_type) is not None else (bond_curr_date or data_as_of_date),
                "fee_drag": fee_drag * 100,
                "warnings": warnings,
                "strict_mode": strict,
            }
            result.update(
                build_quality_metadata(
                    strict,
                    passed=True,
                    used_sources={
                        **used_sources,
                        "a_index": {
                            equity_index_code: "a_index_live",
                            convertible_index_code: "a_index_live",
                        },
                    },
                )
            )

        logger.info(f"估算完成: {fund_code}, 净值日期: {nav_date_resolved}, 估算目标日: {target_date_resolved}, 费率天数: {fee_days}, 估算涨跌: {r_t*100:+.4f}%")
        return result
