"""债券数据获取器。"""

from typing import Any

import akshare as ak
import pandas as pd
from loguru import logger

from .base_fetcher import BaseFetcher
from .historical_provider import AkshareHistoricalDataProvider, HistoricalDataProvider


class BondFetcher(BaseFetcher):
    """债券指数及收益率数据获取。"""

    INDEX_NAME_MAP = {
        "comprehensive": "中债综合全价指数",
        "treasury": "中债国债全价指数",
        "credit": "中债信用债全价指数",
    }

    def __init__(self, historical_provider: HistoricalDataProvider | None = None):
        self.historical_provider = historical_provider or AkshareHistoricalDataProvider()

    @classmethod
    def _resolve_index_name(cls, index_type: str) -> str:
        return cls.INDEX_NAME_MAP.get(index_type, cls.INDEX_NAME_MAP["comprehensive"])

    @staticmethod
    def _resolve_date_column(df: pd.DataFrame, context_name: str) -> str:
        for candidate in ["日期", "date", "Date", "交易日期"]:
            if candidate in df.columns:
                return candidate
        raise ValueError(f"{context_name} 缺少日期字段，实际字段: {list(df.columns)}")

    @BaseFetcher.with_cache(ttl=3600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_bond_index_snapshot(self, index_type: str = "comprehensive") -> dict[str, Any]:
        logger.info(f"获取债券指数快照: {index_type}")
        bond_df = ak.bond_zh_index_daily()
        if bond_df.empty:
            raise ValueError("中债指数数据为空")

        required_columns = {"指数名称", "收盘"}
        if not required_columns.issubset(set(bond_df.columns)):
            raise ValueError(f"中债指数字段异常，实际字段: {list(bond_df.columns)}")

        date_column = self._resolve_date_column(bond_df, "中债指数数据")
        target_name = self._resolve_index_name(index_type)
        target_df = bond_df[bond_df["指数名称"] == target_name].copy()
        if target_df.empty:
            raise ValueError(f"未找到债券指数: {target_name}")

        target_df[date_column] = pd.to_datetime(target_df[date_column], errors="coerce")
        target_df["收盘"] = pd.to_numeric(target_df["收盘"], errors="coerce")
        target_df = target_df.dropna(subset=[date_column, "收盘"])
        if len(target_df) < 2:
            raise ValueError(f"债券指数 {target_name} 数据不足，无法计算单日涨跌幅")

        target_df = target_df.sort_values(by=date_column).tail(2)
        prev_row = target_df.iloc[0]
        curr_row = target_df.iloc[1]
        prev_close = float(prev_row["收盘"])
        curr_close = float(curr_row["收盘"])
        if prev_close <= 0:
            raise ValueError(f"债券指数 {target_name} 前值非法: {prev_close}")

        change_pct = ((curr_close - prev_close) / prev_close) * 100
        snapshot = {
            "index_type": index_type,
            "index_name": target_name,
            "prev_date": prev_row[date_column].date().isoformat(),
            "curr_date": curr_row[date_column].date().isoformat(),
            "prev_close": prev_close,
            "curr_close": curr_close,
            "change_pct": change_pct,
        }
        logger.debug(
            f"债券指数快照 {target_name}: {snapshot['prev_date']}->{snapshot['curr_date']}, {change_pct:+.4f}%"
        )
        return snapshot

    @BaseFetcher.with_cache(ttl=3600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_bond_index_return(self, index_type: str = "comprehensive") -> float:
        snapshot = self.get_bond_index_snapshot(index_type=index_type)
        return float(snapshot["change_pct"])

    @BaseFetcher.with_cache(ttl=3600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_bond_index_return_series(self, index_type: str, lookback_days: int, end_date: str | None = None) -> pd.DataFrame:
        if lookback_days <= 0:
            raise ValueError(f"lookback_days 必须为正整数，当前值: {lookback_days}")
        target_df = self._get_bond_index_history_df(index_type=index_type, end_date=end_date)
        target_df["bond_return"] = target_df["close"].pct_change() * 100
        series_df = target_df.dropna(subset=["bond_return"])[["date", "bond_return"]].copy().tail(lookback_days)
        if series_df.empty:
            raise ValueError(f"债券指数 {self._resolve_index_name(index_type)} 收益样本不足，无法满足回看窗口 {lookback_days}")
        series_df["date"] = series_df["date"].dt.date.astype(str)
        series_df["bond_return"] = series_df["bond_return"].astype(float)
        return series_df.reset_index(drop=True)

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_bond_index_history(self) -> pd.DataFrame:
        return self.historical_provider.get_bond_index_history()

    def get_bond_index_history_payloads(self, index_types: set[str], date_from: str, date_to: str) -> dict[str, dict[str, dict[str, Any]]]:
        history: dict[str, dict[str, dict[str, Any]]] = {}
        if not index_types:
            return history

        for index_type in sorted(index_types):
            target_df = self._get_bond_index_history_df(index_type=index_type, end_date=None)
            target_df["change_pct"] = target_df["close"].pct_change() * 100
            history_by_date: dict[object, dict[str, Any]] = {}
            available_dates = [item.date() for item in target_df["date"].tolist()]
            for index, row in target_df.iterrows():
                row_date = row["date"].date()
                prev_date = None if index == 0 else target_df.iloc[index - 1]["date"].date().isoformat()
                prev_close = None if index == 0 else float(target_df.iloc[index - 1]["close"])
                history_by_date[row_date] = {
                    "index_name": self._resolve_index_name(index_type),
                    "prev_date": prev_date,
                    "curr_date": row_date.isoformat(),
                    "prev_close": prev_close,
                    "curr_close": float(row["close"]),
                    "change_pct": float(row["change_pct"]) if not pd.isna(row["change_pct"]) else 0.0,
                    "data_as_of_date": row_date.isoformat(),
                }

            output: dict[str, dict[str, Any]] = {}
            for current_ts in pd.date_range(start=date_from, end=date_to, freq="D"):
                current_date = current_ts.date()
                eligible_dates = [item for item in available_dates if item <= current_date]
                if not eligible_dates:
                    raise ValueError(f"债券指数 {self._resolve_index_name(index_type)} 在 {current_date.isoformat()} 之前无历史行情")
                effective_date = eligible_dates[-1]
                payload = dict(history_by_date[effective_date])
                if effective_date != current_date:
                    payload["change_pct"] = 0.0
                output[current_date.isoformat()] = payload

            history[index_type] = output

        return history

    @BaseFetcher.with_cache(ttl=3600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_treasury_yield_snapshot(self, tenor: str = "10年") -> dict[str, Any]:
        logger.info(f"获取中国国债收益率快照: {tenor}")
        rate_df = ak.bond_zh_us_rate()
        if rate_df.empty:
            raise ValueError("中国国债收益率数据为空")

        date_column = self._resolve_date_column(rate_df, "中国国债收益率数据")
        yield_column_candidates = [
            column
            for column in rate_df.columns
            if isinstance(column, str) and tenor in column and ("中国国债收益率" in column or "国债收益率" in column)
        ]
        if not yield_column_candidates:
            raise ValueError(f"未找到期限为{tenor}的中国国债收益率字段，实际字段: {list(rate_df.columns)}")
        yield_column = yield_column_candidates[0]

        target_df = rate_df[[date_column, yield_column]].copy()
        target_df[date_column] = pd.to_datetime(target_df[date_column], errors="coerce")
        target_df[yield_column] = pd.to_numeric(target_df[yield_column], errors="coerce")
        target_df = target_df.dropna(subset=[date_column, yield_column])
        if len(target_df) < 2:
            raise ValueError(f"中国国债收益率({tenor}) 数据不足，无法计算日变动")

        target_df = target_df.sort_values(by=date_column).tail(2)
        prev_row = target_df.iloc[0]
        curr_row = target_df.iloc[1]
        prev_yield = float(prev_row[yield_column])
        curr_yield = float(curr_row[yield_column])
        snapshot = {
            "tenor": tenor,
            "yield_column": yield_column,
            "prev_date": prev_row[date_column].date().isoformat(),
            "curr_date": curr_row[date_column].date().isoformat(),
            "prev_yield": prev_yield,
            "curr_yield": curr_yield,
            "delta_yield": curr_yield - prev_yield,
        }
        logger.debug(
            f"中国国债收益率快照({tenor}): {snapshot['prev_date']}->{snapshot['curr_date']}, {snapshot['delta_yield']:+.4f}bp(百分点)"
        )
        return snapshot

    @staticmethod
    def get_daily_accrual(ytm: float, duration: float = 3.0) -> float:
        return ytm / 365

    def _get_bond_index_history_df(self, index_type: str, end_date: str | None) -> pd.DataFrame:
        bond_df = self.get_bond_index_history()
        if bond_df.empty:
            raise ValueError("中债指数数据为空")
        required_columns = {"指数名称", "收盘"}
        if not required_columns.issubset(set(bond_df.columns)):
            raise ValueError(f"中债指数字段异常，实际字段: {list(bond_df.columns)}")

        date_column = self._resolve_date_column(bond_df, "中债指数数据")
        target_name = self._resolve_index_name(index_type)
        target_df = bond_df[bond_df["指数名称"] == target_name].copy()
        if target_df.empty:
            raise ValueError(f"未找到债券指数: {target_name}")

        target_df[date_column] = pd.to_datetime(target_df[date_column], errors="coerce")
        target_df["收盘"] = pd.to_numeric(target_df["收盘"], errors="coerce")
        target_df = target_df.dropna(subset=[date_column, "收盘"])
        if target_df.empty:
            raise ValueError(f"债券指数 {target_name} 历史数据解析失败")

        target_df = (
            target_df.sort_values(by=date_column)
            .drop_duplicates(subset=[date_column], keep="last")
            .rename(columns={date_column: "date", "收盘": "close"})
            .reset_index(drop=True)
        )
        if end_date is not None:
            end_dt = pd.to_datetime(end_date).date()
            target_df = target_df[target_df["date"].dt.date <= end_dt].copy()
        if len(target_df) < 2:
            raise ValueError(f"债券指数 {target_name} 历史样本不足，无法计算收益率")
        return target_df[["date", "close"]]
