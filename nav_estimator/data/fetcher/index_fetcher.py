"""指数行情数据获取器。"""

from datetime import date
import hashlib
import json
import math
from typing import Any

import akshare as ak
import pandas as pd
from loguru import logger
import requests

from .base_fetcher import BaseFetcher
from .fund_fetcher import FundFetcher
from .historical_provider import AkshareHistoricalDataProvider, HistoricalDataProvider


class IndexFetcher(BaseFetcher):
    """A股、港股指数行情获取。"""
    A_INDEX_LIVE_CACHE_SCHEMA_VERSION = FundFetcher.A_INDEX_CACHE_SCHEMA_VERSION
    A_INDEX_UNSUPPORTED_CACHE_TTL = 60
    A_INDEX_LIVE_COVERAGE_CACHE_TTL = 60

    SINA_HEADERS = {
        "Referer": "https://vip.stock.finance.sina.com.cn/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self, historical_provider: HistoricalDataProvider | None = None):
        self.historical_provider = historical_provider or AkshareHistoricalDataProvider()

    @staticmethod
    def _normalize_a_index_code(index_code: str) -> str:
        code = str(index_code).strip().lower()
        if len(code) == 6 and code.isdigit():
            if code.startswith("399"):
                return f"sz{code}"
            return f"sh{code}"
        return code

    @staticmethod
    def _extract_sina_payload_fields(raw_text: str, symbol: str) -> list[str]:
        data_text = raw_text[raw_text.find('"') + 1: raw_text.rfind('"')]
        if data_text.strip() == "":
            raise ValueError(f"新浪指数实时行情为空: {symbol}")
        return [item.strip() for item in data_text.split(",")]

    @classmethod
    @BaseFetcher.retry_on_error(max_retries=3)
    def _get_sina_quote_text(cls, symbol: str) -> str:
        response = requests.get(
            f"https://hq.sinajs.cn/list={symbol}",
            headers=cls.SINA_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        return response.content.decode("gbk", errors="ignore")

    @staticmethod
    def _build_index_live_payload(
        value: float,
        *,
        source: str,
        source_priority: int,
        index_code: str,
        data_as_of_date: str | None = None,
    ) -> dict[str, Any]:
        return BaseFetcher.build_live_payload(
            value=value,
            source=source,
            source_priority=source_priority,
            raw={"index_code": index_code, "change_pct": float(value)},
            data_as_of_date=data_as_of_date or date.today().isoformat(),
        )

    @staticmethod
    def _build_a_index_live_unsupported_message(index_code: str) -> str:
        normalized_symbol = IndexFetcher._normalize_a_index_code(index_code)
        return (
            f"A股指数 {index_code} 实时行情源不支持该指数代码；"
            f"主源异常: 新浪指数实时行情为空: {normalized_symbol}; "
            f"备源异常: 未找到A股指数 {index_code}"
        )

    @staticmethod
    def _normalize_unsupported_a_index_live_error(index_code: str, error: Exception) -> Exception:
        message = str(error)
        if (
            "主备实时源均不可用" in message
            and "新浪指数实时行情为空:" in message
            and f"未找到A股指数 {index_code}" in message
        ):
            return ValueError(IndexFetcher._build_a_index_live_unsupported_message(index_code))
        return error

    @classmethod
    def _build_a_index_live_unsupported_cache_key(cls, index_code: str) -> str:
        normalized_code = cls._normalize_a_index_code(index_code)
        payload = {
            "args": [{"__instance_class__": cls.__name__}, normalized_code],
            "kwargs": {},
            "kind": "unsupported_live_error",
            "schema_version": cls.A_INDEX_LIVE_CACHE_SCHEMA_VERSION,
        }
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return (
            f"{cls.get_a_index_return_live.__module__}.{cls.get_a_index_return_live.__name__}"
            f":negative:{hashlib.sha256(payload_text.encode('utf-8')).hexdigest()}"
        )

    @classmethod
    def _get_cached_a_index_live_unsupported_error(cls, index_code: str) -> str | None:
        cache_key = cls._build_a_index_live_unsupported_cache_key(index_code)
        cached = BaseFetcher.cache.get(cache_key)
        if (
            isinstance(cached, dict)
            and cached.get("kind") == "a_index_live_unsupported"
            and cached.get("schema_version") == cls.A_INDEX_LIVE_CACHE_SCHEMA_VERSION
        ):
            return str(cached.get("message", "")).strip() or None
        return None

    @classmethod
    def _set_cached_a_index_live_unsupported_error(cls, index_code: str, message: str):
        cache_key = cls._build_a_index_live_unsupported_cache_key(index_code)
        BaseFetcher.cache.set(
            cache_key,
            {
                "kind": "a_index_live_unsupported",
                "schema_version": cls.A_INDEX_LIVE_CACHE_SCHEMA_VERSION,
                "message": str(message),
            },
            ttl=cls.A_INDEX_UNSUPPORTED_CACHE_TTL,
        )

    @classmethod
    def _build_a_index_live_coverage_cache_key(cls) -> str:
        payload = {
            "args": [{"__instance_class__": cls.__name__}],
            "kwargs": {},
            "kind": "a_index_live_coverage_snapshot",
            "schema_version": cls.A_INDEX_LIVE_CACHE_SCHEMA_VERSION,
        }
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{cls.__module__}.{cls.__name__}:coverage:{hashlib.sha256(payload_text.encode('utf-8')).hexdigest()}"

    @classmethod
    def _get_cached_a_index_live_coverage_snapshot(cls) -> dict[str, Any] | None:
        cache_key = cls._build_a_index_live_coverage_cache_key()
        cached = BaseFetcher.cache.get(cache_key)
        if (
            isinstance(cached, dict)
            and cached.get("kind") == "a_index_live_coverage_snapshot"
            and cached.get("schema_version") == cls.A_INDEX_LIVE_CACHE_SCHEMA_VERSION
        ):
            return cached
        return None

    @classmethod
    def _set_cached_a_index_live_coverage_snapshot(cls, snapshot: dict[str, Any]):
        cache_key = cls._build_a_index_live_coverage_cache_key()
        BaseFetcher.cache.set(cache_key, snapshot, ttl=cls.A_INDEX_LIVE_COVERAGE_CACHE_TTL)

    def _build_a_index_live_coverage_snapshot(self) -> dict[str, Any]:
        errors: list[str] = []
        sina_codes = set()
        catalog_codes = set()

        try:
            index_df = ak.stock_zh_index_spot_sina()
            if "代码" in index_df.columns:
                sina_codes = {
                    self._normalize_a_index_code(str(code))
                    for code in index_df["代码"].dropna().astype(str)
                    if str(code).strip() != ""
                }
            else:
                errors.append(f"sina_columns={list(index_df.columns)}")
        except Exception as error:
            errors.append(f"sina={error}")

        try:
            catalog_snapshot = FundFetcher.get_shared_a_index_catalog_snapshot()
            catalog_codes = {
                self._normalize_a_index_code(str(item.get("code", "")))
                for item in catalog_snapshot.get("records", [])
                if str(item.get("code", "")).strip() != ""
            }
        except Exception as error:
            errors.append(f"catalog={error}")

        coverage_codes = {code for code in sina_codes | catalog_codes if code}
        if not coverage_codes:
            raise ValueError("A股实时覆盖目录快照为空: " + "; ".join(errors))

        return {
            "kind": "a_index_live_coverage_snapshot",
            "schema_version": self.A_INDEX_LIVE_CACHE_SCHEMA_VERSION,
            "codes": sorted(coverage_codes),
            "sina_codes": sorted(code for code in sina_codes if code),
            "catalog_codes": sorted(code for code in catalog_codes if code),
        }

    @staticmethod
    def _coerce_finite_change_pct(raw_value: Any, *, label: str) -> float:
        change_pct = float(pd.to_numeric(raw_value, errors="coerce"))
        if pd.isna(change_pct) or not math.isfinite(change_pct):
            raise ValueError(f"{label}涨跌幅异常")
        return change_pct

    def _get_a_index_live_coverage_snapshot(self) -> dict[str, Any] | None:
        cached_snapshot = self._get_cached_a_index_live_coverage_snapshot()
        if cached_snapshot is not None:
            return cached_snapshot
        try:
            snapshot = self._build_a_index_live_coverage_snapshot()
        except Exception as error:
            logger.warning(f"构建A股实时覆盖目录快照失败，将继续走主备源探测: {error}")
            return None
        self._set_cached_a_index_live_coverage_snapshot(snapshot)
        return snapshot

    def _get_a_index_return_sina_single(self, index_code: str) -> tuple[float, str | None]:
        symbol = self._normalize_a_index_code(index_code)
        fields = self._extract_sina_payload_fields(self._get_sina_quote_text(symbol), symbol)
        if len(fields) < 31:
            raise ValueError(f"新浪A股指数实时行情字段不足: {index_code}")
        prev_close = pd.to_numeric(fields[2], errors="coerce")
        latest_price = pd.to_numeric(fields[3], errors="coerce")
        if pd.isna(prev_close) or pd.isna(latest_price) or float(prev_close) <= 0:
            raise ValueError(f"新浪A股指数实时行情字段异常: {index_code}")
        change_pct = (float(latest_price) - float(prev_close)) / float(prev_close) * 100
        data_as_of_date = fields[30] if len(fields) > 30 and fields[30] else None
        return float(change_pct), data_as_of_date

    def _get_a_index_return_em(self, index_code: str) -> tuple[float, str | None]:
        candidates = ["沪深重要指数", "上证系列指数", "深证系列指数", "中证系列指数", "指数成份"]
        normalized_code = str(index_code).strip().lower()
        for symbol in candidates:
            index_df = ak.stock_zh_index_spot_em(symbol=symbol)
            if "代码" not in index_df.columns or "涨跌幅" not in index_df.columns:
                continue
            index_row = index_df[index_df["代码"].astype(str).str.lower() == normalized_code]
            if not index_row.empty:
                change_pct = float(index_row.iloc[0]["涨跌幅"])
                return change_pct, date.today().isoformat()
        raise ValueError(f"未找到A股指数 {index_code}")

    def _get_hk_index_return_sina_single(self, index_code: str) -> tuple[float, str | None]:
        normalized_code = str(index_code).strip().upper()
        fields = self._extract_sina_payload_fields(self._get_sina_quote_text(f"hk{normalized_code}"), normalized_code)
        if len(fields) < 18:
            raise ValueError(f"新浪港股指数实时行情字段不足: {index_code}")
        change_pct = pd.to_numeric(fields[8], errors="coerce")
        if pd.isna(change_pct):
            prev_close = pd.to_numeric(fields[3], errors="coerce")
            latest_price = pd.to_numeric(fields[6], errors="coerce")
            if pd.isna(prev_close) or pd.isna(latest_price) or float(prev_close) <= 0:
                raise ValueError(f"新浪港股指数实时行情字段异常: {index_code}")
            change_pct = (float(latest_price) - float(prev_close)) / float(prev_close) * 100
        data_as_of_date = fields[17].replace("/", "-") if len(fields) > 17 and fields[17] else None
        return float(change_pct), data_as_of_date

    @BaseFetcher.with_cache(ttl=60)
    @BaseFetcher.retry_on_error(max_retries=3)
    def _get_global_index_return_em(self, index_code: str) -> tuple[float, str | None]:
        logger.info(f"获取全球指数行情: {index_code}")
        index_df = ak.index_global_spot_em()
        if "代码" not in index_df.columns or "涨跌幅" not in index_df.columns:
            raise ValueError(f"全球指数行情字段异常，实际字段: {list(index_df.columns)}")

        normalized_code = str(index_code).strip().upper()
        matched = index_df[index_df["代码"].astype(str).str.upper() == normalized_code]
        if matched.empty:
            raise ValueError(f"未找到全球指数 {index_code}")

        change_pct = float(matched.iloc[0]["涨跌幅"])
        as_of_date = None
        for column in ["更新时间", "日期", "时间"]:
            if column in matched.columns:
                value = str(matched.iloc[0][column]).strip()
                if value and value.lower() != "nan":
                    as_of_date = value.replace("/", "-")
                    break
        logger.debug(f"全球指数 {index_code} 涨跌幅: {change_pct}%")
        return change_pct, as_of_date

    @BaseFetcher.with_cache(ttl=60)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_a_index_return(self, index_code: str) -> float:
        logger.info(f"获取A股指数行情: {index_code}")
        try:
            index_df = ak.stock_zh_index_spot_sina()
            if "代码" not in index_df.columns or "涨跌幅" not in index_df.columns:
                raise ValueError(f"A股指数行情字段异常，实际字段: {list(index_df.columns)}")

            target_code = self._normalize_a_index_code(index_code)
            index_row = index_df[index_df["代码"].astype(str).str.lower() == target_code]
            if index_row.empty:
                raise ValueError(f"未找到指数 {index_code}（标准化后: {target_code}）")

            change_pct = float(index_row.iloc[0]["涨跌幅"])
            logger.debug(f"指数 {index_code} 涨跌幅: {change_pct}%")
            return change_pct
        except Exception as e:
            logger.error(f"获取指数行情失败 {index_code}: {e}")
            raise

    @BaseFetcher.with_cache(ttl=60)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_hk_index_return(self, index_code: str) -> float:
        logger.info(f"获取港股指数行情: {index_code}")
        try:
            index_df = ak.stock_hk_index_spot_em()
            index_row = index_df[index_df["代码"] == index_code]
            if index_row.empty:
                raise ValueError(f"未找到指数 {index_code}")
            change_pct = float(index_row.iloc[0]["涨跌幅"])
            logger.debug(f"指数 {index_code} 涨跌幅: {change_pct}%")
            return change_pct
        except Exception as e:
            logger.error(f"获取港股指数行情失败 {index_code}: {e}")
            raise

    @BaseFetcher.with_cache(ttl=60)
    def _get_hk_index_return_em_once(self, index_code: str) -> float:
        logger.info(f"获取港股指数行情: {index_code}")
        index_df = ak.stock_hk_index_spot_em()
        index_row = index_df[index_df["代码"] == index_code]
        if index_row.empty:
            raise ValueError(f"未找到指数 {index_code}")
        change_pct = self._coerce_finite_change_pct(index_row.iloc[0]["涨跌幅"], label=f"港股指数 {index_code} 实时")
        logger.debug(f"指数 {index_code} 涨跌幅: {change_pct}%")
        return change_pct

    @BaseFetcher.with_cache(ttl=60)
    def get_a_index_return_live(self, index_code: str, strict: bool = True) -> dict[str, Any]:
        cached_error = self._get_cached_a_index_live_unsupported_error(index_code)
        if cached_error is not None:
            raise ValueError(cached_error)
        coverage_snapshot = self._get_a_index_live_coverage_snapshot()
        normalized_code = self._normalize_a_index_code(index_code)
        if coverage_snapshot is not None and normalized_code not in set(coverage_snapshot.get("codes", [])):
            unsupported_error = self._build_a_index_live_unsupported_message(index_code)
            self._set_cached_a_index_live_unsupported_error(index_code, unsupported_error)
            raise ValueError(unsupported_error)

        def primary_fetcher():
            change_pct, data_as_of_date = self._get_a_index_return_sina_single(index_code)
            return self._build_index_live_payload(
                change_pct,
                source="sina_index_single_quote",
                source_priority=1,
                index_code=index_code,
                data_as_of_date=data_as_of_date,
            )

        def backup_fetcher():
            change_pct, data_as_of_date = self._get_a_index_return_em(index_code)
            return self._build_index_live_payload(
                change_pct,
                source="eastmoney_index_quote",
                source_priority=2,
                index_code=index_code,
                data_as_of_date=data_as_of_date,
            )

        try:
            return BaseFetcher.resolve_live_source(
                primary_fetcher=primary_fetcher,
                backup_fetcher=backup_fetcher,
                strict=strict,
                label=f"A股指数 {index_code} 实时行情",
                threshold=1.0,
            )
        except Exception as error:
            normalized_error = self._normalize_unsupported_a_index_live_error(index_code, error)
            if "实时行情源不支持该指数代码" in str(normalized_error):
                self._set_cached_a_index_live_unsupported_error(index_code, str(normalized_error))
            raise normalized_error from error

    @BaseFetcher.with_cache(ttl=60)
    def get_hk_index_return_live(self, index_code: str, strict: bool = True) -> dict[str, Any]:
        def primary_fetcher():
            return self._build_index_live_payload(
                self._get_hk_index_return_em_once(index_code),
                source="eastmoney_index_quote",
                source_priority=1,
                index_code=index_code,
            )

        def backup_fetcher():
            change_pct, data_as_of_date = self._get_hk_index_return_sina_single(index_code)
            return self._build_index_live_payload(
                change_pct,
                source="sina_index_single_quote",
                source_priority=2,
                index_code=index_code,
                data_as_of_date=data_as_of_date,
            )

        return BaseFetcher.resolve_live_source(
            primary_fetcher=primary_fetcher,
            backup_fetcher=backup_fetcher,
            strict=strict,
            label=f"港股指数 {index_code} 实时行情",
            threshold=1.0,
        )

    @BaseFetcher.with_cache(ttl=60)
    def get_global_index_return_live(self, index_code: str, strict: bool = True) -> dict[str, Any]:
        change_pct, data_as_of_date = self._get_global_index_return_em(index_code)
        return self._build_index_live_payload(
            change_pct,
            source="eastmoney_global_index_quote",
            source_priority=1,
            index_code=index_code,
            data_as_of_date=data_as_of_date,
        )

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_a_index_history(self, index_code: str) -> pd.DataFrame:
        normalized_symbol = self._normalize_a_index_code(index_code)
        return self.historical_provider.get_a_index_history(symbol=normalized_symbol)

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_hk_index_history(self, index_code: str) -> pd.DataFrame:
        return self.historical_provider.get_hk_index_history(symbol=index_code)

    @BaseFetcher.with_cache(ttl=3600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_a_index_return_series(self, index_code: str, lookback_days: int, end_date: str | None = None) -> pd.DataFrame:
        if lookback_days <= 0:
            raise ValueError(f"lookback_days 必须为正整数，当前值: {lookback_days}")

        returns_df = self._build_a_index_return_series_df(index_code=index_code, end_date=end_date)
        returns_df = returns_df.tail(lookback_days)
        if returns_df.empty:
            raise ValueError(f"A股指数 {index_code} 收益样本不足，无法满足回看窗口 {lookback_days}")
        return returns_df.reset_index(drop=True)

    def get_a_index_return_history(self, index_codes: set[str], date_from: str, date_to: str) -> dict[str, dict[str, dict[str, Any]]]:
        history: dict[str, dict[str, dict[str, Any]]] = {}
        if not index_codes:
            return history
        for index_code in sorted(index_codes):
            history[index_code] = self._build_index_return_history_payloads(
                history_df=self.get_a_index_history(index_code),
                date_from=date_from,
                date_to=date_to,
                index_code=index_code,
                market="A股",
            )
        return history

    def get_hk_index_return_history(self, index_codes: set[str], date_from: str, date_to: str) -> dict[str, dict[str, dict[str, Any]]]:
        history: dict[str, dict[str, dict[str, Any]]] = {}
        if not index_codes:
            return history
        for index_code in sorted(index_codes):
            history[index_code] = self._build_index_return_history_payloads(
                history_df=self.get_hk_index_history(index_code),
                date_from=date_from,
                date_to=date_to,
                index_code=index_code,
                market="港股",
            )
        return history

    def _build_a_index_return_series_df(self, index_code: str, end_date: str | None) -> pd.DataFrame:
        normalized_code = str(index_code).strip()
        if not (len(normalized_code) == 6 and normalized_code.isdigit()):
            raise ValueError(f"A股指数代码格式非法: {index_code}")

        index_hist_df = self.get_a_index_history(normalized_code)
        if index_hist_df.empty:
            raise ValueError(f"未获取到A股指数 {index_code} 的历史数据")

        normalized_df = self._normalize_index_history_df(index_hist_df, f"A股指数 {index_code}")
        if end_date is not None:
            end_dt = pd.to_datetime(end_date).date()
            normalized_df = normalized_df[normalized_df["date"].dt.date <= end_dt].copy()
        if len(normalized_df) < 2:
            raise ValueError(f"A股指数 {index_code} 历史样本不足，无法计算收益率")

        normalized_df["index_return"] = normalized_df["close"].pct_change() * 100
        returns_df = normalized_df.dropna(subset=["index_return"])[["date", "index_return"]].copy()
        if returns_df.empty:
            raise ValueError(f"A股指数 {index_code} 收益样本不足")
        returns_df["date"] = returns_df["date"].dt.date.astype(str)
        returns_df["index_return"] = returns_df["index_return"].astype(float)
        return returns_df

    @staticmethod
    def _normalize_index_history_df(index_hist_df: pd.DataFrame, context: str) -> pd.DataFrame:
        date_column = None
        for candidate in ["date", "日期", "Date"]:
            if candidate in index_hist_df.columns:
                date_column = candidate
                break
        if date_column is None:
            raise ValueError(f"{context} 缺少日期字段，实际字段: {list(index_hist_df.columns)}")

        close_column = None
        for candidate in ["close", "收盘", "收盘价"]:
            if candidate in index_hist_df.columns:
                close_column = candidate
                break
        if close_column is None:
            raise ValueError(f"{context} 缺少收盘字段，实际字段: {list(index_hist_df.columns)}")

        normalized_df = index_hist_df.copy()
        normalized_df[date_column] = pd.to_datetime(normalized_df[date_column], errors="coerce")
        normalized_df[close_column] = pd.to_numeric(normalized_df[close_column], errors="coerce")
        normalized_df = normalized_df.dropna(subset=[date_column, close_column])
        if normalized_df.empty:
            raise ValueError(f"{context} 历史数据解析失败")

        normalized_df = (
            normalized_df.sort_values(by=date_column)
            .drop_duplicates(subset=[date_column], keep="last")
            .rename(columns={date_column: "date", close_column: "close"})
            .reset_index(drop=True)
        )
        return normalized_df[["date", "close"]]

    @classmethod
    def _build_index_return_history_payloads(
        cls,
        history_df: pd.DataFrame,
        date_from: str,
        date_to: str,
        index_code: str,
        market: str,
    ) -> dict[str, dict[str, Any]]:
        normalized_df = cls._normalize_index_history_df(history_df, f"{market}指数 {index_code}")
        normalized_df["change_pct"] = normalized_df["close"].pct_change() * 100
        available_dates = [item.date() for item in normalized_df["date"].tolist()]
        history_by_date: dict[Any, dict[str, Any]] = {}
        for _, row in normalized_df.iterrows():
            row_date = row["date"].date()
            history_by_date[row_date] = {
                "change_pct": float(row["change_pct"]) if not pd.isna(row["change_pct"]) else 0.0,
                "data_as_of_date": row_date.isoformat(),
            }

        output: dict[str, dict[str, Any]] = {}
        for current_ts in pd.date_range(start=date_from, end=date_to, freq="D"):
            current_date = current_ts.date()
            eligible_dates = [item for item in available_dates if item <= current_date]
            if not eligible_dates:
                raise ValueError(f"{market}指数 {index_code} 在 {current_date.isoformat()} 之前无历史行情")
            effective_date = eligible_dates[-1]
            payload = dict(history_by_date[effective_date])
            if effective_date != current_date:
                payload["change_pct"] = 0.0
            output[current_date.isoformat()] = payload

        return output
