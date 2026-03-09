"""股票行情数据获取器。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Iterable

import akshare as ak
import pandas as pd
import requests
from loguru import logger

from .base_fetcher import BaseFetcher
from .historical_provider import AkshareHistoricalDataProvider, HistoricalDataProvider


class StockFetcher(BaseFetcher):
    """A股、港股实时行情获取。"""

    SINA_HEADERS = {
        "Referer": "https://finance.sina.com.cn/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self, historical_provider: HistoricalDataProvider | None = None):
        self.historical_provider = historical_provider or AkshareHistoricalDataProvider()

    @staticmethod
    def _normalize_a_codes(codes: Iterable[str]) -> set[str]:
        normalized_codes: set[str] = set()
        for raw_code in codes:
            code = str(raw_code).strip()
            if len(code) == 6 and code.isdigit():
                normalized_codes.add(code)
        return normalized_codes

    @staticmethod
    def _normalize_hk_codes(codes: Iterable[str]) -> set[str]:
        normalized_codes: set[str] = set()
        for raw_code in codes:
            code = str(raw_code).strip()
            if code.isdigit() and len(code) <= 5:
                normalized_codes.add(code.zfill(5))
        return normalized_codes

    @staticmethod
    def _build_prices_from_spot_df(spot_df: pd.DataFrame, code_column: str, normalize_code) -> dict[str, dict[str, float]]:
        required_cols = {code_column, "名称", "最新价", "涨跌幅", "成交量"}
        if not required_cols.issubset(set(spot_df.columns)):
            raise ValueError(f"行情字段异常，实际字段: {list(spot_df.columns)}")

        normalized_df = spot_df[[code_column, "名称", "最新价", "涨跌幅", "成交量"]].copy()
        normalized_df[code_column] = normalized_df[code_column].astype(str).apply(normalize_code)
        normalized_df["最新价"] = pd.to_numeric(normalized_df["最新价"], errors="coerce")
        normalized_df["涨跌幅"] = pd.to_numeric(normalized_df["涨跌幅"], errors="coerce")
        normalized_df["成交量"] = pd.to_numeric(normalized_df["成交量"], errors="coerce")
        normalized_df = normalized_df.dropna(subset=[code_column, "最新价", "涨跌幅", "成交量"])

        prices: dict[str, dict[str, float]] = {}
        codes = normalized_df[code_column].tolist()
        names = normalized_df["名称"].astype(str).tolist()
        latest_prices = normalized_df["最新价"].tolist()
        change_pcts = normalized_df["涨跌幅"].tolist()
        volumes = normalized_df["成交量"].tolist()

        for code, name, latest_price, change_pct, volume in zip(codes, names, latest_prices, change_pcts, volumes):
            prices[code] = {
                "name": name,
                "price": float(latest_price),
                "change_pct": float(change_pct),
                "volume": float(volume),
            }
        return prices

    @staticmethod
    def _get_a_secid(code: str) -> str:
        normalized_code = str(code).strip()
        if not (len(normalized_code) == 6 and normalized_code.isdigit()):
            raise ValueError(f"A股代码非法: {code}")
        if normalized_code.startswith("6"):
            return f"1.{normalized_code}"
        return f"0.{normalized_code}"

    @staticmethod
    def _get_hk_secid(code: str) -> str:
        normalized_code = str(code).strip().zfill(5)
        if not (normalized_code.isdigit() and len(normalized_code) == 5):
            raise ValueError(f"港股代码非法: {code}")
        return f"116.{normalized_code}"

    @staticmethod
    def _build_single_quote_payload(data: dict, code: str, market: str) -> dict[str, float]:
        if not isinstance(data, dict):
            raise ValueError(f"{market} {code} 行情数据结构异常")
        latest_price = pd.to_numeric(data.get("f43"), errors="coerce")
        change_pct = pd.to_numeric(data.get("f170"), errors="coerce")
        volume = pd.to_numeric(data.get("f47"), errors="coerce")
        name = str(data.get("f58", "")).strip()
        if pd.isna(latest_price) or pd.isna(change_pct):
            raise ValueError(f"{market} {code} 行情字段异常")
        return {
            "name": name,
            "price": float(latest_price) / 100,
            "change_pct": float(change_pct) / 100 if abs(float(change_pct)) > 20 else float(change_pct),
            "volume": float(volume) if not pd.isna(volume) else 0.0,
        }

    @staticmethod
    def _extract_sina_payload_fields(raw_text: str, symbol: str) -> list[str]:
        data_text = raw_text[raw_text.find('"') + 1: raw_text.rfind('"')]
        if data_text.strip() == "":
            raise ValueError(f"新浪实时行情为空: {symbol}")
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

    @classmethod
    def _build_sina_a_share_payload(cls, code: str) -> tuple[dict[str, float], str | None]:
        fields = cls._extract_sina_payload_fields(cls._get_sina_quote_text(cls._get_a_index_code_for_sina(code)), code)
        if len(fields) < 31:
            raise ValueError(f"新浪A股实时行情字段不足: {code}")
        name = fields[0]
        prev_close = pd.to_numeric(fields[2], errors="coerce")
        latest_price = pd.to_numeric(fields[3], errors="coerce")
        volume = pd.to_numeric(fields[8], errors="coerce")
        if pd.isna(prev_close) or pd.isna(latest_price) or float(prev_close) <= 0:
            raise ValueError(f"新浪A股实时行情字段异常: {code}")
        change_pct = (float(latest_price) - float(prev_close)) / float(prev_close) * 100
        data_as_of_date = fields[30] if len(fields) > 30 and fields[30] else None
        return {
            "name": name,
            "price": float(latest_price),
            "change_pct": float(change_pct),
            "volume": float(volume) if not pd.isna(volume) else 0.0,
        }, data_as_of_date

    @classmethod
    def _build_sina_hk_share_payload(cls, code: str) -> tuple[dict[str, float], str | None]:
        normalized_code = str(code).strip().zfill(5)
        fields = cls._extract_sina_payload_fields(cls._get_sina_quote_text(f"hk{normalized_code}"), normalized_code)
        if len(fields) < 18:
            raise ValueError(f"新浪港股实时行情字段不足: {normalized_code}")
        name = fields[1] or fields[0]
        prev_close = pd.to_numeric(fields[3], errors="coerce")
        latest_price = pd.to_numeric(fields[6], errors="coerce")
        volume = pd.to_numeric(fields[12], errors="coerce")
        change_pct = pd.to_numeric(fields[8], errors="coerce")
        if pd.isna(prev_close) or pd.isna(latest_price) or float(prev_close) <= 0:
            raise ValueError(f"新浪港股实时行情字段异常: {normalized_code}")
        if pd.isna(change_pct):
            change_pct = (float(latest_price) - float(prev_close)) / float(prev_close) * 100
        data_as_of_date = fields[17].replace("/", "-") if len(fields) > 17 and fields[17] else None
        return {
            "name": name,
            "price": float(latest_price),
            "change_pct": float(change_pct),
            "volume": float(volume) if not pd.isna(volume) else 0.0,
        }, data_as_of_date

    @staticmethod
    def _get_a_index_code_for_sina(code: str) -> str:
        normalized_code = str(code).strip()
        if normalized_code.startswith("6"):
            return f"sh{normalized_code}"
        return f"sz{normalized_code}"

    @staticmethod
    def _build_quote_live_payload(
        payload: dict[str, float],
        *,
        source: str,
        source_priority: int,
        data_as_of_date: str | None = None,
    ) -> dict[str, Any]:
        return BaseFetcher.build_live_payload(
            value=float(payload["change_pct"]),
            source=source,
            source_priority=source_priority,
            raw=payload,
            data_as_of_date=data_as_of_date or date.today().isoformat(),
        )

    @staticmethod
    def _normalize_a_codes_ordered(codes: Iterable[str]) -> list[str]:
        normalized_codes = []
        seen = set()
        for raw_code in codes:
            code = str(raw_code).strip()
            if len(code) == 6 and code.isdigit() and code not in seen:
                seen.add(code)
                normalized_codes.append(code)
        return normalized_codes

    @staticmethod
    def _normalize_hk_codes_ordered(codes: Iterable[str]) -> list[str]:
        normalized_codes = []
        seen = set()
        for raw_code in codes:
            code = str(raw_code).strip()
            if code.isdigit() and len(code) <= 5:
                normalized = code.zfill(5)
                if normalized not in seen:
                    seen.add(normalized)
                    normalized_codes.append(normalized)
        return normalized_codes

    def _fetch_quotes_in_parallel(
        self,
        ordered_codes: list[str],
        quote_getter,
        market_label: str,
    ) -> tuple[dict[str, dict[str, float]], list[str]]:
        if not ordered_codes:
            return {}, []

        if len(ordered_codes) == 1:
            code = ordered_codes[0]
            try:
                with BaseFetcher.suppress_retry_logs():
                    return {code: quote_getter(code)}, []
            except Exception:
                return {}, [code]

        raw_results = {}
        max_workers = min(4, len(ordered_codes))
        parent_direct = BaseFetcher._is_direct_mode_latched()
        parent_strict = BaseFetcher._get_strict_mode_latched()

        def worker(code: str):
            previous_direct = BaseFetcher._is_direct_mode_latched()
            previous_strict = BaseFetcher._get_strict_mode_latched()
            BaseFetcher._set_direct_mode_latched(parent_direct)
            BaseFetcher._set_strict_mode_latched(parent_strict)
            try:
                with BaseFetcher.suppress_retry_logs():
                    try:
                        return code, quote_getter(code), None
                    except Exception as exc:
                        return code, None, exc
            finally:
                BaseFetcher._set_direct_mode_latched(previous_direct)
                BaseFetcher._set_strict_mode_latched(previous_strict)

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{market_label}_quote") as executor:
            futures = {executor.submit(worker, code): code for code in ordered_codes}
            for future in as_completed(futures):
                code, payload, error = future.result()
                raw_results[code] = {"payload": payload, "error": error}

        prices = {}
        missing_codes = []
        for code in ordered_codes:
            payload = raw_results[code]["payload"]
            if payload is not None:
                prices[code] = payload
            else:
                missing_codes.append(code)

        logger.info(f"{market_label}按代码行情请求完成: success={len(prices)}, missing={len(missing_codes)}")
        if missing_codes:
            logger.warning(f"{market_label}按代码行情缺失: {missing_codes}")
        return prices, missing_codes

    def _get_a_share_prices_by_codes_partial(self, codes: Iterable[str]) -> tuple[dict[str, dict[str, float]], list[str]]:
        ordered_codes = self._normalize_a_codes_ordered(codes)
        return self._fetch_quotes_in_parallel(ordered_codes, self._get_a_share_quote, "A股")

    def _get_hk_prices_by_codes_partial(self, codes: Iterable[str]) -> tuple[dict[str, dict[str, float]], list[str]]:
        ordered_codes = self._normalize_hk_codes_ordered(codes)
        return self._fetch_quotes_in_parallel(ordered_codes, self._get_hk_share_quote, "港股")

    def _get_a_share_quotes_live_by_codes_partial(self, codes: Iterable[str], strict: bool) -> tuple[dict[str, dict[str, Any]], list[str]]:
        ordered_codes = self._normalize_a_codes_ordered(codes)
        return self._fetch_quotes_in_parallel(
            ordered_codes,
            lambda code: self.get_a_share_quote_live(code, strict=strict),
            "A股",
        )

    def _get_hk_share_quotes_live_by_codes_partial(self, codes: Iterable[str], strict: bool) -> tuple[dict[str, dict[str, Any]], list[str]]:
        ordered_codes = self._normalize_hk_codes_ordered(codes)
        return self._fetch_quotes_in_parallel(
            ordered_codes,
            lambda code: self.get_hk_share_quote_live(code, strict=strict),
            "港股",
        )

    @BaseFetcher.with_cache(ttl=60)
    @BaseFetcher.retry_on_error(max_retries=3)
    def _get_a_share_quote(self, code: str) -> dict[str, float]:
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "fltt": "2",
            "invt": "2",
            "fields": "f43,f47,f57,f58,f170",
            "secid": self._get_a_secid(code),
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        return self._build_single_quote_payload(payload.get("data"), code, "A股")

    @BaseFetcher.with_cache(ttl=60)
    def get_a_share_quote_live(self, code: str, strict: bool = True) -> dict[str, Any]:
        def backup_fetcher():
            payload, data_as_of_date = self._build_sina_a_share_payload(code)
            return self._build_quote_live_payload(
                payload,
                source="sina_single_quote",
                source_priority=2,
                data_as_of_date=data_as_of_date,
            )

        return BaseFetcher.resolve_live_source(
            primary_fetcher=lambda: self._build_quote_live_payload(
                self._get_a_share_quote(code),
                source="eastmoney_single_quote",
                source_priority=1,
            ),
            backup_fetcher=backup_fetcher,
            strict=strict,
            label=f"A股 {code} 实时行情",
            threshold=2.0,
        )

    @BaseFetcher.with_cache(ttl=60)
    @BaseFetcher.retry_on_error(max_retries=3)
    def _get_hk_share_quote(self, code: str) -> dict[str, float]:
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        normalized_code = str(code).strip().zfill(5)
        params = {
            "fltt": "2",
            "invt": "2",
            "fields": "f43,f47,f57,f58,f170",
            "secid": self._get_hk_secid(normalized_code),
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        return self._build_single_quote_payload(payload.get("data"), normalized_code, "港股")

    @BaseFetcher.with_cache(ttl=60)
    def get_hk_share_quote_live(self, code: str, strict: bool = True) -> dict[str, Any]:
        normalized_code = str(code).strip().zfill(5)
        def backup_fetcher():
            payload, data_as_of_date = self._build_sina_hk_share_payload(normalized_code)
            return self._build_quote_live_payload(
                payload,
                source="sina_single_quote",
                source_priority=2,
                data_as_of_date=data_as_of_date,
            )

        return BaseFetcher.resolve_live_source(
            primary_fetcher=lambda: self._build_quote_live_payload(
                self._get_hk_share_quote(normalized_code),
                source="eastmoney_single_quote",
                source_priority=1,
            ),
            backup_fetcher=backup_fetcher,
            strict=strict,
            label=f"港股 {normalized_code} 实时行情",
            threshold=2.0,
        )

    @BaseFetcher.with_cache(ttl=60)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_a_share_prices(self) -> dict[str, dict[str, float]]:
        logger.info("获取A股实时行情")
        try:
            spot_df = ak.stock_zh_a_spot_em()
            prices = self._build_prices_from_spot_df(
                spot_df,
                code_column="代码",
                normalize_code=lambda value: str(value).strip(),
            )
            logger.debug(f"A股行情获取成功，共 {len(prices)} 只股票")
            return prices
        except Exception as e:
            logger.error(f"获取A股行情失败: {e}")
            raise

    @BaseFetcher.with_cache(ttl=60)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_hk_prices(self) -> dict[str, dict[str, float]]:
        logger.info("获取港股实时行情")
        try:
            spot_df = ak.stock_hk_spot_em()
            prices = self._build_prices_from_spot_df(
                spot_df,
                code_column="代码",
                normalize_code=lambda value: str(value).strip().zfill(5),
            )
            logger.debug(f"港股行情获取成功，共 {len(prices)} 只股票")
            return prices
        except Exception as e:
            logger.error(f"获取港股行情失败: {e}")
            raise

    @BaseFetcher.with_cache(ttl=60)
    def get_a_share_prices_by_codes(self, codes: Iterable[str]) -> dict[str, dict[str, float]]:
        normalized_codes = self._normalize_a_codes(codes)
        if not normalized_codes:
            raise ValueError("A股代码集合为空或全部非法，无法获取行情")

        prices, missing_codes = self._get_a_share_prices_by_codes_partial(normalized_codes)
        if missing_codes:
            raise ValueError(f"以下A股代码未获取到实时行情: {missing_codes}")
        return prices

    @BaseFetcher.with_cache(ttl=60)
    def get_hk_prices_by_codes(self, codes: Iterable[str]) -> dict[str, dict[str, float]]:
        normalized_codes = self._normalize_hk_codes(codes)
        if not normalized_codes:
            raise ValueError("港股代码集合为空或全部非法，无法获取行情")

        prices, missing_codes = self._get_hk_prices_by_codes_partial(normalized_codes)
        if missing_codes:
            raise ValueError(f"以下港股代码未获取到实时行情: {missing_codes}")
        return prices

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_a_share_history(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        normalized_code = self._normalize_a_codes([code])
        if not normalized_code:
            raise ValueError(f"A股代码非法: {code}")
        return self.historical_provider.get_a_share_history(
            symbol=next(iter(normalized_code)),
            start_date=start_date,
            end_date=end_date,
        )

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_hk_share_history(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        normalized_code = self._normalize_hk_codes([code])
        if not normalized_code:
            raise ValueError(f"港股代码非法: {code}")
        return self.historical_provider.get_hk_share_history(
            symbol=next(iter(normalized_code)),
            start_date=start_date,
            end_date=end_date,
        )

    def get_a_share_return_history_by_codes(self, codes: Iterable[str], date_from: str, date_to: str) -> dict[str, dict[str, dict[str, Any]]]:
        normalized_codes = self._normalize_a_codes(codes)
        history: dict[str, dict[str, dict[str, Any]]] = {}
        if not normalized_codes:
            return history

        start_dt = pd.to_datetime(date_from).date()
        fetch_start = (start_dt - timedelta(days=15)).isoformat()
        for code in normalized_codes:
            history_df = self.get_a_share_history(code, fetch_start, date_to)
            history[code] = self._build_daily_return_payloads(
                history_df=history_df,
                date_from=date_from,
                date_to=date_to,
                code=code,
                market="A股",
            )
        return history

    def get_hk_share_return_history_by_codes(self, codes: Iterable[str], date_from: str, date_to: str) -> dict[str, dict[str, dict[str, Any]]]:
        normalized_codes = self._normalize_hk_codes(codes)
        history: dict[str, dict[str, dict[str, Any]]] = {}
        if not normalized_codes:
            return history

        start_dt = pd.to_datetime(date_from).date()
        fetch_start = (start_dt - timedelta(days=15)).isoformat()
        for code in normalized_codes:
            history_df = self.get_hk_share_history(code, fetch_start, date_to)
            history[code] = self._build_daily_return_payloads(
                history_df=history_df,
                date_from=date_from,
                date_to=date_to,
                code=code,
                market="港股",
            )
        return history

    def get_stock_price(self, code: str, market: str = "A股", allow_missing: bool = False) -> dict[str, float]:
        if market == "A股":
            all_prices = self.get_a_share_prices()
        elif market == "港股":
            all_prices = self.get_hk_prices()
        else:
            raise ValueError(f"不支持的市场类型: {market}")

        if code not in all_prices:
            if allow_missing:
                logger.warning(f"未获取到股票 {code}({market}) 的行情数据，按allow_missing返回0")
                return {"name": "", "price": 0.0, "change_pct": 0.0, "volume": 0.0}
            raise ValueError(f"未获取到股票 {code}({market}) 的行情数据")

        return all_prices[code]

    @staticmethod
    def _build_daily_return_payloads(
        history_df: pd.DataFrame,
        date_from: str,
        date_to: str,
        code: str,
        market: str,
    ) -> dict[str, dict[str, Any]]:
        if history_df is None or history_df.empty:
            raise ValueError(f"{market} {code} 历史行情为空")

        date_column = None
        for candidate in ["日期", "date", "Date"]:
            if candidate in history_df.columns:
                date_column = candidate
                break
        if date_column is None:
            raise ValueError(f"{market} {code} 历史行情缺少日期字段: {list(history_df.columns)}")

        close_column = None
        for candidate in ["收盘", "close", "收盘价"]:
            if candidate in history_df.columns:
                close_column = candidate
                break
        if close_column is None:
            raise ValueError(f"{market} {code} 历史行情缺少收盘字段: {list(history_df.columns)}")

        name_column = None
        for candidate in ["名称", "股票名称", "name"]:
            if candidate in history_df.columns:
                name_column = candidate
                break

        volume_column = None
        for candidate in ["成交量", "volume"]:
            if candidate in history_df.columns:
                volume_column = candidate
                break

        normalized_df = history_df.copy()
        normalized_df[date_column] = pd.to_datetime(normalized_df[date_column], errors="coerce")
        normalized_df[close_column] = pd.to_numeric(normalized_df[close_column], errors="coerce")
        if volume_column is not None:
            normalized_df[volume_column] = pd.to_numeric(normalized_df[volume_column], errors="coerce")
        normalized_df = normalized_df.dropna(subset=[date_column, close_column])
        if normalized_df.empty:
            raise ValueError(f"{market} {code} 历史行情解析失败")

        normalized_df = (
            normalized_df.sort_values(by=date_column)
            .drop_duplicates(subset=[date_column], keep="last")
            .reset_index(drop=True)
        )
        normalized_df["__close_change_pct"] = normalized_df[close_column].pct_change() * 100
        if "涨跌幅" in normalized_df.columns:
            normalized_df["__daily_change_pct"] = pd.to_numeric(normalized_df["涨跌幅"], errors="coerce")
        else:
            normalized_df["__daily_change_pct"] = normalized_df["__close_change_pct"]

        normalized_df["__name"] = normalized_df[name_column].astype(str) if name_column is not None else str(code)
        normalized_df["__volume"] = normalized_df[volume_column].fillna(0.0) if volume_column is not None else 0.0

        history_by_date: dict[object, dict[str, Any]] = {}
        for _, row in normalized_df.iterrows():
            row_date = row[date_column].date()
            history_by_date[row_date] = {
                "name": str(row["__name"]),
                "price": float(row[close_column]),
                "change_pct": float(row["__daily_change_pct"]) if not pd.isna(row["__daily_change_pct"]) else 0.0,
                "volume": float(row["__volume"]) if volume_column is not None else 0.0,
                "data_as_of_date": row_date.isoformat(),
            }

        output: dict[str, dict[str, Any]] = {}
        available_dates = sorted(history_by_date.keys())
        for current_ts in pd.date_range(start=date_from, end=date_to, freq="D"):
            current_date = current_ts.date()
            eligible_dates = [item for item in available_dates if item <= current_date]
            if not eligible_dates:
                raise ValueError(f"{market} {code} 在 {current_date.isoformat()} 之前无历史行情")

            effective_date = eligible_dates[-1]
            payload = dict(history_by_date[effective_date])
            if effective_date != current_date:
                payload["change_pct"] = 0.0
            output[current_date.isoformat()] = payload

        return output
