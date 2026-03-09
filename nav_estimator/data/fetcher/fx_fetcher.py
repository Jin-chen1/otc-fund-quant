"""汇率数据获取器。"""

import akshare as ak
import pandas as pd
import requests
from loguru import logger

from ...config.settings import FX_DAILY_CHANGE_MAX_ABS, STRICT_MODE
from .base_fetcher import BaseFetcher
from .historical_provider import AkshareHistoricalDataProvider, HistoricalDataProvider


class FXFetcher(BaseFetcher):
    """外汇汇率数据获取。"""

    def __init__(self, historical_provider: HistoricalDataProvider | None = None):
        self.historical_provider = historical_provider or AkshareHistoricalDataProvider()

    @BaseFetcher.with_cache(ttl=300)
    @BaseFetcher.retry_on_error(max_retries=3)
    def _get_cny_mid_series(self, currency_name: str):
        fx_df = ak.currency_boc_safe()
        if fx_df.empty:
            raise ValueError("未获取到国家外汇管理局人民币中间价数据")
        required_columns = {"日期", currency_name}
        if not required_columns.issubset(set(fx_df.columns)):
            raise ValueError(f"汇率数据缺少必要字段: {required_columns}，实际字段: {list(fx_df.columns)}")

        series_df = fx_df[["日期", currency_name]].dropna().copy()
        if len(series_df) < 2:
            raise ValueError(f"{currency_name}中间价数据不足2条，无法计算日变动")
        return series_df.sort_values("日期")

    @BaseFetcher.with_cache(ttl=300)
    @BaseFetcher.retry_on_error(max_retries=3)
    def _get_hkd_cny_mid_series(self):
        return self._get_cny_mid_series("港元")

    @BaseFetcher.with_cache(ttl=300)
    @BaseFetcher.retry_on_error(max_retries=3)
    def _get_usd_cny_mid_series(self):
        return self._get_cny_mid_series("美元")

    @staticmethod
    def _parse_hkd_cny_spot_rate() -> float:
        spot_df = ak.fx_spot_quote()
        if spot_df.empty:
            raise ValueError("外汇现货行情为空")
        required_cols = {"货币对", "买报价", "卖报价"}
        if not required_cols.issubset(set(spot_df.columns)):
            raise ValueError(f"外汇现货行情字段异常，实际字段: {list(spot_df.columns)}")

        hkd_cny_row = spot_df[spot_df["货币对"].astype(str) == "HKD/CNY"]
        if hkd_cny_row.empty:
            raise ValueError("外汇现货行情中未找到 HKD/CNY 报价")
        buy_rate = float(hkd_cny_row.iloc[0]["买报价"])
        sell_rate = float(hkd_cny_row.iloc[0]["卖报价"])
        return (buy_rate + sell_rate) / 2

    @staticmethod
    def _parse_usd_cny_spot_rate() -> float:
        spot_df = ak.fx_spot_quote()
        if spot_df.empty:
            raise ValueError("外汇现货行情为空")
        required_cols = {"货币对", "买报价", "卖报价"}
        if not required_cols.issubset(set(spot_df.columns)):
            raise ValueError(f"外汇现货行情字段异常，实际字段: {list(spot_df.columns)}")

        usd_cny_row = spot_df[spot_df["货币对"].astype(str) == "USD/CNY"]
        if usd_cny_row.empty:
            raise ValueError("外汇现货行情中未找到 USD/CNY 报价")
        buy_rate = float(usd_cny_row.iloc[0]["买报价"])
        sell_rate = float(usd_cny_row.iloc[0]["卖报价"])
        return (buy_rate + sell_rate) / 2

    @staticmethod
    @BaseFetcher.retry_on_error(max_retries=3)
    def _get_sina_pair_daily_change(pair_code: str, currency_label: str) -> float:
        response = requests.get(
            f"https://hq.sinajs.cn/?list={pair_code}",
            headers={"Referer": "https://finance.sina.com.cn/"},
            timeout=15,
        )
        response.raise_for_status()
        text = response.content.decode("gbk", errors="ignore")
        data_text = text[text.find('"') + 1: text.rfind('"')]
        if data_text.strip() == "":
            raise ValueError(f"{currency_label} 新浪汇率行情为空")
        fields = [item.strip() for item in data_text.split(",")]
        numeric_fields = []
        for item in fields:
            try:
                numeric_fields.append(float(item))
            except ValueError:
                continue
        if len(numeric_fields) < 2:
            raise ValueError(f"{currency_label} 新浪汇率字段不足")
        current_rate = float(numeric_fields[-1])
        prev_rate = float(numeric_fields[1])
        if prev_rate <= 0:
            raise ValueError(f"{currency_label} 新浪前值非法: {prev_rate}")
        return ((current_rate - prev_rate) / prev_rate) * 100

    @staticmethod
    def _normalize_change_ratio(raw_value) -> float:
        if raw_value is None:
            raise ValueError("涨跌幅字段为空")
        if isinstance(raw_value, str):
            raw_text = raw_value.strip()
            if raw_text == "":
                raise ValueError("涨跌幅字段为空字符串")
            has_percent_symbol = raw_text.endswith("%")
            if has_percent_symbol:
                raw_text = raw_text[:-1].strip()
            numeric_value = float(raw_text)
            if has_percent_symbol:
                return numeric_value / 100
        else:
            numeric_value = float(raw_value)

        abs_value = abs(numeric_value)
        if abs_value <= 0.02:
            return numeric_value
        if abs_value <= 100:
            return numeric_value / 100
        return numeric_value

    @staticmethod
    def _validate_daily_change_pct(change_pct: float) -> float:
        if abs(change_pct) <= FX_DAILY_CHANGE_MAX_ABS:
            return change_pct
        adjusted_change_pct = change_pct / 100
        if abs(adjusted_change_pct) <= FX_DAILY_CHANGE_MAX_ABS:
            logger.warning(f"汇率日变动 {change_pct:+.4f}% 超阈值，按单位修正为 {adjusted_change_pct:+.4f}%")
            return adjusted_change_pct
        raise ValueError(f"汇率日变动异常: {change_pct:+.4f}%，超过阈值 ±{FX_DAILY_CHANGE_MAX_ABS}%")

    def _get_hkd_cny_daily_change_from_baidu(self) -> float:
        quote_df = ak.fx_quote_baidu(symbol="人民币")
        if quote_df.empty:
            raise ValueError("外汇行情数据为空")
        required_cols = {"代码", "涨跌幅"}
        if not required_cols.issubset(set(quote_df.columns)):
            raise ValueError(f"外汇行情字段异常，实际字段: {list(quote_df.columns)}")

        hkd_row = quote_df[quote_df["代码"].astype(str) == "CNYHKD"]
        if hkd_row.empty:
            raise ValueError("外汇行情中未找到 CNYHKD")

        cny_hkd_change_ratio = self._normalize_change_ratio(hkd_row.iloc[0]["涨跌幅"])
        if cny_hkd_change_ratio <= -1:
            raise ValueError(f"CNYHKD 涨跌幅非法: {cny_hkd_change_ratio}")

        hkd_cny_change_ratio = (1 / (1 + cny_hkd_change_ratio)) - 1
        hkd_cny_change_pct = self._validate_daily_change_pct(hkd_cny_change_ratio * 100)
        logger.debug(
            f"百度汇率回退计算: CNYHKD涨跌(比例)={cny_hkd_change_ratio:+.6f}, HKD/CNY涨跌={hkd_cny_change_pct:+.4f}%"
        )
        return hkd_cny_change_pct

    def _get_hkd_cny_daily_change_primary_payload(self) -> dict:
        series_df = self._get_hkd_cny_mid_series()
        prev_rate = float(series_df.iloc[-2]["港元"])
        curr_rate = float(series_df.iloc[-1]["港元"])
        if prev_rate <= 0:
            raise ValueError(f"前一日港元中间价非法: {prev_rate}")
        hkd_cny_change_pct = self._validate_daily_change_pct(((curr_rate - prev_rate) / prev_rate) * 100)
        data_as_of_date = str(series_df.iloc[-1]["日期"])
        return BaseFetcher.build_live_payload(
            value=hkd_cny_change_pct,
            source="safe_mid_rate",
            source_priority=1,
            raw={
                "prev_rate": prev_rate,
                "curr_rate": curr_rate,
                "change_pct": hkd_cny_change_pct,
            },
            data_as_of_date=data_as_of_date,
        )

    def _get_hkd_cny_daily_change_backup_payload(self) -> dict:
        hkd_cny_change_pct = self._get_hkd_cny_daily_change_from_baidu()
        return BaseFetcher.build_live_payload(
            value=hkd_cny_change_pct,
            source="baidu_fx_quote",
            source_priority=2,
            raw={"change_pct": hkd_cny_change_pct},
            data_as_of_date=pd.Timestamp.today().date().isoformat(),
        )

    def _get_usd_cny_daily_change_primary_payload(self) -> dict:
        series_df = self._get_usd_cny_mid_series()
        prev_rate = float(series_df.iloc[-2]["美元"])
        curr_rate = float(series_df.iloc[-1]["美元"])
        if prev_rate <= 0:
            raise ValueError(f"前一日美元中间价非法: {prev_rate}")
        usd_cny_change_pct = self._validate_daily_change_pct(((curr_rate - prev_rate) / prev_rate) * 100)
        data_as_of_date = str(series_df.iloc[-1]["日期"])
        return BaseFetcher.build_live_payload(
            value=usd_cny_change_pct,
            source="safe_mid_rate",
            source_priority=1,
            raw={
                "prev_rate": prev_rate,
                "curr_rate": curr_rate,
                "change_pct": usd_cny_change_pct,
            },
            data_as_of_date=data_as_of_date,
        )

    def _get_usd_cny_daily_change_backup_payload(self) -> dict:
        usd_cny_change_pct = self._validate_daily_change_pct(
            self._get_sina_pair_daily_change("USDCNY", "美元兑人民币")
        )
        return BaseFetcher.build_live_payload(
            value=usd_cny_change_pct,
            source="sina_fx_quote",
            source_priority=2,
            raw={"change_pct": usd_cny_change_pct},
            data_as_of_date=pd.Timestamp.today().date().isoformat(),
        )

    @BaseFetcher.with_cache(ttl=300)
    def get_hkd_cny_daily_change_live(self, strict: bool = STRICT_MODE) -> dict:
        return BaseFetcher.resolve_live_source(
            primary_fetcher=self._get_hkd_cny_daily_change_primary_payload,
            backup_fetcher=self._get_hkd_cny_daily_change_backup_payload,
            strict=strict,
            label="港币兑人民币日变动",
            threshold=0.3,
        )

    @BaseFetcher.with_cache(ttl=300)
    def get_usd_cny_daily_change_live(self, strict: bool = STRICT_MODE) -> dict:
        return BaseFetcher.resolve_live_source(
            primary_fetcher=self._get_usd_cny_daily_change_primary_payload,
            backup_fetcher=self._get_usd_cny_daily_change_backup_payload,
            strict=strict,
            label="美元兑人民币日变动",
            threshold=0.3,
        )

    @BaseFetcher.with_cache(ttl=300)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_hkd_cny_daily_change(self, strict: bool = STRICT_MODE) -> float:
        logger.info("获取港币兑人民币汇率变动")
        try:
            series_df = self._get_hkd_cny_mid_series()
            prev_rate = float(series_df.iloc[-2]["港元"])
            curr_rate = float(series_df.iloc[-1]["港元"])
            if prev_rate <= 0:
                raise ValueError(f"前一日港元中间价非法: {prev_rate}")
            hkd_cny_change_pct = self._validate_daily_change_pct(((curr_rate - prev_rate) / prev_rate) * 100)
            logger.debug(f"中间价主路径计算: 前值={prev_rate}, 当前={curr_rate}, HKD/CNY涨跌={hkd_cny_change_pct:+.4f}%")
            return hkd_cny_change_pct
        except Exception as main_error:
            logger.warning(f"中间价主路径失败，回退至百度行情: {main_error}")
            try:
                return self._get_hkd_cny_daily_change_from_baidu()
            except Exception as fallback_error:
                if strict:
                    raise ValueError(
                        f"港币兑人民币日变动获取失败，主路径异常: {main_error}; 回退路径异常: {fallback_error}"
                    ) from fallback_error
                logger.warning(
                    "港币兑人民币日变动获取失败（非严格模式），"
                    f"主路径异常: {main_error}; 回退路径异常: {fallback_error}；回退为0.0%"
                )
                return 0.0

    @BaseFetcher.with_cache(ttl=300)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_usd_cny_daily_change(self, strict: bool = STRICT_MODE) -> float:
        logger.info("获取美元兑人民币汇率变动")
        try:
            series_df = self._get_usd_cny_mid_series()
            prev_rate = float(series_df.iloc[-2]["美元"])
            curr_rate = float(series_df.iloc[-1]["美元"])
            if prev_rate <= 0:
                raise ValueError(f"前一日美元中间价非法: {prev_rate}")
            usd_cny_change_pct = self._validate_daily_change_pct(((curr_rate - prev_rate) / prev_rate) * 100)
            logger.debug(f"中间价主路径计算: 前值={prev_rate}, 当前={curr_rate}, USD/CNY涨跌={usd_cny_change_pct:+.4f}%")
            return usd_cny_change_pct
        except Exception as main_error:
            logger.warning(f"美元中间价主路径失败，回退至新浪行情: {main_error}")
            try:
                return self._validate_daily_change_pct(self._get_sina_pair_daily_change("USDCNY", "美元兑人民币"))
            except Exception as fallback_error:
                if strict:
                    raise ValueError(
                        f"美元兑人民币日变动获取失败，主路径异常: {main_error}; 回退路径异常: {fallback_error}"
                    ) from fallback_error
                logger.warning(
                    "美元兑人民币日变动获取失败（非严格模式），"
                    f"主路径异常: {main_error}; 回退路径异常: {fallback_error}；回退为0.0%"
                )
                return 0.0

    @BaseFetcher.with_cache(ttl=300)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_hkd_cny_rate(self) -> float:
        logger.info("获取港币兑人民币汇率")
        try:
            series_df = self._get_hkd_cny_mid_series()
            rate_per_100_hkd = float(series_df.iloc[-1]["港元"])
            if rate_per_100_hkd <= 0:
                raise ValueError(f"港元中间价非法: {rate_per_100_hkd}")
            rate = rate_per_100_hkd / 100
            logger.debug(f"中间价主路径汇率: 100HKD={rate_per_100_hkd}CNY, HKD/CNY={rate}")
            return rate
        except Exception as main_error:
            logger.warning(f"中间价主路径汇率获取失败，回退至现货行情: {main_error}")
            rate = self._parse_hkd_cny_spot_rate()
            logger.debug(f"现货回退汇率 HKD/CNY: {rate}")
            return rate

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_hkd_cny_mid_history(self) -> pd.DataFrame:
        return self.historical_provider.get_fx_mid_history()

    def get_hkd_cny_daily_change_history(self, date_from: str, date_to: str) -> dict[str, dict[str, float | str]]:
        series_df = self.get_hkd_cny_mid_history()
        if series_df.empty:
            raise ValueError("历史港元中间价为空")
        required_columns = {"日期", "港元"}
        if not required_columns.issubset(set(series_df.columns)):
            raise ValueError(f"历史汇率数据缺少必要字段: {required_columns}，实际字段: {list(series_df.columns)}")

        normalized_df = series_df[["日期", "港元"]].copy()
        normalized_df["日期"] = pd.to_datetime(normalized_df["日期"], errors="coerce")
        normalized_df["港元"] = pd.to_numeric(normalized_df["港元"], errors="coerce")
        normalized_df = normalized_df.dropna(subset=["日期", "港元"])
        if normalized_df.empty:
            raise ValueError("历史港元中间价解析失败")

        normalized_df = normalized_df.sort_values("日期").drop_duplicates(subset=["日期"], keep="last")
        normalized_df["change_pct"] = normalized_df["港元"].pct_change() * 100

        history_by_date: dict[object, dict[str, float | str]] = {}
        available_dates = [item.date() for item in normalized_df["日期"].tolist()]
        for _, row in normalized_df.iterrows():
            row_date = row["日期"].date()
            history_by_date[row_date] = {
                "change_pct": float(row["change_pct"]) if not pd.isna(row["change_pct"]) else 0.0,
                "data_as_of_date": row_date.isoformat(),
            }

        output: dict[str, dict[str, float | str]] = {}
        for current_ts in pd.date_range(start=date_from, end=date_to, freq="D"):
            current_date = current_ts.date()
            eligible_dates = [item for item in available_dates if item <= current_date]
            if not eligible_dates:
                raise ValueError(f"历史汇率在 {current_date.isoformat()} 之前无可用数据")
            effective_date = eligible_dates[-1]
            payload = dict(history_by_date[effective_date])
            if effective_date != current_date:
                payload["change_pct"] = 0.0
            output[current_date.isoformat()] = payload

        return output
