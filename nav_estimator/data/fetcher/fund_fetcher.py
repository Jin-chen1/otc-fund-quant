"""基金数据获取器。"""

import hashlib
import json
from datetime import datetime
import re
import threading
from typing import Any

import akshare as ak
import pandas as pd
import requests
from loguru import logger

from ...config.settings import (
    A_INDEX_ALIAS_CALIBRATIONS,
    A_SHARE_PROXY_TARGET_CALIBRATIONS,
    INDEX_ALIAS_CATALOG,
    NON_EQUITY_BENCHMARK_KEYWORDS,
    PROXY_INDEX_MAP,
    QDII_PROXY_BASKETS,
    TRACKING_TARGET_CALIBRATIONS,
)
from .base_fetcher import BaseFetcher
from .historical_provider import AkshareHistoricalDataProvider, HistoricalDataProvider


class FundFetcher(BaseFetcher):
    """基金基本信息、净值、持仓数据获取。"""

    _catalog_prewarm_lock = threading.Lock()
    _catalog_prewarm_thread: threading.Thread | None = None

    PINGZHONGDATA_HEADERS = {
        "Referer": "https://fund.eastmoney.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
    }

    HOLDINGS_REPORT_COLUMN_CANDIDATES = (
        "季度",
        "报告期",
        "报告期/日期",
        "日期",
        "报告日期",
        "截止日期",
        "公告日期",
    )
    A_INDEX_SPOT_SYMBOL_CANDIDATES = (
        "沪深重要指数",
        "上证系列指数",
        "深证系列指数",
        "中证系列指数",
        "指数成份",
    )
    GENERIC_A_INDEX_PATTERN = re.compile(
        r"([A-Za-z0-9\u4e00-\u9fa5\-]+?指数(?:（人民币）|\(人民币\)|（全价）|\(全价\))?)"
    )
    A_INDEX_CACHE_SCHEMA_VERSION = "v2"
    A_INDEX_CATALOG_CACHE_TTL = 24 * 60 * 60
    A_INDEX_CATALOG_CACHE_RETENTION_TTL = 7 * 24 * 60 * 60
    A_INDEX_CATALOG_MIN_RECORDS = 50
    A_INDEX_LOOKUP_SUCCESS_CACHE_TTL = 24 * 60 * 60
    A_INDEX_LOOKUP_NEGATIVE_CACHE_TTL = 20 * 60
    DYNAMIC_A_INDEX_MIN_NAME_LENGTH = 4
    DYNAMIC_A_INDEX_EXPLANATORY_TOKENS = (
        "跟踪",
        "标的",
        "买入",
        "追求",
        "获得",
        "通过",
        "成份股",
        "备选",
        "基金",
        "税后",
        "存款",
        "银行",
    )
    DYNAMIC_A_INDEX_EXCLUDED_CANDIDATES = {
        "综合指数",
        "股票型-标准指数",
        "标的指数",
        "紧密跟踪标的指数",
        "基金指数",
    }
    DYNAMIC_A_INDEX_LEFT_CONTEXT_EXCLUDED_TOKENS = (
        "中债",
        "国债",
        "信用债",
        "政金债",
        "存款",
        "银行",
    )
    DYNAMIC_A_INDEX_CONTEXT_PATTERN = re.compile(
        r"收益率|(?:\*|×|x|X)\s*\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*%(?:\s*(?:\*|×|x|X))?|[+＋]"
    )

    def __init__(self, historical_provider: HistoricalDataProvider | None = None):
        self.historical_provider = historical_provider or AkshareHistoricalDataProvider()

    @staticmethod
    def _parse_fee_rate(raw_value: str, fee_field_name: str, fund_code: str) -> float | None:
        normalized_value = str(raw_value).strip()
        if normalized_value == "":
            return None

        normalized_value = normalized_value.replace("％", "%")
        if normalized_value in {"--", "-", "暂无", "nan", "None"}:
            return None

        percent_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", normalized_value)
        if percent_match is not None:
            percent_value = float(percent_match.group(1))
            if percent_value < 0 or percent_value > 100:
                raise ValueError(f"基金 {fund_code} 的{fee_field_name}异常: {raw_value}")
            return percent_value / 100

        number_match = re.search(r"-?\d+(?:\.\d+)?", normalized_value)
        if number_match is None:
            return None

        numeric_value = float(number_match.group(0))
        if numeric_value < 0:
            raise ValueError(f"基金 {fund_code} 的{fee_field_name}异常: {raw_value}")
        if numeric_value > 1:
            if numeric_value > 100:
                raise ValueError(f"基金 {fund_code} 的{fee_field_name}异常: {raw_value}")
            return numeric_value / 100
        return numeric_value

    @BaseFetcher.with_cache(ttl=3600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_fund_info(self, fund_code: str) -> dict[str, Any]:
        logger.info(f"获取基金信息: {fund_code}")
        info_df = ak.fund_individual_basic_info_xq(symbol=fund_code)
        if info_df.empty:
            raise ValueError(f"未获取到基金 {fund_code} 的基本信息")
        if "item" not in info_df.columns or "value" not in info_df.columns:
            raise ValueError(f"基金信息字段异常，实际字段: {list(info_df.columns)}")

        def _pick_value(item_name: str, default_value: str = "") -> str:
            matched = info_df.loc[info_df["item"] == item_name, "value"]
            if matched.empty:
                return default_value
            return str(matched.iloc[0]).strip()

        def _pick_first_value(item_names: tuple[str, ...], default_value: str = "") -> str:
            for item_name in item_names:
                value = _pick_value(item_name)
                if value != "":
                    return value
            return default_value

        fund_name = _pick_value("基金全称") or _pick_value("基金名称")
        fund_type = _pick_value("基金类型")
        benchmark = _pick_value("业绩比较基准")
        investment_strategy = _pick_value("投资策略")
        investment_target = _pick_value("投资目标")
        management_fee_raw = _pick_first_value(("管理费率", "管理费", "基金管理费"))
        custody_fee_raw = _pick_first_value(("托管费率", "托管费", "基金托管费"))

        management_fee = self._parse_fee_rate(management_fee_raw, "管理费率", fund_code)
        custody_fee = self._parse_fee_rate(custody_fee_raw, "托管费率", fund_code)
        if not fund_name:
            raise ValueError(f"基金 {fund_code} 缺少名称字段")

        fund_info = {
            "code": fund_code,
            "name": fund_name,
            "type": fund_type,
            "benchmark": benchmark,
            "investment_strategy": investment_strategy,
            "investment_target": investment_target,
            "management_fee": management_fee,
            "custody_fee": custody_fee,
        }
        logger.debug(f"基金信息获取成功: {fund_info['name']}")
        return fund_info

    @BaseFetcher.with_cache(ttl=600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_latest_official_nav(self, fund_code: str) -> tuple[float, str]:
        logger.info(f"获取最新净值: {fund_code}")
        nav_df, date_column = self._get_fund_nav_history_df(fund_code)
        latest_row = nav_df.iloc[-1]
        latest_nav = float(latest_row["单位净值"])
        nav_date = latest_row[date_column].date().isoformat()
        logger.debug(f"最新净值: {latest_nav}, 净值日期: {nav_date}")
        return latest_nav, nav_date

    @BaseFetcher.with_cache(ttl=600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_previous_official_nav(self, fund_code: str, target_date: str | None = None) -> tuple[float, str]:
        logger.info(f"获取目标日前一交易日净值: {fund_code}, 目标日: {target_date or '今日'}")
        nav_df, date_column = self._get_fund_nav_history_df(fund_code)

        if target_date is None:
            target_dt = datetime.now().date()
        else:
            parsed_target = pd.to_datetime(target_date, errors="coerce")
            if pd.isna(parsed_target):
                raise ValueError(f"目标日期格式非法: {target_date}，应为YYYY-MM-DD或可被pandas识别的日期")
            target_dt = parsed_target.date()

        previous_df = nav_df[nav_df[date_column].dt.date < target_dt]
        if previous_df.empty:
            raise ValueError(f"基金 {fund_code} 在目标日 {target_dt.isoformat()} 之前无可用净值")

        previous_row = previous_df.sort_values(by=date_column).iloc[-1]
        previous_nav = float(previous_row["单位净值"])
        previous_nav_date = previous_row[date_column].date().isoformat()
        logger.debug(f"目标日前一交易日净值: {previous_nav}, 净值日期: {previous_nav_date}, 目标日: {target_dt.isoformat()}")
        return previous_nav, previous_nav_date

    @BaseFetcher.with_cache(ttl=3600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_fund_nav_series(self, fund_code: str, lookback_days: int, end_date: str | None = None) -> pd.DataFrame:
        if lookback_days <= 0:
            raise ValueError(f"lookback_days 必须为正整数，当前值: {lookback_days}")
        nav_df, date_column = self._get_fund_nav_history_df(fund_code)
        if end_date is not None:
            end_dt = pd.to_datetime(end_date).date()
            nav_df = nav_df[nav_df[date_column].dt.date <= end_dt].copy()
        if len(nav_df) < 2:
            raise ValueError(f"基金 {fund_code} 净值样本不足，无法构造净值序列")

        nav_df = nav_df.tail(lookback_days + 1).copy()
        if len(nav_df) < 2:
            raise ValueError(f"基金 {fund_code} 净值样本不足，无法满足回看窗口 {lookback_days}")

        nav_df[date_column] = nav_df[date_column].dt.date.astype(str)
        nav_df = nav_df.rename(columns={date_column: "date", "单位净值": "nav"})
        nav_df["nav"] = nav_df["nav"].astype(float)
        return nav_df[["date", "nav"]].reset_index(drop=True)

    @BaseFetcher.with_cache(ttl=3600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_fund_nav_return_series(self, fund_code: str, lookback_days: int, end_date: str | None = None) -> pd.DataFrame:
        if lookback_days <= 0:
            raise ValueError(f"lookback_days 必须为正整数，当前值: {lookback_days}")

        nav_series_df = self.get_fund_nav_series(fund_code, lookback_days=lookback_days, end_date=end_date)
        nav_series_df["fund_return"] = nav_series_df["nav"].pct_change() * 100
        returns_df = nav_series_df.dropna(subset=["fund_return"])[["date", "fund_return"]].copy()
        if returns_df.empty:
            raise ValueError(f"基金 {fund_code} 净值收益样本不足，无法满足回看窗口 {lookback_days}")
        returns_df["fund_return"] = returns_df["fund_return"].astype(float)
        return returns_df.reset_index(drop=True)

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_portfolio_holdings(self, fund_code: str, report_date: str | None = None) -> pd.DataFrame:
        logger.info(f"获取持仓数据: {fund_code}, 报告期: {report_date or '最新'}")
        holdings_df = self.get_portfolio_holdings_history(
            fund_code=fund_code,
            years=self._resolve_holdings_query_years(report_date),
        )
        if holdings_df.empty:
            logger.warning(f"基金 {fund_code} 未获取到持仓数据")
            return pd.DataFrame()

        if report_date is not None:
            holdings_df = self._filter_holdings_by_report_date(holdings_df, report_date)
        else:
            holdings_df = self._select_latest_holdings_period(holdings_df, fund_code)

        logger.debug(
            f"持仓数据获取成功，共 {len(holdings_df)} 条记录，"
            f"报告期={self.extract_holdings_report_metadata(holdings_df)[0]}"
        )
        return holdings_df.reset_index(drop=True)

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_portfolio_holdings_history(self, fund_code: str, years: list[int] | None = None) -> pd.DataFrame:
        if years is None:
            years = [datetime.now().year, datetime.now().year - 1]

        frames: list[pd.DataFrame] = []
        for query_year in years:
            year_df = ak.fund_portfolio_hold_em(symbol=fund_code, date=str(query_year))
            if year_df is None or year_df.empty:
                continue
            frames.append(year_df.copy())

        if not frames:
            return pd.DataFrame()

        holdings_df = pd.concat(frames, ignore_index=True)
        return self._normalize_holdings_df(holdings_df)

    @BaseFetcher.with_cache(ttl=3600)
    @BaseFetcher.retry_on_error(max_retries=3)
    def _get_pingzhongdata_text(self, fund_code: str) -> str:
        url = f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
        response = requests.get(url, headers=self.PINGZHONGDATA_HEADERS, timeout=15)
        response.raise_for_status()
        text = response.text
        if str(text).strip() == "":
            raise ValueError(f"未获取到基金 {fund_code} 的 pingzhongdata.js")
        return text

    @staticmethod
    def _extract_pingzhongdata_var(text: str, var_name: str) -> Any:
        marker = f"var {var_name} ="
        marker_index = text.find(marker)
        if marker_index == -1:
            raise ValueError(f"pingzhongdata 中未找到变量 {var_name}")

        start_index = marker_index + len(marker)
        while start_index < len(text) and text[start_index].isspace():
            start_index += 1
        if start_index >= len(text):
            raise ValueError(f"变量 {var_name} 缺少值")

        opening_char = text[start_index]
        if opening_char not in "{[":
            raise ValueError(f"变量 {var_name} 不是对象/数组字面量")

        stack = [opening_char]
        closing_map = {"{": "}", "[": "]"}
        in_string = False
        escaped = False
        end_index = None

        for idx in range(start_index + 1, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char in "{[":
                stack.append(char)
                continue
            if char in "}]":
                if not stack:
                    raise ValueError(f"变量 {var_name} 结构不完整")
                expected = closing_map[stack[-1]]
                if char != expected:
                    raise ValueError(f"变量 {var_name} 结构异常")
                stack.pop()
                if not stack:
                    end_index = idx
                    break

        if end_index is None:
            raise ValueError(f"变量 {var_name} 未找到完整结束位置")

        raw_payload = text[start_index:end_index + 1]
        try:
            return json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"变量 {var_name} JSON 解析失败: {exc}") from exc

    def _get_pingzhongdata_asset_allocation_payload(self, fund_code: str) -> dict[str, Any]:
        text = self._get_pingzhongdata_text(fund_code)
        payload = self._extract_pingzhongdata_var(text, "Data_assetAllocation")
        if not isinstance(payload, dict):
            raise ValueError(f"基金 {fund_code} 的 Data_assetAllocation 结构异常")
        return payload

    def _get_pingzhongdata_estimated_position_payload(self, fund_code: str) -> list[Any]:
        text = self._get_pingzhongdata_text(fund_code)
        payload = self._extract_pingzhongdata_var(text, "Data_fundSharesPositions")
        if not isinstance(payload, list):
            raise ValueError(f"基金 {fund_code} 的 Data_fundSharesPositions 结构异常")
        return payload

    @staticmethod
    def build_position_fallback_from_holdings(
        holdings_df: pd.DataFrame,
        default_position: float,
    ) -> tuple[float, float]:
        """根据已获取持仓推导仓位回退值。"""
        if holdings_df is None or holdings_df.empty:
            raise ValueError("持仓为空，无法根据持仓推导仓位回退值")
        if "weight" not in holdings_df.columns:
            raise ValueError("持仓数据缺少 weight 字段，无法根据持仓推导仓位回退值")

        weights = pd.to_numeric(holdings_df["weight"], errors="coerce").dropna()
        top10_weight_sum = float(weights.sum()) if not weights.empty else 0.0
        fallback_position = max(top10_weight_sum, float(default_position))
        return fallback_position, top10_weight_sum

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_fund_stock_position_snapshot(self, fund_code: str) -> dict[str, Any]:
        logger.info(f"获取股票仓位快照: {fund_code}")
        payload = self._get_pingzhongdata_asset_allocation_payload(fund_code)
        series_list = payload.get("series")
        categories = payload.get("categories")
        if not isinstance(series_list, list) or not isinstance(categories, list):
            raise ValueError(f"基金 {fund_code} 的资产配置结构异常")

        stock_series = None
        for series_item in series_list:
            if not isinstance(series_item, dict):
                continue
            if "股票" in str(series_item.get("name", "")):
                stock_series = series_item
                break
        if stock_series is None:
            raise ValueError(f"未找到基金 {fund_code} 的股票仓位信息")

        data_points = stock_series.get("data")
        if not isinstance(data_points, list):
            raise ValueError(f"基金 {fund_code} 的股票仓位数据结构异常")

        latest_snapshot = None
        for raw_report_period, raw_ratio in zip(categories, data_points):
            report_timestamp = self._normalize_report_date(raw_report_period)
            ratio_value = pd.to_numeric(raw_ratio, errors="coerce")
            if pd.isna(report_timestamp) or pd.isna(ratio_value):
                continue
            latest_snapshot = {
                "position": float(ratio_value),
                "report_date": report_timestamp.date().isoformat(),
                "raw_report_period": str(raw_report_period).strip(),
            }

        if latest_snapshot is None:
            raise ValueError(f"基金 {fund_code} 的资产配置报告期解析失败，无法确定最新股票仓位")

        snapshot = {
            "position": latest_snapshot["position"],
            "report_date": latest_snapshot["report_date"],
            "raw_report_period": latest_snapshot["raw_report_period"],
            "ratio_column": str(stock_series.get("name", "股票占净比")),
            "report_column": "categories",
            "snapshot_source": "pingzhongdata_asset_allocation",
        }
        logger.debug(
            f"股票仓位快照: fund={fund_code}, position={snapshot['position']}%, "
            f"report_date={snapshot['report_date']}, source={snapshot['snapshot_source']}"
        )
        return snapshot

    @BaseFetcher.with_cache(ttl=86400)
    @BaseFetcher.retry_on_error(max_retries=3)
    def get_estimated_stock_position_series_latest(self, fund_code: str) -> float:
        payload = self._get_pingzhongdata_estimated_position_payload(fund_code)
        latest_position = None
        latest_timestamp = None
        for row in payload:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            timestamp = pd.to_datetime(row[0], unit="ms", utc=True, errors="coerce")
            value = pd.to_numeric(row[1], errors="coerce")
            if pd.isna(timestamp) or pd.isna(value):
                continue
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_position = float(value)
        if latest_position is None:
            raise ValueError(f"基金 {fund_code} 的测算仓位序列为空")
        return latest_position

    @BaseFetcher.with_cache(ttl=86400)
    def get_fund_stock_position(self, fund_code: str) -> float:
        snapshot = self.get_fund_stock_position_snapshot(fund_code)
        return float(snapshot["position"])

    @staticmethod
    def _normalize_report_date(raw_value: Any) -> pd.Timestamp:
        if pd.isna(raw_value):
            return pd.NaT
        text = str(raw_value).strip()
        if text == "":
            return pd.NaT

        quarter_match = re.search(r"(\d{4})\s*[年/-]?\s*([1-4])\s*季", text)
        if quarter_match is not None:
            year = int(quarter_match.group(1))
            quarter = int(quarter_match.group(2))
            month_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[quarter]
            return pd.Timestamp(year=year, month=month_end[0], day=month_end[1])

        q_format_match = re.search(r"(\d{4})\s*[Qq]\s*([1-4])", text)
        if q_format_match is not None:
            year = int(q_format_match.group(1))
            quarter = int(q_format_match.group(2))
            month_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[quarter]
            return pd.Timestamp(year=year, month=month_end[0], day=month_end[1])

        parsed_date = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed_date):
            return pd.NaT
        return pd.Timestamp(parsed_date)

    @classmethod
    def extract_holdings_report_metadata(cls, holdings_df: pd.DataFrame) -> tuple[str | None, str | None]:
        if holdings_df is None or holdings_df.empty:
            return None, None

        report_period = None
        if "report_period" in holdings_df.columns:
            valid_periods = [
                str(value).strip()
                for value in holdings_df["report_period"].tolist()
                if str(value).strip() != "" and str(value).strip().lower() != "nan"
            ]
            if valid_periods:
                report_period = valid_periods[0]

        report_date_text = None
        if "report_date" in holdings_df.columns:
            report_dates = holdings_df["report_date"].dropna()
            if not report_dates.empty:
                first_date = report_dates.iloc[0]
                if isinstance(first_date, pd.Timestamp):
                    report_date_text = first_date.date().isoformat()
                else:
                    normalized = cls._normalize_report_date(first_date)
                    if not pd.isna(normalized):
                        report_date_text = normalized.date().isoformat()

        return report_period, report_date_text

    @staticmethod
    def _build_analysis_text(fund_info: dict[str, Any]) -> str:
        text_items = [
            str(fund_info.get("name", "")),
            str(fund_info.get("type", "")),
            str(fund_info.get("benchmark", "")),
            str(fund_info.get("investment_strategy", "")),
            str(fund_info.get("investment_target", "")),
        ]
        return " ".join(text_items)

    @staticmethod
    def _get_benchmark_text(fund_info: dict[str, Any]) -> str:
        return str(fund_info.get("benchmark", "")).strip()

    @staticmethod
    def _normalize_index_name(index_name: str, *, strip_index_suffix: bool = True) -> str:
        normalized = str(index_name).strip().lower()
        normalized = normalized.replace("（", "(").replace("）", ")").replace("＋", "+")
        normalized = re.sub(r"\s+", "", normalized)
        for token in (
            "收益率",
            "(人民币)",
            "（人民币）",
            "(全价)",
            "（全价）",
            "(税后)",
            "（税后）",
        ):
            normalized = normalized.replace(token, "")
        if strip_index_suffix and normalized.endswith("指数"):
            normalized = normalized[:-2]
        return normalized

    @staticmethod
    def _is_non_equity_benchmark_text(text: str) -> bool:
        normalized = str(text).strip().lower()
        return any(keyword.lower() in normalized for keyword in NON_EQUITY_BENCHMARK_KEYWORDS)

    @classmethod
    def _normalize_tracking_name_key(cls, index_name: str) -> str:
        return cls._normalize_index_name(index_name, strip_index_suffix=False)

    @classmethod
    def _get_a_index_calibration_entry(cls, index_name: str) -> dict[str, Any] | None:
        target_name = str(index_name).strip()
        if target_name == "":
            return None
        normalized_candidates = {
            cls._normalize_tracking_name_key(target_name),
            cls._normalize_index_name(target_name),
        }
        normalized_candidates.discard("")
        if not normalized_candidates:
            return None

        for entry in A_INDEX_ALIAS_CALIBRATIONS:
            entry_names = [str(entry.get("canonical_name", ""))] + [str(alias) for alias in entry.get("aliases", [])]
            entry_normalized_names = {
                cls._normalize_tracking_name_key(name)
                for name in entry_names
                if str(name).strip() != ""
            }
            entry_normalized_names.update(
                cls._normalize_index_name(name)
                for name in entry_names
                if str(name).strip() != ""
            )
            entry_normalized_names.discard("")
            if normalized_candidates & entry_normalized_names:
                return entry
        return None

    @classmethod
    def _resolve_a_index_code_from_calibration(cls, index_name: str) -> dict[str, str] | None:
        entry = cls._get_a_index_calibration_entry(index_name)
        if entry is None:
            return None
        preferred_code = str(entry.get("preferred_code", "")).strip()
        if preferred_code == "":
            return None
        return {
            "code": preferred_code,
            "name": str(entry.get("canonical_name") or index_name).strip(),
        }

    @classmethod
    def _iter_a_share_proxy_alias_entries(cls) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for item in A_SHARE_PROXY_TARGET_CALIBRATIONS:
            security_code = str(item.get("security_code", "")).strip()
            canonical_name = str(item.get("canonical_name", "")).strip()
            target_type = str(item.get("target_type", "")).strip()
            if security_code == "" or canonical_name == "" or target_type == "":
                continue
            entries.append(
                {
                    "canonical_name": canonical_name,
                    "code": security_code,
                    "quote_code": security_code,
                    "target_type": target_type,
                    "market": str(item.get("market", "A股")).strip() or "A股",
                    "tracking_name": str(item.get("tracking_name", canonical_name)).strip() or canonical_name,
                    "aliases": [str(alias) for alias in item.get("aliases", [])],
                }
            )
        return entries

    @classmethod
    def _resolve_a_share_proxy_target_from_name(cls, index_name: str) -> dict[str, Any] | None:
        target_name = str(index_name).strip()
        if target_name == "":
            return None
        normalized_candidates = {
            cls._normalize_tracking_name_key(target_name),
            cls._normalize_index_name(target_name),
        }
        normalized_candidates.discard("")
        if not normalized_candidates:
            return None
        for entry in cls._iter_a_share_proxy_alias_entries():
            entry_names = [str(entry.get("canonical_name", ""))] + [str(alias) for alias in entry.get("aliases", [])]
            entry_normalized_names = {
                cls._normalize_tracking_name_key(name)
                for name in entry_names
                if str(name).strip() != ""
            }
            entry_normalized_names.update(
                cls._normalize_index_name(name)
                for name in entry_names
                if str(name).strip() != ""
            )
            entry_normalized_names.discard("")
            if normalized_candidates & entry_normalized_names:
                quote_code = str(entry.get("quote_code") or entry.get("code", "")).strip()
                if quote_code == "":
                    return None
                return {
                    "code": quote_code,
                    "name": str(entry.get("tracking_name") or entry.get("canonical_name", "")).strip(),
                    "market": str(entry.get("market", "A股")).strip() or "A股",
                    "target_type": "a_share_etf_proxy",
                    "quote_code": quote_code,
                }
        return None

    @classmethod
    def _run_catalog_prewarm(cls, force_refresh: bool):
        try:
            cls.get_shared_a_index_catalog_snapshot(force_refresh=force_refresh)
        except Exception as error:
            logger.warning(f"A股指数目录预热失败: {error}")

    @classmethod
    def schedule_a_index_catalog_prewarm(cls, force_refresh: bool = False):
        with cls._catalog_prewarm_lock:
            thread = cls._catalog_prewarm_thread
            if thread is not None and thread.is_alive():
                return
            cls._catalog_prewarm_thread = threading.Thread(
                target=cls._run_catalog_prewarm,
                args=(force_refresh,),
                name="a-index-catalog-prewarm",
                daemon=True,
            )
            cls._catalog_prewarm_thread.start()

    @classmethod
    def _build_a_index_catalog_cache_key(cls) -> str:
        payload = {
            "args": [{"__instance_class__": cls.__name__}, list(cls.A_INDEX_SPOT_SYMBOL_CANDIDATES)],
            "kwargs": {},
            "kind": "a_index_catalog_snapshot",
            "schema_version": cls.A_INDEX_CACHE_SCHEMA_VERSION,
        }
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{cls.__module__}.{cls.__name__}:catalog:{hashlib.sha256(payload_text.encode('utf-8')).hexdigest()}"

    @classmethod
    def _is_valid_a_index_catalog_snapshot(cls, snapshot: Any) -> bool:
        if not isinstance(snapshot, dict):
            return False
        if snapshot.get("kind") != "a_index_catalog_snapshot":
            return False
        if snapshot.get("schema_version") != cls.A_INDEX_CACHE_SCHEMA_VERSION:
            return False
        records = snapshot.get("records")
        if not isinstance(records, list) or len(records) < cls.A_INDEX_CATALOG_MIN_RECORDS:
            return False
        for record in records:
            if not isinstance(record, dict):
                return False
            code = str(record.get("code", "")).strip()
            name = str(record.get("name", "")).strip()
            normalized_name = str(record.get("normalized_name", "")).strip()
            if not (len(code) == 6 and code.isdigit()):
                return False
            if name == "" or normalized_name == "":
                return False
        return True

    @classmethod
    def _get_cached_a_index_catalog_snapshot(cls) -> dict[str, Any] | None:
        cache_key = cls._build_a_index_catalog_cache_key()
        cached = BaseFetcher.cache.get(cache_key)
        if cls._is_valid_a_index_catalog_snapshot(cached):
            return cached
        return None

    @classmethod
    def _set_cached_a_index_catalog_snapshot(cls, snapshot: dict[str, Any]):
        cache_key = cls._build_a_index_catalog_cache_key()
        BaseFetcher.cache.set(
            cache_key,
            snapshot,
            ttl=cls.A_INDEX_CATALOG_CACHE_RETENTION_TTL,
        )

    @classmethod
    def _is_a_index_catalog_snapshot_fresh(cls, snapshot: dict[str, Any]) -> bool:
        refreshed_at = str(snapshot.get("refreshed_at", "")).strip()
        if refreshed_at == "":
            return False
        try:
            refreshed_at_dt = datetime.fromisoformat(refreshed_at)
        except ValueError:
            return False
        return (datetime.utcnow() - refreshed_at_dt).total_seconds() <= cls.A_INDEX_CATALOG_CACHE_TTL

    @classmethod
    def _fetch_a_index_catalog_records(cls) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        seen_records = set()
        for symbol in cls.A_INDEX_SPOT_SYMBOL_CANDIDATES:
            try:
                index_df = ak.stock_zh_index_spot_em(symbol=symbol)
            except Exception as error:
                logger.warning(f"A股指数目录拉取失败: symbol={symbol}, error={error}")
                continue
            if "名称" not in index_df.columns or "代码" not in index_df.columns:
                logger.warning(f"A股指数目录字段异常: symbol={symbol}, columns={list(index_df.columns)}")
                continue
            for _, row in index_df[["名称", "代码"]].dropna().drop_duplicates().iterrows():
                name = str(row["名称"]).strip()
                code = str(row["代码"]).strip()
                if name == "" or not (len(code) == 6 and code.isdigit()):
                    continue
                if cls._is_non_equity_benchmark_text(name):
                    continue
                normalized_name = cls._normalize_index_name(name)
                if normalized_name == "":
                    continue
                record_key = (code, name, normalized_name)
                if record_key in seen_records:
                    continue
                seen_records.add(record_key)
                records.append(
                    {
                        "code": code,
                        "name": name,
                        "normalized_name": normalized_name,
                        "source_symbol": symbol,
                    }
                )
        if len(records) < cls.A_INDEX_CATALOG_MIN_RECORDS:
            raise ValueError(f"A股指数目录快照记录数不足: {len(records)}")
        records.sort(key=lambda item: (item["normalized_name"], item["code"], item["name"]))
        return records

    @classmethod
    def get_shared_a_index_catalog_snapshot(cls, force_refresh: bool = False) -> dict[str, Any]:
        cached_snapshot = cls._get_cached_a_index_catalog_snapshot()
        if cached_snapshot is not None and not force_refresh:
            if cls._is_a_index_catalog_snapshot_fresh(cached_snapshot):
                return cached_snapshot
            cls.schedule_a_index_catalog_prewarm(force_refresh=True)
            return cached_snapshot

        try:
            snapshot = {
                "kind": "a_index_catalog_snapshot",
                "schema_version": cls.A_INDEX_CACHE_SCHEMA_VERSION,
                "refreshed_at": datetime.utcnow().isoformat(),
                "records": cls._fetch_a_index_catalog_records(),
            }
            cls._set_cached_a_index_catalog_snapshot(snapshot)
            return snapshot
        except Exception as error:
            if cached_snapshot is not None:
                logger.warning(f"刷新A股指数目录快照失败，将继续使用旧快照: {error}")
                return cached_snapshot
            raise

    @classmethod
    def _build_a_index_lookup_negative_cache_key(cls, index_name: str) -> str:
        normalized_name = cls._normalize_index_name(index_name) or str(index_name).strip()
        payload = {
            "args": [{"__instance_class__": cls.__name__}, normalized_name],
            "kwargs": {},
            "kind": "negative_lookup",
            "schema_version": cls.A_INDEX_CACHE_SCHEMA_VERSION,
        }
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return (
            f"{cls._lookup_a_index_code_by_name.__module__}.{cls._lookup_a_index_code_by_name.__name__}"
            f":negative:{hashlib.sha256(payload_text.encode('utf-8')).hexdigest()}"
        )

    @classmethod
    def _build_a_index_lookup_success_cache_key(cls, index_name: str) -> str:
        normalized_name = cls._normalize_index_name(index_name) or str(index_name).strip()
        payload = {
            "args": [{"__instance_class__": cls.__name__}, normalized_name],
            "kwargs": {},
            "kind": "success_lookup",
            "schema_version": cls.A_INDEX_CACHE_SCHEMA_VERSION,
        }
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return (
            f"{cls._lookup_a_index_code_by_name.__module__}.{cls._lookup_a_index_code_by_name.__name__}"
            f":success:{hashlib.sha256(payload_text.encode('utf-8')).hexdigest()}"
        )

    @classmethod
    def _get_cached_a_index_lookup_success(cls, index_name: str) -> dict[str, str] | None:
        cache_key = cls._build_a_index_lookup_success_cache_key(index_name)
        cached = BaseFetcher.cache.get(cache_key)
        if (
            isinstance(cached, dict)
            and cached.get("kind") == "a_index_lookup_success"
            and cached.get("schema_version") == cls.A_INDEX_CACHE_SCHEMA_VERSION
        ):
            result = cached.get("result")
            if isinstance(result, dict):
                code = str(result.get("code", "")).strip()
                name = str(result.get("name", "")).strip()
                if code != "" and name != "":
                    return {"code": code, "name": name}
        return None

    @classmethod
    def _get_cached_a_index_lookup_failure(cls, index_name: str) -> str | None:
        cache_key = cls._build_a_index_lookup_negative_cache_key(index_name)
        cached = BaseFetcher.cache.get(cache_key)
        if (
            isinstance(cached, dict)
            and cached.get("kind") == "a_index_lookup_failure"
            and cached.get("schema_version") == cls.A_INDEX_CACHE_SCHEMA_VERSION
        ):
            return str(cached.get("message", "")).strip() or None
        return None

    @classmethod
    def _set_cached_a_index_lookup_success(cls, index_name: str, resolved: dict[str, str]):
        cache_key = cls._build_a_index_lookup_success_cache_key(index_name)
        BaseFetcher.cache.set(
            cache_key,
            {
                "kind": "a_index_lookup_success",
                "schema_version": cls.A_INDEX_CACHE_SCHEMA_VERSION,
                "result": {"code": str(resolved["code"]), "name": str(resolved["name"])},
            },
            ttl=cls.A_INDEX_LOOKUP_SUCCESS_CACHE_TTL,
        )

    @classmethod
    def _set_cached_a_index_lookup_failure(cls, index_name: str, message: str):
        cache_key = cls._build_a_index_lookup_negative_cache_key(index_name)
        BaseFetcher.cache.set(
            cache_key,
            {
                "kind": "a_index_lookup_failure",
                "schema_version": cls.A_INDEX_CACHE_SCHEMA_VERSION,
                "message": str(message),
            },
            ttl=cls.A_INDEX_LOOKUP_NEGATIVE_CACHE_TTL,
        )

    @staticmethod
    def _is_negative_a_index_lookup_error(error: Exception) -> bool:
        message = str(error)
        return any(
            marker in message
            for marker in (
                "映射到多个A股指数代码",
                "无法映射到A股指数代码",
                "指数名称不能为空",
            )
        )

    @classmethod
    def _append_unresolved_equity_candidate(
        cls,
        unresolved_components: list[dict[str, Any]],
        seen_unresolved: set[tuple[str, str]],
        *,
        name: str,
        market: str,
        source_text: str,
        start: int,
        end: int,
        resolution_error: Exception,
    ):
        normalized_name = cls._normalize_index_name(name)
        if normalized_name == "":
            return
        unresolved_key = (market, normalized_name)
        if unresolved_key in seen_unresolved:
            return
        seen_unresolved.add(unresolved_key)
        unresolved_components.append(
            {
                "name": str(name),
                "market": str(market),
                "raw_weight_pct": cls._extract_weight_nearby(source_text, start, end),
                "position": start,
                "match_end": end,
                "resolution_error": str(resolution_error),
            }
        )

    @staticmethod
    def _format_unresolved_weighted_components(unresolved_components: list[dict[str, Any]]) -> str:
        return ", ".join(
            f"{item['name']}(weight={item['raw_weight_pct']})"
            for item in unresolved_components
            if item.get("raw_weight_pct") is not None
        )

    @classmethod
    def _is_formula_like_dynamic_a_index_candidate(
        cls,
        source_text: str,
        start: int,
        end: int,
        candidate_name: str,
    ) -> bool:
        candidate = str(candidate_name).strip()
        if candidate == "" or "指数" not in candidate:
            return False
        if len(candidate) < cls.DYNAMIC_A_INDEX_MIN_NAME_LENGTH:
            return False
        if cls._is_non_equity_benchmark_text(candidate):
            return False
        if candidate in cls.DYNAMIC_A_INDEX_EXCLUDED_CANDIDATES:
            return False
        if any(token in candidate for token in cls.DYNAMIC_A_INDEX_EXPLANATORY_TOKENS):
            return False

        left_context = source_text[max(0, start - 16):start]
        right_context = source_text[end:min(len(source_text), end + 16)]
        recent_left_context = left_context[-10:]
        if any(token in recent_left_context for token in cls.DYNAMIC_A_INDEX_LEFT_CONTEXT_EXCLUDED_TOKENS):
            return False
        context_text = f"{left_context}{right_context}"
        return cls.DYNAMIC_A_INDEX_CONTEXT_PATTERN.search(context_text) is not None

    @staticmethod
    def _extract_weight_nearby(source_text: str, start: int, end: int) -> float | None:
        suffix_window = source_text[end:]
        prefix_window = source_text[:start]
        suffix_match = re.search(r"(?:\*|×|x|X)?\s*(\d+(?:\.\d+)?)\s*%", suffix_window)
        if suffix_match is not None:
            return float(suffix_match.group(1))
        prefix_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:\*|×|x|X)?\s*$", prefix_window)
        if prefix_match is not None:
            return float(prefix_match.group(1))
        return None

    @staticmethod
    def _dedupe_index_records(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
        deduped: list[tuple[str, str]] = []
        seen_codes = set()
        for code, name in records:
            if code in seen_codes:
                continue
            seen_codes.add(code)
            deduped.append((code, name))
        return deduped

    @BaseFetcher.retry_on_error(max_retries=3)
    def _lookup_a_index_code_by_name(self, index_name: str) -> dict[str, str]:
        target_name = str(index_name).strip()
        calibrated = self._resolve_a_index_code_from_calibration(target_name)
        if calibrated is not None:
            self._set_cached_a_index_lookup_success(target_name, calibrated)
            return calibrated
        cached_success = self._get_cached_a_index_lookup_success(target_name)
        if cached_success is not None:
            return cached_success
        cached_failure = self._get_cached_a_index_lookup_failure(target_name)
        if cached_failure is not None:
            raise ValueError(cached_failure)
        if target_name == "":
            raise ValueError("指数名称不能为空")

        try:
            target_normalized_full = self._normalize_index_name(target_name, strip_index_suffix=False)
            target_normalized = self._normalize_index_name(target_name)
            exact_matches: list[tuple[str, str]] = []
            normalized_exact_matches: list[tuple[str, str]] = []
            normalized_stripped_matches: list[tuple[str, str]] = []
            catalog_snapshot = self.get_shared_a_index_catalog_snapshot()
            catalog_records = catalog_snapshot.get("records", [])

            for record in catalog_records:
                name = str(record.get("name", "")).strip()
                code = str(record.get("code", "")).strip()
                normalized_name = str(record.get("normalized_name", "")).strip()
                normalized_name_full = self._normalize_index_name(name, strip_index_suffix=False)
                if name == "" or normalized_name == "" or normalized_name_full == "":
                    continue
                candidate = (code, name)
                if name == target_name:
                    exact_matches.append(candidate)
                elif normalized_name_full == target_normalized_full:
                    normalized_exact_matches.append(candidate)
                elif normalized_name == target_normalized:
                    normalized_stripped_matches.append(candidate)

            exact_matches = self._dedupe_index_records(exact_matches)
            normalized_exact_matches = self._dedupe_index_records(normalized_exact_matches)
            normalized_stripped_matches = self._dedupe_index_records(normalized_stripped_matches)

            for candidates, reason in (
                (exact_matches, "精确名称"),
                (normalized_exact_matches, "规范化名称"),
                (normalized_stripped_matches, "去尾缀规范化名称"),
            ):
                if len(candidates) == 1:
                    code, matched_name = candidates[0]
                    logger.debug(f"A股指数名称查码成功: {target_name} -> {matched_name} ({code}), reason={reason}")
                    resolved = {"code": code, "name": matched_name}
                    self._set_cached_a_index_lookup_success(target_name, resolved)
                    return resolved
                if len(candidates) > 1:
                    raise ValueError(
                        f"已识别指数名称 {target_name}，但映射到多个A股指数代码: "
                        + ", ".join(f"{name}({code})" for code, name in candidates)
                    )

            raise ValueError(f"已识别指数名称 {target_name}，但无法映射到A股指数代码")
        except Exception as error:
            if self._is_negative_a_index_lookup_error(error):
                self._set_cached_a_index_lookup_failure(target_name, str(error))
            raise

    def _resolve_index_code_from_alias_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        if str(entry.get("target_type", "")).strip() == "a_share_etf_proxy":
            quote_code = str(entry.get("quote_code") or entry.get("code", "")).strip()
            if quote_code == "":
                raise ValueError(f"指数别名 {entry.get('canonical_name')} 缺少ETF代理代码")
            return {
                "code": quote_code,
                "name": str(entry.get("tracking_name") or entry.get("canonical_name", "")).strip(),
                "market": str(entry.get("market", "A股")).strip() or "A股",
                "target_type": "a_share_etf_proxy",
                "quote_code": quote_code,
            }
        code = entry.get("code")
        if code:
            return {
                "code": str(code),
                "name": str(entry["canonical_name"]),
                "market": str(entry.get("market", "")).strip(),
                "target_type": "a_index" if str(entry.get("market", "")).strip() == "A股" else "index",
            }
        if entry.get("market") != "A股":
            raise ValueError(f"指数别名 {entry.get('canonical_name')} 缺少代码")
        proxy_target = self._resolve_a_share_proxy_target_from_name(str(entry.get("canonical_name", "")))
        if proxy_target is not None:
            return proxy_target
        resolved = self._lookup_a_index_code_by_name(str(entry["canonical_name"]))
        return {
            "code": str(resolved["code"]),
            "name": str(resolved["name"]),
            "market": "A股",
            "target_type": "a_index",
        }

    def extract_benchmark_equity_index_components(
        self,
        source_text: str,
        allowed_markets: set[str] | None = None,
        *,
        return_unresolved: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        normalized_source_text = str(source_text).strip()
        if normalized_source_text == "":
            if return_unresolved:
                return [], []
            return []

        searchable_text = normalized_source_text.lower()
        raw_components: list[dict[str, Any]] = []
        unresolved_components: list[dict[str, Any]] = []
        occupied_spans: list[tuple[int, int]] = []
        seen_codes = set()
        seen_names = set()
        attempted_a_lookup_names = set()
        seen_unresolved = set()

        alias_entries: list[tuple[int, dict[str, Any], str]] = []
        for entry in INDEX_ALIAS_CATALOG:
            market = str(entry.get("market", "")).strip()
            if allowed_markets is not None and market not in allowed_markets:
                continue
            for alias in entry.get("aliases", []):
                alias_entries.append((len(str(alias)), entry, str(alias)))
        if allowed_markets is None or "A股" in allowed_markets:
            for entry in self._iter_a_share_proxy_alias_entries():
                for alias in entry.get("aliases", []):
                    alias_entries.append((len(str(alias)), entry, str(alias)))

        alias_entries.sort(key=lambda item: item[0], reverse=True)

        for _, entry, alias in alias_entries:
            alias_lower = alias.lower()
            for match in re.finditer(re.escape(alias_lower), searchable_text):
                start, end = match.span()
                if any(not (end <= span_start or start >= span_end) for span_start, span_end in occupied_spans):
                    continue
                if str(entry.get("market", "")).strip() == "A股" and not entry.get("code"):
                    attempted_name = self._normalize_index_name(str(entry.get("canonical_name", alias)))
                    if attempted_name != "":
                        if attempted_name in attempted_a_lookup_names:
                            continue
                        attempted_a_lookup_names.add(attempted_name)
                try:
                    resolved_meta = self._resolve_index_code_from_alias_entry(entry)
                except Exception as e:
                    self._append_unresolved_equity_candidate(
                        unresolved_components,
                        seen_unresolved,
                        name=str(entry.get("canonical_name", alias)),
                        market=str(entry.get("market", "")),
                        source_text=normalized_source_text,
                        start=start,
                        end=end,
                        resolution_error=e,
                    )
                    logger.debug(f"指数别名解析失败: alias={alias}, reason={e}")
                    continue
                code = str(resolved_meta["code"])
                resolved_name = str(resolved_meta["name"])
                if code in seen_codes:
                    continue
                normalized_name = self._normalize_index_name(resolved_name)
                if normalized_name in seen_names:
                    continue
                raw_components.append(
                    {
                        "code": code,
                        "name": resolved_name,
                        "market": str(resolved_meta.get("market", entry["market"])),
                        "raw_weight_pct": self._extract_weight_nearby(normalized_source_text, start, end),
                        "position": start,
                        "match_end": end,
                        "is_fallback": False,
                        "target_type": str(resolved_meta.get("target_type", "a_index")),
                        "quote_code": str(resolved_meta.get("quote_code", code)),
                    }
                )
                occupied_spans.append((start, end))
                seen_codes.add(code)
                seen_names.add(normalized_name)
                break

        if allowed_markets is None or "A股" in allowed_markets:
            for match in self.GENERIC_A_INDEX_PATTERN.finditer(normalized_source_text):
                start, end = match.span(1)
                if any(not (end <= span_start or start >= span_end) for span_start, span_end in occupied_spans):
                    continue
                candidate_name = str(match.group(1)).strip()
                if not self._is_formula_like_dynamic_a_index_candidate(
                    normalized_source_text,
                    start,
                    end,
                    candidate_name,
                ):
                    continue
                normalized_name = self._normalize_index_name(candidate_name)
                if normalized_name in seen_names or normalized_name in attempted_a_lookup_names:
                    continue
                attempted_a_lookup_names.add(normalized_name)
                try:
                    resolved = self._lookup_a_index_code_by_name(candidate_name)
                except Exception as e:
                    self._append_unresolved_equity_candidate(
                        unresolved_components,
                        seen_unresolved,
                        name=candidate_name,
                        market="A股",
                        source_text=normalized_source_text,
                        start=start,
                        end=end,
                        resolution_error=e,
                    )
                    logger.debug(f"动态A股指数查码失败: {candidate_name}, reason={e}")
                    continue
                code = str(resolved["code"])
                if code in seen_codes:
                    continue
                raw_components.append(
                    {
                        "code": code,
                        "name": str(resolved["name"]),
                        "market": "A股",
                        "raw_weight_pct": self._extract_weight_nearby(normalized_source_text, start, end),
                        "position": start,
                        "match_end": end,
                        "is_fallback": False,
                    }
                )
                occupied_spans.append((start, end))
                seen_codes.add(code)
                seen_names.add(normalized_name)

        raw_components.sort(key=lambda item: item["position"])
        if raw_components:
            logger.debug(
                "解析权益基准组件: "
                + ", ".join(
                    f"{item['name']}({item['code']},{item['market']},weight={item['raw_weight_pct']})"
                    for item in raw_components
                )
            )
        if return_unresolved:
            return raw_components, unresolved_components
        return raw_components

    def _normalize_weighted_proxy_components(
        self,
        raw_components: list[dict[str, Any]],
        source_text: str,
        fund_info: dict[str, Any],
        *,
        missing_weight_message: str,
        empty_component_message: str,
    ) -> list[dict[str, Any]]:
        if not raw_components:
            raise ValueError(empty_component_message)

        normalized_source_text = str(source_text)
        raw_components = sorted(raw_components, key=lambda item: item["position"])
        for idx, component in enumerate(raw_components):
            if component.get("is_fallback", False):
                component["raw_weight_pct"] = None
                continue
            left_boundary = 0 if idx == 0 else raw_components[idx - 1]["match_end"]
            right_boundary = len(normalized_source_text) if idx == len(raw_components) - 1 else raw_components[idx + 1]["position"]
            left_boundary = min(left_boundary, component["position"])
            right_boundary = max(right_boundary, component["match_end"])
            local_text = normalized_source_text[left_boundary:right_boundary]
            local_start = component["position"] - left_boundary
            local_end = component["match_end"] - left_boundary
            component["raw_weight_pct"] = self._extract_weight_nearby(local_text, local_start, local_end)

        explicit_weights = []
        missing_weight_count = 0
        for component in raw_components:
            weight = component["raw_weight_pct"]
            if weight is None:
                missing_weight_count += 1
                continue
            if weight <= 0:
                raise ValueError(f"基金 {fund_info.get('code', '')} 的业绩比较基准权重非法(<=0): {weight}")
            explicit_weights.append(weight)

        if missing_weight_count > 0 and explicit_weights and len(raw_components) > 1:
            raise ValueError(missing_weight_message)

        components: list[dict[str, Any]] = []
        if explicit_weights:
            weight_sum = sum(explicit_weights)
            if weight_sum <= 0:
                raise ValueError(f"基金 {fund_info.get('code', '')} 的业绩比较基准权重和非法: {weight_sum}")
            for component in raw_components:
                weight = component["raw_weight_pct"] or 0.0
                components.append(
                    {
                        "code": component["code"],
                        "name": component["name"],
                        "market": component["market"],
                        "weight": weight / weight_sum,
                        "target_type": component.get("target_type", "a_index"),
                        "quote_code": component.get("quote_code", component["code"]),
                    }
                )
            return components

        equal_weight = 1.0 / len(raw_components)
        for component in raw_components:
            components.append(
                {
                    "code": component["code"],
                    "name": component["name"],
                    "market": component["market"],
                    "weight": equal_weight,
                    "target_type": component.get("target_type", "a_index"),
                    "quote_code": component.get("quote_code", component["code"]),
                }
            )
        return components

    def resolve_qdii_market_profile(self, fund_info: dict[str, Any]) -> str:
        analysis_text = self._build_analysis_text(fund_info)
        analysis_text_lower = analysis_text.lower()

        hk_keywords = ["港股", "香港", "恒生", "港股通", "hsi", "hstech", "hang seng"]
        us_keywords = ["纳斯达克", "nasdaq", "标普", "s&p", "道琼斯", "dow jones", "美股", "美国", "msci美国"]
        global_keywords = ["全球", "世界", "亚太", "欧洲", "德国", "法国", "日本", "印度", "越南", "东南亚", "新兴市场", "发达市场"]

        has_hk = any(keyword in analysis_text_lower for keyword in hk_keywords)
        has_us = any(keyword in analysis_text_lower for keyword in us_keywords)
        has_global = any(keyword in analysis_text_lower for keyword in global_keywords)

        if has_global:
            return "unsupported"
        if has_hk and has_us:
            return "hk_us_mixed"
        if has_hk:
            return "hk"
        if has_us:
            return "us"
        return "unsupported"

    @staticmethod
    def _copy_proxy_basket(basket_key: str) -> list[dict[str, Any]]:
        basket = QDII_PROXY_BASKETS[basket_key]
        return [
            {
                "code": item["code"],
                "name": item["name"],
                "market": item["market"],
                "weight": float(item["weight"]),
            }
            for item in basket
        ]

    def resolve_qdii_proxy_components(self, fund_info: dict[str, Any]) -> list[dict[str, Any]]:
        benchmark_text = str(fund_info.get("benchmark", "")).strip()
        analysis_text = self._build_analysis_text(fund_info)
        source_text = benchmark_text or analysis_text
        if source_text.strip() == "":
            raise ValueError(f"基金 {fund_info.get('code', '')} 缺少业绩比较基准与策略文本，无法解析QDII代理指数")

        def _extract_weight_nearby(text: str, start: int, end: int) -> float | None:
            suffix_window = text[end:]
            prefix_window = text[:start]
            suffix_match = re.search(r"(?:\*|×|x|X)?\s*(\d+(?:\.\d+)?)\s*%", suffix_window)
            if suffix_match is not None:
                return float(suffix_match.group(1))
            prefix_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:\*|×|x|X)?\s*$", prefix_window)
            if prefix_match is not None:
                return float(prefix_match.group(1))
            return None

        keyword_candidates = [
            ("恒生科技指数", "恒生科技", "HSTECH", "港股"),
            ("恒生科技", "恒生科技", "HSTECH", "港股"),
            ("hstech", "恒生科技", "HSTECH", "港股"),
            ("恒生指数", "恒生指数", "HSI", "港股"),
            ("恒生综合", "恒生指数", "HSI", "港股"),
            ("hsi", "恒生指数", "HSI", "港股"),
            ("标普500", "标普500", "SPX", "美股"),
            ("s&p500", "标普500", "SPX", "美股"),
            ("s&p 500", "标普500", "SPX", "美股"),
            ("sp500", "标普500", "SPX", "美股"),
            ("纳斯达克100", "纳斯达克100", "NDX", "美股"),
            ("纳指100", "纳斯达克100", "NDX", "美股"),
            ("nasdaq100", "纳斯达克100", "NDX", "美股"),
            ("nasdaq 100", "纳斯达克100", "NDX", "美股"),
            ("纳斯达克", "纳斯达克100", "NDX", "美股"),
            ("道琼斯", "道琼斯", "DJIA", "美股"),
            ("dow jones", "道琼斯", "DJIA", "美股"),
        ]

        raw_components: list[dict[str, Any]] = []
        seen_codes = set()
        searchable_text = source_text.lower()
        for keyword, name, code, market in keyword_candidates:
            if code in seen_codes:
                continue
            matched = re.search(re.escape(keyword.lower()), searchable_text)
            if matched is None:
                continue
            raw_components.append(
                {
                    "code": code,
                    "name": name,
                    "market": market,
                    "raw_weight_pct": _extract_weight_nearby(source_text, matched.start(), matched.end()),
                    "position": matched.start(),
                    "match_end": matched.end(),
                }
            )
            seen_codes.add(code)

        if raw_components:
            raw_components = sorted(raw_components, key=lambda item: item["position"])
            for idx, component in enumerate(raw_components):
                left_boundary = 0 if idx == 0 else raw_components[idx - 1]["match_end"]
                right_boundary = len(source_text) if idx == len(raw_components) - 1 else raw_components[idx + 1]["position"]
                local_text = source_text[left_boundary:right_boundary]
                local_start = component["position"] - left_boundary
                local_end = component["match_end"] - left_boundary
                component["raw_weight_pct"] = _extract_weight_nearby(local_text, local_start, local_end)

            explicit_weights = [float(item["raw_weight_pct"]) for item in raw_components if item["raw_weight_pct"] is not None]
            missing_weight_count = sum(item["raw_weight_pct"] is None for item in raw_components)
            if missing_weight_count > 0 and explicit_weights and len(raw_components) > 1:
                raise ValueError(f"基金 {fund_info.get('code', '')} 的业绩比较基准存在部分指数缺失权重，无法确定QDII代理指数组合")

            if explicit_weights:
                total_weight = sum(explicit_weights)
                if total_weight <= 0:
                    raise ValueError(f"基金 {fund_info.get('code', '')} 的业绩比较基准权重和非法: {total_weight}")
                return [
                    {
                        "code": item["code"],
                        "name": item["name"],
                        "market": item["market"],
                        "weight": float(item["raw_weight_pct"]) / total_weight,
                    }
                    for item in raw_components
                ]

            equal_weight = 1.0 / len(raw_components)
            return [
                {
                    "code": item["code"],
                    "name": item["name"],
                    "market": item["market"],
                    "weight": equal_weight,
                }
                for item in raw_components
            ]

        profile = self.resolve_qdii_market_profile(fund_info)
        if profile == "hk":
            return self._copy_proxy_basket("hk_broad")
        if profile == "us":
            return self._copy_proxy_basket("us_broad")
        if profile == "hk_us_mixed":
            return self._copy_proxy_basket("hk_us_balanced")
        raise ValueError(f"基金 {fund_info.get('code', '')} 无法从业绩比较基准识别高质量代理指数或代理篮子")

    def resolve_a_index_code(self, fund_info: dict[str, Any]) -> str:
        benchmark_text = self._get_benchmark_text(fund_info)
        raw_components, unresolved_components = self.extract_benchmark_equity_index_components(
            benchmark_text,
            allowed_markets={"A股"},
            return_unresolved=True,
        )
        unresolved_weighted_components = [
            item for item in unresolved_components if item["market"] == "A股" and item.get("raw_weight_pct") is not None
        ]
        if unresolved_weighted_components:
            unresolved_text = self._format_unresolved_weighted_components(unresolved_weighted_components)
            raise ValueError(f"基金 {fund_info.get('code', '')} 的业绩比较基准存在未解析指数成分: {unresolved_text}")
        a_index_components = [item for item in raw_components if item.get("target_type", "a_index") == "a_index"]
        if a_index_components:
            return str(a_index_components[0]["code"])
        code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", benchmark_text)
        if code_match:
            return code_match.group(1)
        raise ValueError(f"基金 {fund_info.get('code', '')} 未识别到任何A股权益指数名称，请检查业绩比较基准字段")

    def resolve_hk_index_code(self, fund_info: dict[str, Any]) -> str:
        benchmark_text = self._get_benchmark_text(fund_info)
        raw_components, unresolved_components = self.extract_benchmark_equity_index_components(
            benchmark_text,
            allowed_markets={"港股"},
            return_unresolved=True,
        )
        unresolved_weighted_components = [
            item for item in unresolved_components if item["market"] == "港股" and item.get("raw_weight_pct") is not None
        ]
        if unresolved_weighted_components:
            unresolved_text = self._format_unresolved_weighted_components(unresolved_weighted_components)
            raise ValueError(f"基金 {fund_info.get('code', '')} 的业绩比较基准存在未解析指数成分: {unresolved_text}")
        if raw_components:
            return str(raw_components[0]["code"])
        raise ValueError(f"基金 {fund_info.get('code', '')} 未识别到任何港股权益指数名称，请检查业绩比较基准字段")

    def resolve_index_tracking_target(self, fund_info: dict[str, Any]) -> dict[str, Any]:
        fund_code = str(fund_info.get("code", "")).strip()
        calibration = TRACKING_TARGET_CALIBRATIONS.get(fund_code)
        if calibration is not None and self.is_etf_or_linked_fund(fund_info):
            return {
                "target_type": str(calibration["target_type"]),
                "security_code": str(calibration["security_code"]),
                "market": str(calibration.get("market", "A股")),
                "tracking_name": str(calibration.get("tracking_name", fund_info.get("name", ""))).strip(),
            }

        return {
            "target_type": "a_index",
            "code": self.resolve_a_index_code(fund_info),
            "market": "A股",
            "tracking_name": "",
        }

    def resolve_active_proxy_components(self, fund_info: dict[str, Any]) -> list[dict[str, Any]]:
        benchmark_text = str(fund_info.get("benchmark", "")).strip()
        if benchmark_text == "":
            raise ValueError(f"基金 {fund_info.get('code', '')} 缺少业绩比较基准，无法解析主动权益代理指数")
        raw_components, unresolved_components = self.extract_benchmark_equity_index_components(
            benchmark_text,
            allowed_markets={"A股", "港股"},
            return_unresolved=True,
        )
        strict_mode = BaseFetcher._get_strict_mode_latched()
        unresolved_weighted_components = [item for item in unresolved_components if item.get("raw_weight_pct") is not None]
        if strict_mode is not False and unresolved_weighted_components:
            unresolved_text = self._format_unresolved_weighted_components(unresolved_weighted_components)
            raise ValueError(f"基金 {fund_info.get('code', '')} 的权益业绩基准存在未解析指数成分: {unresolved_text}")
        return self._normalize_weighted_proxy_components(
            raw_components,
            benchmark_text,
            fund_info,
            missing_weight_message=f"基金 {fund_info.get('code', '')} 的权益业绩基准存在部分指数缺失权重，无法确定代理指数组合",
            empty_component_message=f"基金 {fund_info.get('code', '')} 未识别到任何权益指数名称，请检查业绩比较基准字段",
        )

    def is_index_fund(self, fund_info: dict[str, Any]) -> bool:
        analysis_text = self._build_analysis_text(fund_info).lower()
        return any(keyword in analysis_text for keyword in ["指数", "etf", "联接", "index", "被动"])

    def is_etf_or_linked_fund(self, fund_info: dict[str, Any]) -> bool:
        analysis_text = self._build_analysis_text(fund_info).lower()
        fund_name = str(fund_info.get("name", "")).lower()
        fund_type = str(fund_info.get("type", "")).lower()
        keyword_hits = ["etf", "联接", "交易型开放式"]
        return any(keyword in analysis_text for keyword in keyword_hits) or any(
            keyword in fund_name or keyword in fund_type for keyword in keyword_hits
        )

    def is_index_enhanced_fund(self, fund_info: dict[str, Any]) -> bool:
        analysis_text = self._build_analysis_text(fund_info).lower()
        return any(keyword in analysis_text for keyword in ["指数增强", "增强指数", "enhanced"])

    @staticmethod
    def _get_fund_nav_history_df(fund_code: str) -> tuple[pd.DataFrame, str]:
        provider = AkshareHistoricalDataProvider()
        nav_df = provider.get_fund_nav_history(symbol=fund_code)
        if nav_df.empty:
            raise ValueError(f"未获取到基金 {fund_code} 的净值数据")
        if "单位净值" not in nav_df.columns:
            raise ValueError(f"基金净值字段异常，实际字段: {list(nav_df.columns)}")

        date_column = None
        for candidate in ["净值日期", "日期"]:
            if candidate in nav_df.columns:
                date_column = candidate
                break
        if date_column is None:
            raise ValueError(f"基金净值日期字段异常，实际字段: {list(nav_df.columns)}")

        nav_df = nav_df.copy()
        nav_df[date_column] = pd.to_datetime(nav_df[date_column], errors="coerce")
        nav_df["单位净值"] = pd.to_numeric(nav_df["单位净值"], errors="coerce")
        nav_df = nav_df.dropna(subset=[date_column, "单位净值"])
        if nav_df.empty:
            raise ValueError(f"基金 {fund_code} 的净值走势解析失败")
        nav_df = nav_df.sort_values(by=date_column).drop_duplicates(subset=[date_column], keep="last")
        return nav_df.reset_index(drop=True), date_column

    @staticmethod
    def _resolve_holdings_query_years(report_date: str | None) -> list[int]:
        if report_date is None:
            current_year = datetime.now().year
            return [current_year, current_year - 1]

        normalized_report_date = FundFetcher._normalize_report_date(report_date)
        if not pd.isna(normalized_report_date):
            return [int(normalized_report_date.year)]

        year_match = re.search(r"(\d{4})", str(report_date))
        if year_match is None:
            raise ValueError(f"无法从报告期识别年份: {report_date}")
        return [int(year_match.group(1))]

    @classmethod
    def _normalize_holdings_df(cls, holdings_df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "股票代码": "code",
            "股票名称": "name",
            "占净值比例": "weight",
            "持股数": "shares",
            "持仓数": "shares",
        }
        normalized_df = holdings_df.rename(
            columns={key: value for key, value in rename_map.items() if key in holdings_df.columns}
        ).copy()

        for required_col in ["code", "name", "weight"]:
            if required_col not in normalized_df.columns:
                raise ValueError(f"持仓数据缺少必要字段 {required_col}，实际字段: {list(normalized_df.columns)}")

        report_source_column = None
        for candidate in cls.HOLDINGS_REPORT_COLUMN_CANDIDATES:
            if candidate in normalized_df.columns:
                report_source_column = candidate
                break

        if report_source_column is not None:
            normalized_df["report_period"] = normalized_df[report_source_column].astype(str).str.strip()
            normalized_df["report_date"] = normalized_df["report_period"].apply(cls._normalize_report_date)
        else:
            normalized_df["report_period"] = ""
            normalized_df["report_date"] = pd.NaT

        normalized_df["code"] = normalized_df["code"].astype(str).str.strip()
        normalized_df["weight"] = normalized_df["weight"].astype(str).str.rstrip("%").astype(float)

        def classify_market(code: str) -> str:
            if code.startswith(("0", "3", "6")) and len(code) == 6:
                return "A股"
            if len(code) == 5 and code.isdigit():
                return "港股"
            return "其他"

        normalized_df["market"] = normalized_df["code"].apply(classify_market)
        return normalized_df

    @classmethod
    def _filter_holdings_by_report_date(cls, holdings_df: pd.DataFrame, report_date: str) -> pd.DataFrame:
        if holdings_df.empty:
            return holdings_df
        target_report_date = cls._normalize_report_date(report_date)
        if not pd.isna(target_report_date) and "report_date" in holdings_df.columns:
            filtered_df = holdings_df[holdings_df["report_date"] == target_report_date].copy()
            if not filtered_df.empty:
                return filtered_df
        if "report_period" in holdings_df.columns:
            filtered_df = holdings_df[holdings_df["report_period"].astype(str).str.strip() == str(report_date).strip()].copy()
            if not filtered_df.empty:
                return filtered_df
        return holdings_df.iloc[0:0].copy()

    @classmethod
    def _select_latest_holdings_period(cls, holdings_df: pd.DataFrame, fund_code: str) -> pd.DataFrame:
        if holdings_df.empty:
            return holdings_df
        if "report_date" not in holdings_df.columns:
            raise ValueError(f"基金 {fund_code} 的持仓数据缺少 report_date 字段，无法定位最新报告期")
        valid_df = holdings_df.dropna(subset=["report_date"]).copy()
        if valid_df.empty:
            raise ValueError(f"基金 {fund_code} 的持仓报告期解析失败，无法确定最新报告期")
        latest_report_date = valid_df["report_date"].max()
        latest_df = valid_df[valid_df["report_date"] == latest_report_date].copy()
        if latest_df.empty:
            raise ValueError(f"基金 {fund_code} 的最新持仓报告期为空")
        return latest_df
