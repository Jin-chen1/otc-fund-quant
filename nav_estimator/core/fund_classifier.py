"""基金分类器。"""

import re
from typing import Literal

from loguru import logger

from ..config.settings import TRACKING_TARGET_CALIBRATIONS
from ..data.fetcher.fund_fetcher import FundFetcher


FundType = Literal["index_a", "index_hk", "active_a", "active_hk", "bond_pure", "bond_plus", "qdii"]


class FundClassifier:
    """基金类型分类器。"""

    def __init__(self):
        self.fund_fetcher = FundFetcher()

    def classify(self, fund_code: str, fund_info: dict | None = None) -> FundType:
        logger.info(f"开始分类基金: {fund_code}")

        if fund_info is None:
            fund_info = self.fund_fetcher.get_fund_info(fund_code)

        fund_name = str(fund_info.get("name", "")).strip()
        fund_type_str = str(fund_info.get("type", "")).strip()
        logger.debug(f"基金名称: {fund_name}, 类型: {fund_type_str}")

        fund_name_lower = fund_name.lower()
        fund_type_lower = fund_type_str.lower()
        benchmark_text = str(fund_info.get("benchmark", "")).strip()
        benchmark_lower = benchmark_text.lower()

        hk_keywords = ["港股", "恒生", "香港", "hk", "hsi", "港股通"]
        qdii_keywords = ["qdii", "纳斯达克", "标普", "道琼斯", "海外", "全球"]

        calibrated_tracking_target = TRACKING_TARGET_CALIBRATIONS.get(str(fund_info.get("code", fund_code)).strip())
        if calibrated_tracking_target is not None and self.fund_fetcher.is_etf_or_linked_fund(fund_info):
            return "index_a"

        if self._has_effective_qdii_marker(fund_name_lower, fund_type_lower) or any(
            kw in fund_name_lower for kw in qdii_keywords
        ):
            return "qdii"

        if (
            "指数" in fund_name
            or "指数" in fund_type_str
            or "etf" in fund_name_lower
            or "联接" in fund_name
            or "指数增强" in fund_name
        ):
            if self._is_a_share_index_fund(
                fund_name=fund_name,
                benchmark_text=benchmark_text,
                benchmark_lower=benchmark_lower,
            ):
                return "index_a"
            if self._is_hk_index_fund(
                fund_name_lower=fund_name_lower,
                benchmark_text=benchmark_text,
                benchmark_lower=benchmark_lower,
                hk_keywords=hk_keywords,
            ):
                return "index_hk"
            return "index_a"

        if "债" in fund_name or "债券" in fund_type_str:
            if any(kw in fund_name for kw in ["固收+", "二级债", "偏债", "混合债"]):
                return "bond_plus"
            if any(kw in fund_name for kw in ["纯债", "短债", "中短债", "信用债", "利率债"]):
                return "bond_pure"
            if "债券型" in fund_type_str:
                return "bond_pure"
            return "bond_plus"

        if any(kw in fund_name_lower for kw in hk_keywords):
            return "active_hk"

        return "active_a"

    def _is_a_share_index_fund(
        self,
        *,
        fund_name: str,
        benchmark_text: str,
        benchmark_lower: str,
    ) -> bool:
        if any(keyword in benchmark_text or keyword in fund_name for keyword in ["A股", "恒生A股"]):
            return True
        if any(char.isdigit() for char in benchmark_text) and any(
            code.isdigit() and len(code) == 6 for code in re.findall(r"(?<!\d)(\d{6})(?!\d)", benchmark_text)
        ):
            return True
        if self.fund_fetcher.extract_benchmark_equity_index_components(benchmark_text, allowed_markets={"A股"}):
            return True
        return any(keyword in benchmark_lower for keyword in ["中证", "上证", "深证", "沪深", "创业板", "科创"])

    def _is_hk_index_fund(
        self,
        *,
        fund_name_lower: str,
        benchmark_text: str,
        benchmark_lower: str,
        hk_keywords: list[str],
    ) -> bool:
        if self.fund_fetcher.extract_benchmark_equity_index_components(benchmark_text, allowed_markets={"港股"}):
            return True
        if "a股" in benchmark_lower:
            return False
        return any(kw in fund_name_lower for kw in hk_keywords) or any(kw in benchmark_lower for kw in hk_keywords)

    @staticmethod
    def _has_effective_qdii_marker(fund_name_lower: str, fund_type_lower: str) -> bool:
        qdii_negative_markers = ("非qdii", "非 qdii")
        type_has_qdii = "qdii" in fund_type_lower and not any(
            marker in fund_type_lower for marker in qdii_negative_markers
        )
        return type_has_qdii or "qdii" in fund_name_lower

    def get_estimator_params(self, fund_code: str, fund_type: FundType) -> dict:
        return {"fund_code": fund_code, "fund_type": fund_type}
