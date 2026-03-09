"""历史数据 provider 抽象。"""

from __future__ import annotations

from typing import Dict

import akshare as ak
import pandas as pd


class HistoricalDataProvider:
    """历史数据 provider 接口。"""

    def get_a_share_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_hk_share_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_a_index_history(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_hk_index_history(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_bond_index_history(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_fx_mid_history(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_fund_nav_history(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_fund_report_announcements(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError


class AkshareHistoricalDataProvider(HistoricalDataProvider):
    """AkShare 历史数据 provider。"""

    HK_INDEX_SYMBOL_MAP: Dict[str, str] = {
        "HSI": "HSI",
        "HSTECH": "HSTECH",
    }

    def get_a_share_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="",
        )

    def get_hk_share_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return ak.stock_hk_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="",
        )

    def get_a_index_history(self, symbol: str) -> pd.DataFrame:
        return ak.stock_zh_index_daily(symbol=symbol)

    def get_hk_index_history(self, symbol: str) -> pd.DataFrame:
        resolved_symbol = self.HK_INDEX_SYMBOL_MAP.get(symbol, symbol)
        try:
            return ak.stock_hk_index_daily_em(symbol=resolved_symbol)
        except Exception:
            return ak.stock_hk_index_daily_sina(symbol=resolved_symbol)

    def get_bond_index_history(self) -> pd.DataFrame:
        return ak.bond_zh_index_daily()

    def get_fx_mid_history(self) -> pd.DataFrame:
        return ak.currency_boc_safe()

    def get_fund_nav_history(self, symbol: str) -> pd.DataFrame:
        return ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势")

    def get_fund_report_announcements(self, symbol: str) -> pd.DataFrame:
        return ak.fund_announcement_report_em(symbol=symbol)
