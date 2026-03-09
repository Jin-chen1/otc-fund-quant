"""全局配置文件。"""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_FEES = {
    "equity": {"management": 0.015, "custody": 0.002},
    "index": {"management": 0.005, "custody": 0.001},
    "bond_pure": {"management": 0.006, "custody": 0.002},
    "bond_plus": {"management": 0.008, "custody": 0.002},
    "qdii": {"management": 0.015, "custody": 0.003},
}

DEFAULT_POSITION = {
    "index_etf": 97.0,
    "index_regular": 93.0,
    "equity_active": 85.0,
    "bond_plus_stock": 15.0,
    "bond_plus_convertible": 10.0,
}

INDEX_ENHANCED_REGRESSION = {
    "lookback_days": 60,
    "min_samples": 20,
    "beta_min": 0.50,
    "beta_max": 1.00,
}

BOND_PLUS_REGRESSION = {
    "lookback_days": 60,
    "min_samples": 20,
    "weight_min": 0.0,
    "weight_max": 1.0,
}

BOND_STALE_FALLBACK = {
    "duration": 3.0,
    "yield_tenor": "10年",
    "default_ytm": 2.0,
}

STRICT_MODE = True
FX_DAILY_CHANGE_MAX_ABS = 5.0

CACHE_DIR = PROJECT_ROOT / "data" / "nav_estimator_cache"
CACHE_TTL_SECONDS = 300
PROXY_MODE = os.environ.get("OTC_FUND_QUANT_PROXY_MODE", "auto").strip().lower() or "auto"
if PROXY_MODE not in {"auto", "env", "direct"}:
    PROXY_MODE = "auto"

AKSHARE_TIMEOUT = 10
AKSHARE_RETRIES = 3

PROXY_INDEX_MAP = {
    "沪深300": "000300",
    "中证500": "000905",
    "中证1000": "000852",
    "中证转债": "000832",
    "上证50": "000016",
    "中证白酒": "399997",
    "创业板指": "399006",
    "科创50": "000688",
    "恒生综合": "HSI",
    "恒生指数": "HSI",
    "恒生科技": "HSTECH",
    "标普500": "SPX",
    "纳斯达克100": "NDX",
    "纳斯达克": "NDX",
    "道琼斯": "DJIA",
}

QDII_PROXY_BASKETS = {
    "hk_broad": [
        {"code": "HSI", "name": "恒生指数", "market": "港股", "weight": 0.7},
        {"code": "HSTECH", "name": "恒生科技", "market": "港股", "weight": 0.3},
    ],
    "us_broad": [
        {"code": "SPX", "name": "标普500", "market": "美股", "weight": 0.7},
        {"code": "NDX", "name": "纳斯达克100", "market": "美股", "weight": 0.3},
    ],
    "hk_us_balanced": [
        {"code": "HSI", "name": "恒生指数", "market": "港股", "weight": 0.35},
        {"code": "HSTECH", "name": "恒生科技", "market": "港股", "weight": 0.15},
        {"code": "SPX", "name": "标普500", "market": "美股", "weight": 0.35},
        {"code": "NDX", "name": "纳斯达克100", "market": "美股", "weight": 0.15},
    ],
}

BOND_INDEX_TYPES = {
    "comprehensive": "中债综合全价指数",
    "treasury": "中债国债全价指数",
    "credit": "中债信用债全价指数",
}

BOND_PLUS_MODEL = {
    "convertible_index_code": "000832",
    "default_convertible_position": 15.0,
    "default_treasury_ytm": 2.20,
    "default_duration": 3.0,
    "regression": {
        "enabled": True,
        "lookback_days": 60,
        "min_samples": 20,
        "stock_weight_max": 0.50,
        "convertible_weight_max": 0.50,
    },
}


class Settings:
    """配置类。"""

    def __init__(self):
        self.base_dir = BASE_DIR
        self.cache_dir = CACHE_DIR
        self.cache_ttl = CACHE_TTL_SECONDS
        self.proxy_mode = PROXY_MODE
        self.default_fees = DEFAULT_FEES
        self.default_position = DEFAULT_POSITION
        self.index_enhanced_regression = INDEX_ENHANCED_REGRESSION
        self.bond_plus_regression = BOND_PLUS_REGRESSION
        self.bond_stale_fallback = BOND_STALE_FALLBACK
        self.proxy_index_map = PROXY_INDEX_MAP
        self.qdii_proxy_baskets = QDII_PROXY_BASKETS
        self.bond_index_types = BOND_INDEX_TYPES
        self.bond_plus_model = BOND_PLUS_MODEL
        self.strict_mode = STRICT_MODE
        self.fx_daily_change_max_abs = FX_DAILY_CHANGE_MAX_ABS
        self.cache_dir.mkdir(parents=True, exist_ok=True)
