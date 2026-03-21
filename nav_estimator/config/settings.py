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
    "中证800": "000906",
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

INDEX_ALIAS_CATALOG = [
    {"canonical_name": "沪深300", "code": "000300", "market": "A股", "aliases": ["沪深300", "沪深300指数"]},
    {"canonical_name": "中证800", "code": "000906", "market": "A股", "aliases": ["中证800", "中证800指数"]},
    {"canonical_name": "中证500", "code": "000905", "market": "A股", "aliases": ["中证500", "中证500指数"]},
    {"canonical_name": "中证1000", "code": "000852", "market": "A股", "aliases": ["中证1000", "中证1000指数"]},
    {"canonical_name": "上证50", "code": "000016", "market": "A股", "aliases": ["上证50", "上证50指数"]},
    {"canonical_name": "创业板指", "code": "399006", "market": "A股", "aliases": ["创业板指", "创业板指数"]},
    {"canonical_name": "科创50", "code": "000688", "market": "A股", "aliases": ["科创50", "科创50指数"]},
    {"canonical_name": "中证白酒", "code": "399997", "market": "A股", "aliases": ["中证白酒", "中证白酒指数"]},
    {"canonical_name": "中证医药卫生", "code": None, "market": "A股", "aliases": ["中证医药卫生", "中证医药卫生指数"]},
    {"canonical_name": "中证机器人", "code": None, "market": "A股", "aliases": ["中证机器人", "中证机器人指数"]},
    {"canonical_name": "中证人工智能主题", "code": None, "market": "A股", "aliases": ["中证人工智能主题", "中证人工智能主题指数"]},
    {"canonical_name": "中证港股通综合", "code": None, "market": "A股", "aliases": ["中证港股通综合", "中证港股通综合指数"]},
    {"canonical_name": "中证全指半导体产品与设备", "code": None, "market": "A股", "aliases": ["中证全指半导体产品与设备", "中证全指半导体产品与设备指数"]},
    {"canonical_name": "恒生A股电网设备", "code": None, "market": "A股", "aliases": ["恒生A股电网设备", "恒生A股电网设备指数"]},
    {"canonical_name": "恒生科技", "code": "HSTECH", "market": "港股", "aliases": ["恒生科技指数", "恒生科技", "hstech"]},
    {"canonical_name": "恒生指数", "code": "HSI", "market": "港股", "aliases": ["恒生指数", "恒生综合", "hsi", "hang seng"]},
    {"canonical_name": "标普500", "code": "SPX", "market": "美股", "aliases": ["标普500", "标普500指数", "s&p500", "s&p 500", "sp500"]},
    {"canonical_name": "纳斯达克100", "code": "NDX", "market": "美股", "aliases": ["纳斯达克100", "纳指100", "nasdaq100", "nasdaq 100", "纳斯达克"]},
    {"canonical_name": "道琼斯", "code": "DJIA", "market": "美股", "aliases": ["道琼斯", "dow jones"]},
]

A_INDEX_ALIAS_CALIBRATIONS = [
    {
        "canonical_name": "中证医药卫生",
        "preferred_code": "000933",
        "alternate_codes": ["399933"],
        "quote_channel": "a_share_numeric",
        "market": "A股",
        "aliases": ["中证医药卫生", "中证医药卫生指数"],
    },
    {
        "canonical_name": "中证港股通综合",
        "preferred_code": "930930",
        "alternate_codes": [],
        "quote_channel": "a_share_numeric",
        "market": "A股",
        "aliases": ["中证港股通综合", "中证港股通综合指数"],
    },
]

A_SHARE_PROXY_TARGET_CALIBRATIONS = [
    {
        "canonical_name": "中证人工智能主题",
        "target_type": "a_share_etf_proxy",
        "security_code": "159819",
        "tracking_name": "人工智能ETF",
        "market": "A股",
        "aliases": ["中证人工智能主题", "中证人工智能主题指数"],
    },
    {
        "canonical_name": "中证红利低波动",
        "target_type": "a_share_etf_proxy",
        "security_code": "512890",
        "tracking_name": "红利低波ETF",
        "market": "A股",
        "aliases": ["中证红利低波动", "中证红利低波动指数"],
    },
    {
        "canonical_name": "中证港股通创新药",
        "target_type": "a_share_etf_proxy",
        "security_code": "513780",
        "tracking_name": "港股创新药ETF",
        "market": "A股",
        "aliases": ["中证港股通创新药", "中证港股通创新药指数"],
    },
    {
        "canonical_name": "创业板人工智能",
        "target_type": "a_share_etf_proxy",
        "security_code": "159246",
        "tracking_name": "创业板人工智能ETF富国",
        "market": "A股",
        "aliases": ["创业板人工智能", "创业板人工智能指数"],
    },
    {
        "canonical_name": "中证全指半导体产品与设备",
        "target_type": "a_share_etf_proxy",
        "security_code": "512480",
        "tracking_name": "半导体ETF",
        "market": "A股",
        "aliases": ["中证全指半导体产品与设备", "中证全指半导体产品与设备指数"],
    },
    {
        "canonical_name": "中证港股通科技",
        "target_type": "a_share_etf_proxy",
        "security_code": "513020",
        "tracking_name": "国泰中证港股通科技ETF",
        "market": "A股",
        "aliases": ["中证港股通科技", "中证港股通科技指数"],
    },
    {
        "canonical_name": "国证新能源电池",
        "target_type": "a_share_etf_proxy",
        "security_code": "159566",
        "tracking_name": "易方达国证新能源电池ETF",
        "market": "A股",
        "aliases": ["国证新能源电池", "国证新能源电池指数"],
    },
    {
        "canonical_name": "中证沪深港黄金产业股票",
        "target_type": "a_share_etf_proxy",
        "security_code": "159562",
        "tracking_name": "华夏中证沪深港黄金产业股票ETF",
        "market": "A股",
        "aliases": ["中证沪深港黄金产业股票", "中证沪深港黄金产业股票指数"],
    },
]

TRACKING_TARGET_CALIBRATIONS = {
    "012733": {
        "target_type": "linked_etf_a_share",
        "security_code": "159819",
        "tracking_name": "人工智能ETF",
        "market": "A股",
    },
    "012734": {
        "target_type": "linked_etf_a_share",
        "security_code": "159819",
        "tracking_name": "人工智能ETF",
        "market": "A股",
    },
    "007467": {
        "target_type": "linked_etf_a_share",
        "security_code": "512890",
        "tracking_name": "红利低波ETF",
        "market": "A股",
    },
    "020640": {
        "target_type": "linked_etf_a_share",
        "security_code": "560780",
        "tracking_name": "广发中证半导体材料设备主题ETF",
        "market": "A股",
    },
    "018345": {
        "target_type": "linked_etf_a_share",
        "security_code": "562500",
        "tracking_name": "中证机器人ETF",
        "market": "A股",
    },
    "023639": {
        "target_type": "linked_etf_a_share",
        "security_code": "560880",
        "tracking_name": "恒生A股电网设备ETF",
        "market": "A股",
    },
    "023598": {
        "target_type": "linked_etf_a_share",
        "security_code": "513780",
        "tracking_name": "港股创新药ETF",
        "market": "A股",
    },
    "024663": {
        "target_type": "linked_etf_a_share",
        "security_code": "159246",
        "tracking_name": "创业板人工智能ETF富国",
        "market": "A股",
    },
    "015740": {
        "target_type": "linked_etf_a_share",
        "security_code": "513020",
        "tracking_name": "国泰中证港股通科技ETF",
        "market": "A股",
    },
    "021034": {
        "target_type": "linked_etf_a_share",
        "security_code": "159566",
        "tracking_name": "易方达国证新能源电池ETF",
        "market": "A股",
    },
    "021363": {
        "target_type": "linked_etf_a_share",
        "security_code": "159562",
        "tracking_name": "华夏中证沪深港黄金产业股票ETF",
        "market": "A股",
        "allow_non_linked": True,
    },
}

NON_EQUITY_BENCHMARK_KEYWORDS = [
    "中债",
    "国债",
    "信用债",
    "债券",
    "存款",
    "活期",
    "定期",
    "银行",
    "税后",
    "现金",
    "货币市场",
]

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
        self.index_alias_catalog = INDEX_ALIAS_CATALOG
        self.a_index_alias_calibrations = A_INDEX_ALIAS_CALIBRATIONS
        self.a_share_proxy_target_calibrations = A_SHARE_PROXY_TARGET_CALIBRATIONS
        self.tracking_target_calibrations = TRACKING_TARGET_CALIBRATIONS
        self.non_equity_benchmark_keywords = NON_EQUITY_BENCHMARK_KEYWORDS
        self.qdii_proxy_baskets = QDII_PROXY_BASKETS
        self.bond_index_types = BOND_INDEX_TYPES
        self.bond_plus_model = BOND_PLUS_MODEL
        self.strict_mode = STRICT_MODE
        self.fx_daily_change_max_abs = FX_DAILY_CHANGE_MAX_ABS
        self.cache_dir.mkdir(parents=True, exist_ok=True)
