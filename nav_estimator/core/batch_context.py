"""批量估算共享上下文。"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class BatchContext:
    """批量估算过程中可复用的公共数据快照。"""

    data_as_of_date: str = field(default_factory=lambda: date.today().isoformat())
    is_historical: bool = False
    requested_as_of_date: str | None = None

    a_prices: dict[str, dict[str, float]] = field(default_factory=dict)
    hk_prices: dict[str, dict[str, float]] = field(default_factory=dict)
    all_prices: dict[str, dict[str, float]] = field(default_factory=dict)
    required_a_codes: set[str] = field(default_factory=set)
    required_hk_codes: set[str] = field(default_factory=set)
    a_price_as_of_dates: dict[str, str] = field(default_factory=dict)
    hk_price_as_of_dates: dict[str, str] = field(default_factory=dict)

    a_index_returns: dict[str, float] = field(default_factory=dict)
    hk_index_returns: dict[str, float] = field(default_factory=dict)
    bond_index_returns: dict[str, float] = field(default_factory=dict)
    bond_index_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    a_index_as_of_dates: dict[str, str] = field(default_factory=dict)
    hk_index_as_of_dates: dict[str, str] = field(default_factory=dict)
    bond_index_as_of_dates: dict[str, str] = field(default_factory=dict)
    hkd_cny_daily_change: float | None = None
    fx_data_as_of_date: str | None = None

    a_index_errors: dict[str, str] = field(default_factory=dict)
    hk_index_errors: dict[str, str] = field(default_factory=dict)
    bond_index_errors: dict[str, str] = field(default_factory=dict)
    a_prices_error: str | None = None
    hk_prices_error: str | None = None
    fx_error: str | None = None
