"""基础估算器。"""

from datetime import date, datetime


class BaseEstimator:
    """所有估算器的基类。"""

    @staticmethod
    def normalize_date(date_value: str | date | datetime, field_name: str) -> date:
        if isinstance(date_value, datetime):
            return date_value.date()
        if isinstance(date_value, date):
            return date_value
        if isinstance(date_value, str):
            raw_value = date_value.strip()
            if raw_value == "":
                raise ValueError(f"{field_name} 不能为空字符串")
            try:
                return datetime.strptime(raw_value, "%Y-%m-%d").date()
            except ValueError as exc:
                raise ValueError(f"{field_name} 日期格式非法: {date_value}，应为 YYYY-MM-DD") from exc
        raise TypeError(f"{field_name} 类型不支持: {type(date_value)}")

    @classmethod
    def resolve_fee_days(cls, nav_date: str | date | datetime, target_date: str | date | datetime | None = None) -> tuple[str, str, int]:
        nav_dt = cls.normalize_date(nav_date, "nav_date")
        target_dt = date.today() if target_date is None else cls.normalize_date(target_date, "target_date")
        fee_days = (target_dt - nav_dt).days
        if fee_days < 0:
            raise ValueError(f"target_date 早于 nav_date，不合法: nav_date={nav_dt.isoformat()}, target_date={target_dt.isoformat()}")
        return nav_dt.isoformat(), target_dt.isoformat(), fee_days

    @classmethod
    def day_count(cls, nav_date: str | date | datetime, target_date: str | date | datetime | None = None) -> int:
        _, _, fee_days = cls.resolve_fee_days(nav_date, target_date)
        return fee_days

    @staticmethod
    def validate_percentage(value, field_name: str) -> float:
        if value is None:
            raise ValueError(f"{field_name} 不能为空")

        parsed_value = value
        if isinstance(value, str):
            raw_value = value.strip()
            if raw_value == "":
                raise ValueError(f"{field_name} 不能为空字符串")
            if raw_value.endswith("%"):
                raw_value = raw_value[:-1].strip()
            try:
                parsed_value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"{field_name} 无法解析为百分比数值: {value}") from exc

        if not isinstance(parsed_value, (int, float)):
            raise TypeError(f"{field_name} 必须是数值类型，当前为: {type(parsed_value)}")

        value_float = float(parsed_value)
        if 0 < value_float < 1:
            raise ValueError(f"{field_name} 取值为 {value_float}，疑似传入比例值；请传入百分比（例如 93 表示 93%）")
        if value_float < 0 or value_float > 100:
            raise ValueError(f"{field_name} 超出范围[0,100]，当前值: {value_float}")
        return value_float

    @classmethod
    def daily_fee_drag(
        cls,
        mgmt_rate: float,
        custody_rate: float,
        nav_date: str | date | datetime,
        target_date: str | date | datetime | None = None,
        days_in_year: int = 365,
    ) -> float:
        if days_in_year <= 0:
            raise ValueError(f"days_in_year 必须为正整数，当前值: {days_in_year}")
        fee_days = cls.day_count(nav_date, target_date)
        return (mgmt_rate + custody_rate) / days_in_year * fee_days

    @staticmethod
    def estimate_nav(last_nav: float, daily_return: float) -> float:
        return last_nav * (1 + daily_return)

    def estimate(self, *args, **kwargs) -> float:
        raise NotImplementedError("子类必须实现estimate方法")
