"""置信度评分器。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


POSITION_SOURCE_SCORE_MAP: dict[str, float] = {
    "fund_disclosure": 1.00,
    "fund_disclosure_latest_report": 1.00,
    "input_parameter": 0.90,
    "regression_beta_calibrated": 0.88,
    "regression_three_factor": 0.84,
    "adjusted_by_top10_weight": 0.75,
    "default_assumption_etf": 0.68,
    "default_assumption_regular": 0.60,
    "default_assumption": 0.60,
    "default_assumption_fallback": 0.45,
    "not_applicable": 0.70,
}

POSITION_SOURCE_TEXT_MAP: dict[str, str] = {
    "fund_disclosure": "基金披露仓位",
    "fund_disclosure_latest_report": "基金最新报告期披露仓位",
    "input_parameter": "外部输入仓位",
    "regression_beta_calibrated": "指数增强历史回归校准仓位（beta）",
    "regression_three_factor": "固收+历史三因子回归拟合仓位",
    "adjusted_by_top10_weight": "由前十大权重反推调整仓位",
    "default_assumption_etf": "ETF/联接默认仓位假设",
    "default_assumption_regular": "普通指数默认仓位假设",
    "default_assumption": "默认仓位假设",
    "default_assumption_fallback": "异常后回退默认仓位",
    "not_applicable": "不适用",
}


def _safe_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"字段 {field_name} 无法转换为浮点数: {value}") from exc


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_iso_date(date_text: Any, field_name: str) -> date:
    if not isinstance(date_text, str):
        raise TypeError(f"字段 {field_name} 必须是 YYYY-MM-DD 字符串，当前类型: {type(date_text)}")
    raw_text = date_text.strip()
    if raw_text == "":
        raise ValueError(f"字段 {field_name} 不能为空")
    try:
        return datetime.strptime(raw_text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"字段 {field_name} 日期格式非法: {date_text}，应为 YYYY-MM-DD") from exc


def _normalize_warnings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        warnings: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                warnings.append(text)
        return warnings
    text = str(value).strip()
    return [text] if text else []


def _get_component_weight_map(fund_type: str) -> dict[str, float]:
    base_weight_map: dict[str, float] = {
        "position_reliability": 0.30,
        "market_data_date_match": 0.25,
    }
    if fund_type == "active_a":
        base_weight_map["top10_quote_coverage"] = 0.45
    return base_weight_map


def _score_top10_quote_coverage(result: dict[str, Any]) -> dict[str, Any]:
    top10_weight = _safe_float(result.get("top10_weight"), "top10_weight")
    top10_weight_known = _safe_float(result.get("top10_weight_known"), "top10_weight_known")

    if top10_weight < 0:
        raise ValueError(f"top10_weight 不能为负数，当前值: {top10_weight}")
    if top10_weight_known < 0:
        raise ValueError(f"top10_weight_known 不能为负数，当前值: {top10_weight_known}")
    if top10_weight_known - top10_weight > 1e-6:
        raise ValueError(
            f"top10_weight_known 不能大于 top10_weight，当前值: {top10_weight_known} > {top10_weight}"
        )

    coverage = 0.0 if top10_weight <= 1e-12 else _clamp_01(top10_weight_known / top10_weight)
    return {"score": round(coverage, 4), "metric": f"{top10_weight_known:.2f}%/{top10_weight:.2f}%"}


def _score_position_reliability(result: dict[str, Any]) -> dict[str, Any]:
    position_source = result.get("position_source", "not_applicable")
    if position_source not in POSITION_SOURCE_SCORE_MAP:
        raise ValueError(f"未知position_source: {position_source}，可选值: {list(POSITION_SOURCE_SCORE_MAP.keys())}")
    return {
        "score": POSITION_SOURCE_SCORE_MAP[position_source],
        "metric": POSITION_SOURCE_TEXT_MAP[position_source],
        "position_source": position_source,
    }


def _score_market_data_date_match(result: dict[str, Any]) -> dict[str, Any]:
    target_dt = _parse_iso_date(result.get("target_date"), "target_date")
    date_fields = [
        ("stock_data_as_of_date", "股票行情"),
        ("index_data_as_of_date", "指数行情"),
        ("bond_data_as_of_date", "债券行情"),
        ("fx_data_as_of_date", "汇率行情"),
    ]

    scored_items: list[dict[str, Any]] = []
    for field_name, field_label in date_fields:
        if field_name not in result or result[field_name] is None:
            continue
        as_of_dt = _parse_iso_date(result[field_name], field_name)
        lag_days = (target_dt - as_of_dt).days
        if lag_days == 0:
            field_score = 1.0
        elif lag_days == 1:
            field_score = 0.7
        elif lag_days == 2:
            field_score = 0.5
        elif lag_days > 2:
            field_score = 0.3
        else:
            field_score = 0.6
        scored_items.append(
            {
                "label": field_label,
                "as_of_date": as_of_dt.isoformat(),
                "score": field_score,
                "lag_days": lag_days,
            }
        )

    if not scored_items:
        raise ValueError("缺少用于评分的行情日期字段，至少应提供一项 *_as_of_date")

    avg_score = sum(item["score"] for item in scored_items) / len(scored_items)
    metric_text = "，".join(f"{item['label']}={item['as_of_date']}" for item in scored_items)
    return {"score": round(avg_score, 4), "metric": metric_text, "items": scored_items}


def build_confidence_payload(result: dict[str, Any], fund_type: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError(f"result 必须为字典，当前类型: {type(result)}")
    if not isinstance(fund_type, str) or fund_type.strip() == "":
        raise ValueError("fund_type 不能为空")

    weight_map = _get_component_weight_map(fund_type)
    components: dict[str, dict[str, Any]] = {}

    if "top10_quote_coverage" in weight_map:
        top10_component = _score_top10_quote_coverage(result)
        top10_component["weight"] = weight_map["top10_quote_coverage"]
        components["top10_quote_coverage"] = top10_component

    position_component = _score_position_reliability(result)
    position_component["weight"] = weight_map["position_reliability"]
    components["position_reliability"] = position_component

    data_date_component = _score_market_data_date_match(result)
    data_date_component["weight"] = weight_map["market_data_date_match"]
    components["market_data_date_match"] = data_date_component

    total_weight = sum(component["weight"] for component in components.values())
    if total_weight <= 0:
        raise ValueError("评分权重之和必须大于0")

    weighted_sum = sum(component["score"] * component["weight"] for component in components.values())
    confidence_score = round(_clamp_01(weighted_sum / total_weight), 4)

    explainable_error_sources: list[str] = []

    if "top10_quote_coverage" in components and components["top10_quote_coverage"]["score"] < 1.0:
        explainable_error_sources.append("前十大持仓行情覆盖率不足100%，未覆盖部分使用代理估算，可能带来误差")

    position_source = position_component["position_source"]
    if position_source in {
        "default_assumption",
        "default_assumption_etf",
        "default_assumption_regular",
        "default_assumption_fallback",
    }:
        explainable_error_sources.append("仓位来自默认值而非最新披露，估算不确定性上升")
    elif position_source == "regression_beta_calibrated":
        explainable_error_sources.append("指数增强仓位由历史回归beta校准，受样本窗口与市场状态变化影响")
    elif position_source == "regression_three_factor":
        explainable_error_sources.append("固收+仓位由历史三因子回归拟合，受样本窗口与市场风格切换影响")
    elif position_source == "adjusted_by_top10_weight":
        explainable_error_sources.append("披露仓位与前十大权重存在冲突，系统已按前十大权重调整")

    if data_date_component["score"] < 1.0:
        explainable_error_sources.append("指数/汇率/行情数据日期与估算目标日不完全一致，可能影响当日精度")

    for warning in _normalize_warnings(result.get("warnings")):
        explainable_error_sources.append(f"告警提示：{warning}")

    deduplicated_sources: list[str] = []
    for source in explainable_error_sources:
        if source not in deduplicated_sources:
            deduplicated_sources.append(source)

    return {
        "confidence_score": confidence_score,
        "confidence_components": components,
        "explainable_error_sources": deduplicated_sources,
    }
