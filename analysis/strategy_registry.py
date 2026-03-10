"""
策略注册表。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable

from otc_fund_quant.analysis.signals import (
    generate_recommendation,
    generate_recommendation_index_momentum,
    generate_recommendation_regime,
)


@dataclass(frozen=True)
class StrategyDefinition:
    id: str
    label: str
    description: str
    generator: Callable


_STRATEGY_DEFINITIONS: "OrderedDict[str, StrategyDefinition]" = OrderedDict(
    [
        (
            "v6",
            StrategyDefinition(
                id="v6",
                label="V6 估值趋势",
                description=(
                    "基于估值百分位 + 趋势信号的混合策略。在低估值区间叠加 MA 金叉、"
                    "RSI 超卖等技术信号触发买入，高 RSI 或死叉触发卖出。"
                ),
                generator=generate_recommendation,
            ),
        ),
        (
            "regime_adaptive",
            StrategyDefinition(
                id="regime_adaptive",
                label="状态自适应",
                description=(
                    "基于 ADX 判断市场状态，在趋势市与震荡市之间切换交易逻辑，"
                    "融合估值、RSI、MACD 与 ATR 风险控制。"
                ),
                generator=generate_recommendation_regime,
            ),
        ),
        (
            "index_momentum",
            StrategyDefinition(
                id="index_momentum",
                label="指数动量",
                description=(
                    "专为指数基金设计，用布林带替代估值百分位，以多周期动量评分"
                    "和 ATR 自适应仓位管理驱动建仓。"
                ),
                generator=generate_recommendation_index_momentum,
            ),
        ),
    ]
)


def get_strategy_definition(strategy_id: str) -> StrategyDefinition:
    return _STRATEGY_DEFINITIONS[strategy_id]


def get_strategy_catalog(strategy_ids: Iterable[str] | None = None) -> list[dict[str, str]]:
    if strategy_ids is None:
        selected_ids = list(_STRATEGY_DEFINITIONS.keys())
    else:
        selected_ids = [strategy_id for strategy_id in strategy_ids if strategy_id in _STRATEGY_DEFINITIONS]

    return [
        {
            "id": definition.id,
            "label": definition.label,
            "description": definition.description,
        }
        for strategy_id, definition in _STRATEGY_DEFINITIONS.items()
        if strategy_id in selected_ids
    ]


def is_known_strategy(strategy_id: str) -> bool:
    return strategy_id in _STRATEGY_DEFINITIONS
