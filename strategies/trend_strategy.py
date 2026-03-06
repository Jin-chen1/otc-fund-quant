from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import date
from ..core.account import Account
from ..analysis.indicators import calc_indicators, count_buy_signals


# ===== 策略常量配置（V6 — 在V1基础上最小改动） =====
# 核心思路：V1已经表现良好(+46.33%)，只做微调不做大改
# 仅微调DCA金额(3%→4%)和冷静期(10-30→8-25)

MAX_POSITION_RATIO = 0.80          # 持仓上限（V1原值）
SOFT_STOP_LOSS = -0.18             # 软止损（V1原值）
HARD_STOP_LOSS = -0.25             # 硬止损（V1原值）
MIN_COOLDOWN = 8                   # 冷静期从10微调到8天
MAX_COOLDOWN = 25                  # 冷静期从30微调到25天
DCA_BASE_RATIO = 0.04              # 基础定投从3%微调到4%
DCA_INTERVAL = 7                   # 定投间隔保持7天（V1原值）
CHEAP_PERCENTILE = 0.20            # 低估区阈值（V1原值）
BATCH_INTERVAL = 3                 # 分批间隔（V1原值）
MIN_HISTORY_DAYS = 250             # 最少历史数据天数
TREND_BATCH_RATIOS = [0.20, 0.15, 0.15]  # 分3批建仓比例（V1原值）
RE_ENTRY_STRONG_RATIO = 0.85             # 冷静期后强信号折扣
RE_ENTRY_WEAK_RATIO = 0.65               # 冷静期后弱信号折扣
LOW_POSITION_RATIO = 0.30               # 低仓位保护阈值（V1原值）
RSI_SELL_THRESHOLD = 70                  # RSI卖出阈值（V1原值）


class ValuationTrendHybridStrategy:
    """
    估值趋势混合策略（V6 — V1微调版）

    在V1基础上只做最小幅度微调：
    1. DCA基础金额从3%微调到4%
    2. 冷静期从10-30天微调到8-25天
    3. 其余完全保持V1原版
    """

    def __init__(self, account: Account, fund_code: str, params: Dict[str, Any] = None):
        self.account = account
        self.fund_code = fund_code
        self.dca_counter = 0
        self.cooldown_remaining = 0
        self.soft_stop_triggered = False
        self.re_entry_ratio = 0.0
        self.pending_batches: List[Dict] = []
        self.bar_count = 0

        # 从 params 加载参数，未传则使用模块级默认常量
        p = params or {}
        self._params = p
        self.p_max_position_ratio = p.get("max_position_ratio", MAX_POSITION_RATIO)
        self.p_soft_stop_loss = p.get("soft_stop_loss", SOFT_STOP_LOSS)
        self.p_hard_stop_loss = p.get("hard_stop_loss", HARD_STOP_LOSS)
        self.p_min_cooldown = p.get("min_cooldown", MIN_COOLDOWN)
        self.p_max_cooldown = p.get("max_cooldown", MAX_COOLDOWN)
        self.p_dca_base_ratio = p.get("dca_base_ratio", DCA_BASE_RATIO)
        self.p_dca_interval = p.get("dca_interval", DCA_INTERVAL)
        self.p_cheap_percentile = p.get("cheap_percentile", CHEAP_PERCENTILE)
        self.p_batch_interval = p.get("batch_interval", BATCH_INTERVAL)
        self.p_min_history_days = p.get("min_history_days", MIN_HISTORY_DAYS)
        self.p_trend_batch_ratios = p.get("trend_batch_ratios", TREND_BATCH_RATIOS)
        self.p_re_entry_strong_ratio = p.get("re_entry_strong_ratio", RE_ENTRY_STRONG_RATIO)
        self.p_re_entry_weak_ratio = p.get("re_entry_weak_ratio", RE_ENTRY_WEAK_RATIO)
        self.p_low_position_ratio = p.get("low_position_ratio", LOW_POSITION_RATIO)
        self.p_rsi_sell_threshold = p.get("rsi_sell_threshold", RSI_SELL_THRESHOLD)
        self.p_tail_size = p.get("tail_size", 300)
        self.p_percentile_window = p.get("percentile_window", 250)

        # 基础定投金额：初始总资产 × dca_base_ratio
        self.base_dca_amount = account.cash * self.p_dca_base_ratio

    def on_bar(self, current_date: date, history_df: pd.DataFrame) -> Dict[str, Any]:
        """每个交易日调用。history_df 包含截至 T-1 的数据。"""
        if len(history_df) < self.p_min_history_days:
            return {}

        self.bar_count += 1
        indicators = calc_indicators(history_df, tail_size=self.p_tail_size,
                                        percentile_window=self.p_percentile_window)
        position = self.account.get_position(self.fund_code)
        nav = indicators["current_nav"]

        # 1. 检查止损条件（优先级最高）
        stop_signal = self._check_stop_loss(position, nav, indicators)
        if stop_signal:
            return stop_signal

        # 2. 处理待执行的分批建仓
        batch_signal = self._process_pending_batches(indicators, position, nav)
        if batch_signal:
            return batch_signal

        # 3. 更新冷静期
        self._update_cooldown(indicators)

        # 4. 分区域处理
        if indicators["is_cheap_zone"]:
            return self._handle_cheap_zone(indicators, position, nav)
        return self._handle_normal_zone(indicators, position, nav)

    def _check_stop_loss(
        self, position: Any, nav: float, indicators: Dict
    ) -> Dict[str, Any]:
        """分级止损检查（V1原版逻辑）。"""
        if position.total_shares <= 0 or position.avg_cost <= 0:
            return {}
        pnl_ratio = (nav - position.avg_cost) / position.avg_cost

        if pnl_ratio <= self.p_hard_stop_loss:
            self._enter_cooldown()
            return {"sell_shares": position.total_shares}

        if pnl_ratio <= self.p_soft_stop_loss and not self.soft_stop_triggered:
            self.soft_stop_triggered = True
            sell_shares = self._apply_low_position_protection(position, nav, 0.5)
            if sell_shares > 0:
                return {"sell_shares": sell_shares}

        return {}

    def _apply_low_position_protection(
        self, position: Any, nav: float, target_ratio: float
    ) -> float:
        """低仓位卖出保护（V1原版）。"""
        position_ratio = self._get_position_ratio(position, nav)
        sell_shares = position.total_shares * target_ratio
        if position_ratio < self.p_low_position_ratio and target_ratio > 0.5:
            sell_shares = position.total_shares * 0.5
        return sell_shares

    def _get_position_ratio(self, position: Any, nav: float) -> float:
        """计算当前持仓市值占总资产的比例。"""
        market_value = position.total_shares * nav
        total_assets = self.account.cash + market_value
        if total_assets <= 0:
            return 0.0
        return market_value / total_assets

    def _enter_cooldown(self):
        """进入冷静期。"""
        self.cooldown_remaining = self.p_max_cooldown
        self.dca_counter = 0
        self.pending_batches = []
        self.soft_stop_triggered = False

    def _update_cooldown(self, indicators: Dict):
        """更新冷静期（缩短至7-20天）。"""
        if self.cooldown_remaining <= 0:
            return
        self.cooldown_remaining -= 1
        days_elapsed = self.p_max_cooldown - self.cooldown_remaining

        if days_elapsed < self.p_min_cooldown:
            return

        # 提前退出条件A：金叉 + RSI > 40
        if indicators["gold_cross"] and indicators["rsi"] > 40:
            self.cooldown_remaining = 0
            self.re_entry_ratio = 1.0
            return

        # 提前退出条件B：价格连续3天站上 MA20
        if indicators.get("above_ma20_3d", False):
            self.cooldown_remaining = 0
            self.re_entry_ratio = 1.0

    def _handle_cheap_zone(
        self, indicators: Dict, position: Any, nav: float
    ) -> Dict[str, Any]:
        """低估区逻辑：加大定投 + 更强加速 + 冷静期内允许半价DCA。"""
        self.dca_counter += 1
        if self.dca_counter < self.p_dca_interval:
            return {}

        if self.cooldown_remaining > 0:
            return {}

        self.dca_counter = 0
        percentile = indicators["percentile"]

        # 估值加速：越低估买越多（V1原版加速因子）
        if percentile < 0.10:
            multiplier = 2.0
        elif percentile < 0.15:
            multiplier = 1.5
        else:
            multiplier = 1.0

        raw_amount = self.base_dca_amount * multiplier
        buy_amount = self._apply_position_limit_with_params(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}

        return {"buy_amount": buy_amount}

    def _handle_normal_zone(
        self, indicators: Dict, position: Any, nav: float
    ) -> Dict[str, Any]:
        """正常区逻辑：三重过滤买入（V1原版）+ V1原版卖出信号。"""
        self.dca_counter = 0

        # 买入信号：V1三重过滤（金叉/RSI/MACD 满足2/3）
        signal_count = count_buy_signals(indicators, params=self._params)
        if self.cooldown_remaining <= 0 and signal_count >= 2:
            self._create_batch_plan(nav)
            return self._execute_first_batch(position, nav, signal_count)

        # 卖出信号：V1原版（死叉+RSI>70 或 死叉+MACD5天负）
        if _check_sell_signal_v6(indicators, self.p_rsi_sell_threshold) and position.total_shares > 0:
            self.pending_batches = []
            sell_shares = self._apply_low_position_protection(position, nav, 1.0)
            if sell_shares > 0:
                return {"sell_shares": sell_shares}

        return {}

    def _create_batch_plan(self, nav: float):
        """创建分3批建仓计划（V1原版比例）。"""
        self.pending_batches = []
        for i, ratio in enumerate(self.p_trend_batch_ratios):
            self.pending_batches.append({
                "trigger_bar": self.bar_count + i * self.p_batch_interval,
                "cash_ratio": ratio,
            })
        self.pending_batches.pop(0)

    def _execute_first_batch(
        self, position: Any, nav: float, signal_count: int = 3
    ) -> Dict[str, Any]:
        """执行第1批建仓。"""
        ratio = self.p_trend_batch_ratios[0]
        if self.re_entry_ratio > 0:
            discount = self.p_re_entry_strong_ratio if signal_count >= 3 else self.p_re_entry_weak_ratio
            ratio *= discount
            self.re_entry_ratio = 0.0
        raw_amount = self.account.cash * ratio
        buy_amount = self._apply_position_limit_with_params(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}

        return {"buy_amount": buy_amount}

    def _apply_position_limit_with_params(
        self, raw_amount: float, position: Any, nav: float
    ) -> float:
        """应用仓位上限保护（使用实例参数）。"""
        current_market_value = position.total_shares * nav
        total_assets = self.account.cash + current_market_value
        max_market_value = total_assets * self.p_max_position_ratio
        room = max_market_value - current_market_value
        capped_amount = min(raw_amount, room, self.account.cash)
        return max(capped_amount, 0.0)

    def _process_pending_batches(
        self, indicators: Dict, position: Any, nav: float
    ) -> Dict[str, Any]:
        """处理待执行的分批建仓（第2、3批，V1原版）。"""
        if not self.pending_batches:
            return {}

        next_batch = self.pending_batches[0]
        if self.bar_count < next_batch["trigger_bar"]:
            return {}

        if not indicators.get("above_ma20", True):
            self.pending_batches = []
            return {}

        self.pending_batches.pop(0)
        ratio = next_batch["cash_ratio"]
        raw_amount = self.account.cash * ratio
        buy_amount = self._apply_position_limit_with_params(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}
        return {"buy_amount": buy_amount}


# ===== 策略专用函数 =====

def _check_sell_signal_v6(indicators: Dict, rsi_threshold: float = RSI_SELL_THRESHOLD) -> bool:
    """V6原版卖出信号：死叉+RSI>70 或 死叉+MACD连续5天负。"""
    if not indicators["death_cross"]:
        return False
    if indicators["rsi"] > rsi_threshold:
        return True
    if indicators["macd_5d_negative"]:
        return True
    return False


def _apply_position_limit(
    raw_amount: float,
    account: Account,
    position: Any,
    current_nav: float,
) -> float:
    """应用仓位上限保护（V1原版）。"""
    current_market_value = position.total_shares * current_nav
    total_assets = account.cash + current_market_value
    max_market_value = total_assets * MAX_POSITION_RATIO
    room = max_market_value - current_market_value
    capped_amount = min(raw_amount, room, account.cash)
    return max(capped_amount, 0.0)
