"""
市场状态自适应策略 V2（Regime-Adaptive Strategy）

核心思路：以V6验证过的逻辑为基础，叠加ADX市场状态转换信号。
- 低估区DCA、三重过滤买入、分级止损 → 与V6完全一致
- 新增：BEAR→BULL转换时额外加仓，BULL→BEAR转换时提前减仓
- 新增：牛市DCA加速1.5倍

技术指标：ADX(14)、ATR(14)、MA20/MA60、RSI(14)、MACD
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import date
from ..core.account import Account
from ..analysis.indicators import calc_indicators, count_buy_signals, detect_regime


# ===== 市场状态 =====
REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_RANGE = "RANGE"
ADX_TREND_THRESHOLD = 25

# ===== V6 原版参数（已验证 +50.57%） =====
MAX_POSITION_RATIO = 0.80
SOFT_STOP_LOSS = -0.18
HARD_STOP_LOSS = -0.25
MIN_COOLDOWN = 8
MAX_COOLDOWN = 25
DCA_BASE_RATIO = 0.04
DCA_INTERVAL = 7
CHEAP_PERCENTILE = 0.20
BATCH_INTERVAL = 3
MIN_HISTORY_DAYS = 250
TREND_BATCH_RATIOS = [0.20, 0.15, 0.15]
RE_ENTRY_STRONG_RATIO = 0.85
RE_ENTRY_WEAK_RATIO = 0.65
LOW_POSITION_RATIO = 0.30
RSI_SELL_THRESHOLD = 70

# ===== 新增：regime 叠加参数 =====
BULL_DCA_BOOST = 1.5        # 牛市定投加速倍数
TRANSITION_BUY_RATIO = 0.15 # BEAR→BULL转换加仓比例
TRANSITION_SELL_RATIO = 0.3 # BULL→BEAR转换减仓比例


class RegimeAdaptiveStrategy:
    """
    V6 + 市场状态转换信号叠加策略

    保留V6全部逻辑，额外新增：
    1. BEAR→BULL 转换时触发额外加仓
    2. BULL→BEAR 转换时触发提前减仓
    3. 牛市期间DCA定投金额×1.5
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
        # regime 专属参数
        self.p_adx_trend_threshold = p.get("adx_trend_threshold", ADX_TREND_THRESHOLD)
        self.p_bull_dca_boost = p.get("bull_dca_boost", BULL_DCA_BOOST)
        self.p_transition_buy_ratio = p.get("transition_buy_ratio", TRANSITION_BUY_RATIO)
        self.p_transition_sell_ratio = p.get("transition_sell_ratio", TRANSITION_SELL_RATIO)
        self.p_tail_size = p.get("tail_size", 300)
        self.p_percentile_window = p.get("percentile_window", 250)

        self.base_dca_amount = account.cash * self.p_dca_base_ratio

        # 新增：regime 状态追踪
        self.current_regime = REGIME_RANGE
        self.prev_regime = REGIME_RANGE

    def on_bar(self, current_date: date, history_df: pd.DataFrame) -> Dict[str, Any]:
        """每个交易日调用（V6逻辑 + regime叠加）。"""
        if len(history_df) < self.p_min_history_days:
            return {}

        self.bar_count += 1
        indicators = calc_indicators(history_df, tail_size=self.p_tail_size,
                                        percentile_window=self.p_percentile_window)
        position = self.account.get_position(self.fund_code)
        nav = indicators["current_nav"]

        # 1. 更新市场状态
        self.prev_regime = self.current_regime
        self.current_regime = detect_regime(indicators)

        # 2. 检查止损（V6原版）
        stop_signal = self._check_stop_loss(position, nav, indicators)
        if stop_signal:
            return stop_signal

        # 3. 处理分批建仓（V6原版）
        batch_signal = self._process_pending_batches(indicators, position, nav)
        if batch_signal:
            return batch_signal

        # 4. 更新冷静期（V6原版）
        self._update_cooldown(indicators)

        # ★ 新增：检测状态转换信号（不受冷静期限制）
        transition_signal = self._check_regime_transition(position, nav)
        if transition_signal:
            return transition_signal

        if self.cooldown_remaining > 0:
            return {}

        # 5. 分区域处理（V6原版，牛市DCA加速）
        if indicators["is_cheap_zone"]:
            return self._handle_cheap_zone(indicators, position, nav)
        return self._handle_normal_zone(indicators, position, nav)

    # ===== ★ 新增：状态转换信号 =====

    def _check_regime_transition(self, position: Any, nav: float) -> Dict[str, Any]:
        """检测市场状态转换，触发额外买入或卖出。"""
        if self.prev_regime == self.current_regime:
            return {}

        # BEAR/RANGE → BULL：趋势确认，额外加仓
        if (self.current_regime == REGIME_BULL and
                self.prev_regime in (REGIME_BEAR, REGIME_RANGE)):
            if self.cooldown_remaining > 0:
                self.cooldown_remaining = 0  # 趋势转牛，提前结束冷静期
            raw_amount = self.account.cash * self.p_transition_buy_ratio
            buy_amount = self._apply_position_limit_with_params(raw_amount, position, nav)
            if buy_amount >= 1.0:
                return {"buy_amount": buy_amount}

        # BULL → BEAR：趋势反转，提前减仓
        if (self.current_regime == REGIME_BEAR and
                self.prev_regime == REGIME_BULL and
                position.total_shares > 0):
            sell_shares = position.total_shares * self.p_transition_sell_ratio
            if sell_shares > 0:
                self.pending_batches = []
                return {"sell_shares": sell_shares}

        return {}

    # ===== 以下全部为V6原版逻辑 =====

    def _check_stop_loss(self, position: Any, nav: float, indicators: Dict) -> Dict[str, Any]:
        """分级止损检查（V6原版逻辑）。"""
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

    def _handle_cheap_zone(self, indicators: Dict, position: Any,
                           nav: float) -> Dict[str, Any]:
        """低估区逻辑（V6原版 + 牛市DCA加速）。"""
        self.dca_counter += 1
        if self.dca_counter < self.p_dca_interval:
            return {}

        if self.cooldown_remaining > 0:
            return {}

        self.dca_counter = 0
        percentile = indicators["percentile"]

        # 估值加速（V6原版）
        if percentile < 0.10:
            multiplier = 2.0
        elif percentile < 0.15:
            multiplier = 1.5
        else:
            multiplier = 1.0

        # ★ 新增：牛市DCA加速
        if self.current_regime == REGIME_BULL:
            multiplier *= self.p_bull_dca_boost

        raw_amount = self.base_dca_amount * multiplier
        buy_amount = self._apply_position_limit_with_params(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}
        return {"buy_amount": buy_amount}

    def _handle_normal_zone(self, indicators: Dict, position: Any,
                            nav: float) -> Dict[str, Any]:
        """正常区逻辑（V6原版）。"""
        self.dca_counter = 0

        # 买入信号：V6三重过滤
        signal_count = count_buy_signals(indicators, params=self._params)
        if self.cooldown_remaining <= 0 and signal_count >= 2:
            self._create_batch_plan(nav)
            return self._execute_first_batch(position, nav, signal_count)

        # 卖出信号：V6原版
        if _check_sell_signal_v6(indicators, self.p_rsi_sell_threshold) and position.total_shares > 0:
            self.pending_batches = []
            sell_shares = self._apply_low_position_protection(position, nav, 1.0)
            if sell_shares > 0:
                return {"sell_shares": sell_shares}

        return {}

    # ===== V6 辅助函数 =====

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

    def _apply_low_position_protection(self, position: Any, nav: float,
                                        target_ratio: float) -> float:
        position_ratio = self._get_position_ratio(position, nav)
        sell_shares = position.total_shares * target_ratio
        if position_ratio < self.p_low_position_ratio and target_ratio > 0.5:
            sell_shares = position.total_shares * 0.5
        return sell_shares

    def _get_position_ratio(self, position: Any, nav: float) -> float:
        market_value = position.total_shares * nav
        total_assets = self.account.cash + market_value
        if total_assets <= 0:
            return 0.0
        return market_value / total_assets

    def _enter_cooldown(self):
        self.cooldown_remaining = self.p_max_cooldown
        self.dca_counter = 0
        self.pending_batches = []
        self.soft_stop_triggered = False

    def _update_cooldown(self, indicators: Dict):
        if self.cooldown_remaining <= 0:
            return
        self.cooldown_remaining -= 1
        days_elapsed = self.p_max_cooldown - self.cooldown_remaining

        if days_elapsed < self.p_min_cooldown:
            return

        if indicators["gold_cross"] and indicators["rsi"] > 40:
            self.cooldown_remaining = 0
            self.re_entry_ratio = 1.0
            return

        if indicators.get("above_ma20_3d", False):
            self.cooldown_remaining = 0
            self.re_entry_ratio = 1.0

    def _create_batch_plan(self, nav: float):
        self.pending_batches = []
        for i, ratio in enumerate(self.p_trend_batch_ratios):
            self.pending_batches.append({
                "trigger_bar": self.bar_count + i * self.p_batch_interval,
                "cash_ratio": ratio,
            })
        self.pending_batches.pop(0)

    def _execute_first_batch(self, position: Any, nav: float,
                              signal_count: int = 3) -> Dict[str, Any]:
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

    def _process_pending_batches(self, indicators: Dict, position: Any,
                                  nav: float) -> Dict[str, Any]:
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
    """V6原版卖出信号：死叉+RSI>70 或 死叉+MACD连续5天为负。"""
    if not indicators["death_cross"]:
        return False
    if indicators["rsi"] > rsi_threshold:
        return True
    if indicators["macd_5d_negative"]:
        return True
    return False


def _apply_position_limit(raw_amount: float, account: Account,
                           position: Any, current_nav: float,
                           max_ratio: float) -> float:
    """应用仓位上限保护。"""
    current_market_value = position.total_shares * current_nav
    total_assets = account.cash + current_market_value
    max_market_value = total_assets * max_ratio
    room = max_market_value - current_market_value
    capped_amount = min(raw_amount, room, account.cash)
    return max(capped_amount, 0.0)
