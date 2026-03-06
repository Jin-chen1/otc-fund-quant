"""
指数动量策略（IndexMomentumStrategy）— 专为指数ETF联接基金设计。

核心思路：用布林带替代百分位估值，用多维动量评分替代三重过滤，
用ATR自适应仓位管理替代固定DCA比例。

技术指标：布林带(20,2)、多周期动量(5d/10d/20d)、RSI(14)、MACD、ATR(14)
区域划分：超卖区(价格<下轨) / 正常区 / 超买区(价格>上轨)
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import date
from ..core.account import Account
from ..analysis.indicators import calc_indicators


# ===== 默认参数 =====
MAX_POSITION_RATIO = 0.85
SOFT_STOP_LOSS = -0.20
HARD_STOP_LOSS = -0.30
MIN_COOLDOWN = 5
MAX_COOLDOWN = 15
DCA_BASE_RATIO = 0.08
DCA_OVERSOLD_INTERVAL = 3
DCA_NORMAL_INTERVAL = 5
MIN_HISTORY_DAYS = 60
MOMENTUM_BUY_THRESHOLD = 2
RSI_OVERBOUGHT = 75
ATR_SCALE_MAX = 2.0
LOW_POSITION_RATIO = 0.30
BATCH_INTERVAL = 3
TREND_BATCH_RATIOS = [0.25, 0.20, 0.15]


class IndexMomentumStrategy:
    """指数动量策略 — 布林带区域划分 + 多维动量评分 + ATR自适应仓位"""

    def __init__(self, account: Account, fund_code: str, params: Dict[str, Any] = None):
        self.account = account
        self.fund_code = fund_code
        self.dca_counter = 0
        self.cooldown_remaining = 0
        self.soft_stop_triggered = False
        self.pending_batches: List[Dict] = []
        self.bar_count = 0
        self.prev_bb_position = 0.5
        self.sell_cooldown = 0

        p = params or {}
        self._params = p
        self.p_max_position_ratio = p.get("max_position_ratio", MAX_POSITION_RATIO)
        self.p_soft_stop_loss = p.get("soft_stop_loss", SOFT_STOP_LOSS)
        self.p_hard_stop_loss = p.get("hard_stop_loss", HARD_STOP_LOSS)
        self.p_min_cooldown = p.get("min_cooldown", MIN_COOLDOWN)
        self.p_max_cooldown = p.get("max_cooldown", MAX_COOLDOWN)
        self.p_dca_base_ratio = p.get("dca_base_ratio", DCA_BASE_RATIO)
        self.p_dca_oversold_interval = p.get("dca_oversold_interval", DCA_OVERSOLD_INTERVAL)
        self.p_dca_normal_interval = p.get("dca_normal_interval", DCA_NORMAL_INTERVAL)
        self.p_min_history_days = p.get("min_history_days", MIN_HISTORY_DAYS)
        self.p_momentum_buy_threshold = p.get("momentum_buy_threshold", MOMENTUM_BUY_THRESHOLD)
        self.p_rsi_overbought = p.get("rsi_overbought", RSI_OVERBOUGHT)
        self.p_atr_scale_max = p.get("atr_scale_max", ATR_SCALE_MAX)
        self.p_low_position_ratio = p.get("low_position_ratio", LOW_POSITION_RATIO)
        self.p_batch_interval = p.get("batch_interval", BATCH_INTERVAL)
        self.p_trend_batch_ratios = p.get("trend_batch_ratios", TREND_BATCH_RATIOS)
        self.p_tail_size = p.get("tail_size", 200)
        self.p_percentile_window = p.get("percentile_window", 150)

        self._initial_cash = account.cash

    def on_bar(self, current_date: date, history_df: pd.DataFrame) -> Dict[str, Any]:
        """每个交易日调用。"""
        if len(history_df) < self.p_min_history_days:
            return {}

        self.bar_count += 1
        indicators = calc_indicators(history_df, tail_size=self.p_tail_size,
                                     percentile_window=self.p_percentile_window)
        position = self.account.get_position(self.fund_code)
        nav = indicators["current_nav"]
        bb_pos = indicators.get("bb_position", 0.5)

        # 卖出冷却期递减
        if self.sell_cooldown > 0:
            self.sell_cooldown -= 1

        # 1. 止损检查
        stop = self._check_stop_loss(position, nav)
        if stop:
            self.prev_bb_position = bb_pos
            return stop

        # 2. 分批建仓
        batch = self._process_pending_batches(indicators, position, nav)
        if batch:
            self.prev_bb_position = bb_pos
            return batch

        # 3. 冷静期
        self._update_cooldown(indicators)
        if self.cooldown_remaining > 0:
            self.prev_bb_position = bb_pos
            return {}

        # 4. 布林带区域分派
        if bb_pos < 0.0:
            result = self._handle_oversold_zone(indicators, position, nav)
        elif bb_pos > 1.0:
            result = self._handle_overbought_zone(indicators, position, nav)
        else:
            result = self._handle_normal_zone(indicators, position, nav)

        self.prev_bb_position = bb_pos
        return result

    # ===== 区域处理逻辑 =====

    def _handle_oversold_zone(self, indicators: Dict, position: Any,
                              nav: float) -> Dict[str, Any]:
        """超卖区：每日加速定投，金额随偏离下轨距离递增。"""
        bb_pos = indicators.get("bb_position", 0.0)

        # 越跌越买：%B越负，加速因子越大
        if bb_pos < -0.5:
            multiplier = 2.5
        elif bb_pos < -0.2:
            multiplier = 2.0
        else:
            multiplier = 1.5

        atr_scale = self._calc_atr_scale(indicators)
        dca_amount = self._calc_dca_amount(position, nav)
        raw_amount = dca_amount * multiplier * atr_scale
        buy_amount = self._apply_position_limit(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}
        return {"buy_amount": buy_amount}

    def _handle_overbought_zone(self, indicators: Dict, position: Any,
                                nav: float) -> Dict[str, Any]:
        """超买区：检查卖出条件，同时继续减量DCA避免现金闲置。"""
        # 超买卖出：价格突破上轨 + RSI确认超买（受卖出冷却期保护）
        if position.total_shares > 0:
            rsi = indicators.get("rsi", 50)
            if rsi > self.p_rsi_overbought and self.sell_cooldown <= 0:
                sell_shares = self._apply_low_position_protection(position, nav, 0.08)
                if sell_shares > 0:
                    self.pending_batches = []
                    self.sell_cooldown = 25
                    return {"sell_shares": sell_shares}

        # 超买区减量DCA：强趋势中价格可持续在上轨之上，不应完全停止买入
        self.dca_counter += 1
        if self.dca_counter >= self.p_dca_normal_interval * 2:
            self.dca_counter = 0
            dca_amount = self._calc_dca_amount(position, nav) * 0.5
            buy_amount = self._apply_position_limit(dca_amount, position, nav)
            if buy_amount >= 1.0:
                return {"buy_amount": buy_amount}

        return {}

    def _handle_normal_zone(self, indicators: Dict, position: Any,
                            nav: float) -> Dict[str, Any]:
        """正常区：动量评分驱动批量建仓 + 定期DCA兜底 + 弱势确认卖出。"""
        # 动量评分买入（触发批量建仓）
        mom_score = self._calc_momentum_score(indicators)
        if mom_score >= self.p_momentum_buy_threshold:
            self.dca_counter = 0
            self._create_batch_plan()
            return self._execute_first_batch(position, nav, mom_score)

        # 弱势卖出：5日动量转负 + 跌破MA20 + MACD连续5天负
        # 盈利保护：PnL>0时禁止弱势卖出（上涨趋势中短暂回调不应卖出）
        pnl_ratio = 0.0
        if position.total_shares > 0 and position.avg_cost > 0:
            pnl_ratio = (nav - position.avg_cost) / position.avg_cost
        if (pnl_ratio <= 0 and
                indicators.get("mom_5d", 0) < 0 and
                not indicators.get("above_ma20", True) and
                position.total_shares > 0 and
                self.sell_cooldown <= 0):
            if indicators.get("macd_5d_negative", False):
                sell_shares = self._apply_low_position_protection(position, nav, 0.10)
                if sell_shares > 0:
                    self.pending_batches = []
                    self.sell_cooldown = 25
                    return {"sell_shares": sell_shares}

        # 正常区定期DCA兜底：动量不达标时也持续买入
        # 低仓位加速建仓：仓位<70%时每日买入，快速建立基础仓位
        pos_ratio = self._get_position_ratio(position, nav)
        effective_interval = 1 if pos_ratio < 0.7 else self.p_dca_normal_interval
        self.dca_counter += 1
        if self.dca_counter >= effective_interval:
            self.dca_counter = 0
            atr_scale = self._calc_atr_scale(indicators)
            dca_amount = self._calc_dca_amount(position, nav)
            # 仓位缺口加速补仓：仓位距目标越远买越多
            target = self.p_max_position_ratio
            gap = max(target - pos_ratio, 0)
            gap_boost = 1.0 + gap  # 仓位0%→boost=1.95, 仓位70%→boost=1.25, 仓位95%→boost=1.0
            dca_amount *= gap_boost
            # 趋势加成：站上MA20时DCA金额×1.3
            if indicators.get("above_ma20", False):
                dca_amount *= 1.3
            raw_amount = dca_amount * atr_scale
            buy_amount = self._apply_position_limit(raw_amount, position, nav)
            if buy_amount >= 1.0:
                return {"buy_amount": buy_amount}

        return {}

    # ===== 动量评分 =====

    def _calc_momentum_score(self, indicators: Dict) -> int:
        """6维动量评分（满分6）。"""
        score = 0
        # 1. 5日短期动量为正
        if indicators.get("mom_5d", 0) > 0:
            score += 1
        # 2. 10日中期动量为正
        if indicators.get("mom_10d", 0) > 0:
            score += 1
        # 3. 20日中长期动量为正
        if indicators.get("mom_20d", 0) > 0:
            score += 1
        # 4. 价格站上MA20
        if indicators.get("above_ma20", False):
            score += 1
        # 5. MACD柱由负转正
        if indicators.get("macd_turn_positive", False):
            score += 1
        # 6. 布林带下轨反弹（近3日触及下轨 + 当前%B>0.3）
        bb_pos = indicators.get("bb_position", 0.5)
        touched = indicators.get("bb_touched_lower_3d", False)
        if touched and bb_pos > 0.3:
            score += 1
        return score

    # ===== 动态DCA金额 =====

    def _calc_dca_amount(self, position: Any, nav: float) -> float:
        """基于总资产动态计算DCA金额，随组合增长而扩大。"""
        total_assets = self.account.cash + position.total_shares * nav
        return total_assets * self.p_dca_base_ratio

    # ===== ATR自适应仓位 =====

    def _calc_atr_scale(self, indicators: Dict) -> float:
        """ATR反波动率缩放：波动低→买多，波动高→买少。"""
        atr = indicators.get("atr", 0)
        atr_median = indicators.get("atr_median", atr)
        if atr <= 0 or atr_median <= 0:
            return 1.0
        scale = atr_median / atr
        return max(0.7, min(scale, self.p_atr_scale_max))

    # ===== 止损 =====

    def _check_stop_loss(self, position: Any, nav: float) -> Dict[str, Any]:
        """分级止损检查。"""
        if position.total_shares <= 0 or position.avg_cost <= 0:
            return {}
        pnl_ratio = (nav - position.avg_cost) / position.avg_cost

        if pnl_ratio <= self.p_hard_stop_loss:
            self._enter_cooldown()
            return {"sell_shares": position.total_shares}

        if pnl_ratio <= self.p_soft_stop_loss and not self.soft_stop_triggered:
            self.soft_stop_triggered = True
            sell_shares = self._apply_low_position_protection(position, nav, 0.20)
            if sell_shares > 0:
                self.sell_cooldown = 25
                return {"sell_shares": sell_shares}

        return {}

    # ===== 冷静期 =====

    def _enter_cooldown(self):
        """进入冷静期。"""
        self.cooldown_remaining = self.p_max_cooldown
        self.dca_counter = 0
        self.pending_batches = []
        self.soft_stop_triggered = False

    def _update_cooldown(self, indicators: Dict):
        """更新冷静期，布林带下轨反弹可提前结束。"""
        if self.cooldown_remaining <= 0:
            return
        self.cooldown_remaining -= 1
        days_elapsed = self.p_max_cooldown - self.cooldown_remaining

        if days_elapsed < self.p_min_cooldown:
            return

        # 提前退出：价格从下轨反弹至带内 + 短期动量转正
        bb_pos = indicators.get("bb_position", 0.5)
        if bb_pos > 0.3 and indicators.get("mom_5d", 0) > 0:
            self.cooldown_remaining = 0
            return

        # 提前退出：价格连续3天站上MA20
        if indicators.get("above_ma20_3d", False):
            self.cooldown_remaining = 0

    # ===== 分批建仓 =====

    def _create_batch_plan(self):
        """创建分批建仓计划。"""
        self.pending_batches = []
        for i, ratio in enumerate(self.p_trend_batch_ratios):
            self.pending_batches.append({
                "trigger_bar": self.bar_count + i * self.p_batch_interval,
                "cash_ratio": ratio,
            })
        self.pending_batches.pop(0)

    def _execute_first_batch(self, position: Any, nav: float,
                             mom_score: int = 3) -> Dict[str, Any]:
        """执行第1批建仓（基于总资产计算）。"""
        ratio = self.p_trend_batch_ratios[0]
        total_assets = self.account.cash + position.total_shares * nav
        raw_amount = total_assets * ratio
        buy_amount = self._apply_position_limit(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}
        return {"buy_amount": buy_amount}

    def _process_pending_batches(self, indicators: Dict, position: Any,
                                 nav: float) -> Dict[str, Any]:
        """处理待执行的分批建仓（第2、3批）。"""
        if not self.pending_batches:
            return {}

        next_batch = self.pending_batches[0]
        if self.bar_count < next_batch["trigger_bar"]:
            return {}

        # 超卖区暂停分批计划（不取消，等待价格回归布林带内后继续执行）
        bb_pos = indicators.get("bb_position", 0.5)
        if bb_pos < 0.0:
            return {}

        self.pending_batches.pop(0)
        ratio = next_batch["cash_ratio"]
        total_assets = self.account.cash + position.total_shares * nav
        raw_amount = total_assets * ratio
        buy_amount = self._apply_position_limit(raw_amount, position, nav)
        if buy_amount < 1.0:
            return {}
        return {"buy_amount": buy_amount}

    # ===== 仓位管理 =====

    def _apply_position_limit(self, raw_amount: float, position: Any,
                              nav: float) -> float:
        """仓位上限保护。"""
        current_market_value = position.total_shares * nav
        total_assets = self.account.cash + current_market_value
        max_market_value = total_assets * self.p_max_position_ratio
        room = max_market_value - current_market_value
        capped = min(raw_amount, room, self.account.cash)
        return max(capped, 0.0)

    def _apply_low_position_protection(self, position: Any, nav: float,
                                       target_ratio: float) -> float:
        """低仓位卖出保护。"""
        pos_ratio = self._get_position_ratio(position, nav)
        sell_shares = position.total_shares * target_ratio
        if pos_ratio < self.p_low_position_ratio and target_ratio > 0.5:
            sell_shares = position.total_shares * 0.5
        return sell_shares

    def _get_position_ratio(self, position: Any, nav: float) -> float:
        """计算当前持仓占比。"""
        market_value = position.total_shares * nav
        total_assets = self.account.cash + market_value
        if total_assets <= 0:
            return 0.0
        return market_value / total_assets
