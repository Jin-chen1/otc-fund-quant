"""
策略参数配置加载器 — 根据基金代码和策略名加载对应参数。

配置文件路径: config/strategy_params.json
合并逻辑:
  1. 加载 default[strategy] 作为基础参数
  2. regime_adaptive 通过 _inherit 继承 v6 参数
  3. 用 fund_code 级别的覆盖值合并
"""

import json
import os
from typing import Dict, Any, Optional

# 配置文件路径
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_params.json")

# 缓存已加载的配置，避免重复读取文件
_config_cache: Optional[dict] = None


def _load_config() -> dict:
    """读取并缓存配置文件。"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _config_cache = json.load(f)
    return _config_cache


def reload_config():
    """强制重新加载配置文件（修改配置后调用）。"""
    global _config_cache
    _config_cache = None
    return _load_config()


def load_strategy_params(fund_code: str, strategy: str) -> Dict[str, Any]:
    """根据基金代码和策略名加载合并后的参数字典。

    合并优先级: default[strategy] < fund_code[strategy]
    regime_adaptive 会先继承 default[v6]，再覆盖 default[regime_adaptive]，
    最后覆盖 fund_code[regime_adaptive]。

    Args:
        fund_code: 基金代码，如 '007343'
        strategy: 策略名，'v6' 或 'regime_adaptive'

    Returns:
        合并后的参数字典
    """
    config = _load_config()
    defaults = config.get("default", {})

    # 第一步：构建基础参数（default.v6）
    base_params = dict(defaults.get("v6", {}))

    if strategy == "index_momentum":
        # index_momentum: 独立参数体系，不继承v6
        # default.index_momentum → fund.index_momentum
        params = dict(defaults.get("index_momentum", {}))
        fund_overrides = config.get(fund_code, {}).get("index_momentum", {})
        if fund_overrides:
            params.update(fund_overrides)
    elif strategy == "v6":
        # v6: default.v6 → fund.v6
        params = base_params
        fund_overrides = config.get(fund_code, {}).get("v6", {})
        if fund_overrides:
            params.update(fund_overrides)
    else:
        # regime_adaptive 合并顺序:
        # default.v6 → fund.v6 → default.regime_adaptive → fund.regime_adaptive
        fund_v6_overrides = config.get(fund_code, {}).get("v6", {})
        if fund_v6_overrides:
            base_params.update(fund_v6_overrides)

        regime_defaults = dict(defaults.get("regime_adaptive", {}))
        regime_defaults.pop("_inherit", None)  # 移除元数据键
        params = {**base_params, **regime_defaults}

        fund_regime_overrides = config.get(fund_code, {}).get("regime_adaptive", {})
        if fund_regime_overrides:
            params.update(fund_regime_overrides)

    return params


def get_all_fund_configs() -> Dict[str, list]:
    """获取所有已配置的基金代码列表（不含 default）。"""
    config = _load_config()
    return {k: list(v.keys()) for k, v in config.items() if k != "default"}
