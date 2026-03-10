# 场外基金量化分析与回测系统

这是一个面向场外基金的量化分析项目，提供基金净值拉取、技术指标计算、交易信号生成、区间回测，以及基于 Flask 的交互式分析界面。

## 功能概览

- 支持基金净值数据获取与本地 SQLite 缓存
- 提供均线、RSI、MACD 等常用技术指标
- 内置 V6 估值趋势、Regime Adaptive 状态自适应、指数动量三类策略
- 支持历史买卖信号分析与区间收益对比
- 提供 Web 页面用于基金分析、按画像动态切换策略和建议记录展示

## 项目结构

```text
otc_fund_quant/
├── analysis/      # 指标、信号、回测、图表相关逻辑
├── config/        # 策略画像、参数配置与加载
├── core/          # 账户、持仓、订单、回测引擎
├── data/          # 数据加载与本地缓存
├── strategies/    # 交易策略实现
├── tests/         # 回测和测试脚本
├── tools/         # 辅助工具脚本
├── web/           # Flask Web 应用
├── requirements.txt
└── 命令.md         # 本地运行说明
```

## 快速开始

1. 创建并激活 Python 环境。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行回测：

```bash
python tests/run_simulation.py
```

4. 启动 Web 界面：

```bash
python -m web.app
```

浏览器访问 `http://127.0.0.1:5000`。

## 说明

- 优先读取 `config/strategy_profiles.json` 进行“基金类型 -> 画像 -> 策略 -> 参数”分层解析。
- `config/strategy_params.json` 仍保留为兼容回退配置。
- 需要从旧配置生成新画像配置时，可运行 `python tools/migrate_strategy_config.py --force`。
- `data/*.db`、`web/*.db` 和 `results/` 下内容为本地缓存或运行产物，默认不纳入版本控制。
- 具体运行命令和策略说明可参考 `命令.md`。
