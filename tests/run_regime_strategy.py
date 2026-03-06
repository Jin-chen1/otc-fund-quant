"""
对比回测：V6策略 vs 市场状态自适应策略
"""

import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from otc_fund_quant.data.loader import DataLoader
from otc_fund_quant.core.account import Account
from otc_fund_quant.core.engine import BacktestEngine
from otc_fund_quant.strategies.trend_strategy import ValuationTrendHybridStrategy
from otc_fund_quant.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy


def run_backtest(strategy_class, strategy_name, fund_code, start_date, end_date,
                 initial_cash, loader):
    """运行单个策略的回测并返回结果。"""
    account = Account(initial_cash=initial_cash)
    engine = BacktestEngine(account, loader)
    strategy = strategy_class(account, fund_code)

    engine.run(fund_code, start_date, end_date, strategy)

    pos = account.get_position(fund_code)
    df_equity = pd.DataFrame(engine.equity_curve)

    if df_equity.empty:
        print(f"  {strategy_name}: No data generated.")
        return None

    last_nav = df_equity.iloc[-1]['nav']
    market_value = pos.total_shares * last_nav
    total_assets = account.cash + market_value
    total_return = (total_assets - initial_cash) / initial_cash * 100

    # 基准收益
    start_nav = df_equity.iloc[0]['nav']
    end_nav = df_equity.iloc[-1]['nav']
    benchmark_return = (end_nav - start_nav) / start_nav * 100

    # 最大回撤
    df_equity['peak'] = df_equity['total_assets'].cummax()
    df_equity['drawdown'] = (df_equity['total_assets'] - df_equity['peak']) / df_equity['peak']
    max_dd = df_equity['drawdown'].min() * 100

    # 基准回撤
    df_equity['nav_peak'] = df_equity['nav'].cummax()
    df_equity['nav_dd'] = (df_equity['nav'] - df_equity['nav_peak']) / df_equity['nav_peak']
    bench_max_dd = df_equity['nav_dd'].min() * 100

    # 交易次数
    trade_count = len(engine.trade_logs)
    buy_count = sum(1 for t in engine.trade_logs if t['action'] == 'BUY')
    sell_count = sum(1 for t in engine.trade_logs if t['action'] == 'SELL')

    # 年化收益率（简单计算）
    days = (df_equity.iloc[-1]['date'] - df_equity.iloc[0]['date']).days
    years = days / 365.25
    annualized = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 年化波动率
    df_equity['daily_return'] = df_equity['total_assets'].pct_change()
    volatility = df_equity['daily_return'].std() * (252 ** 0.5) * 100

    # 夏普比率（假设无风险利率2%）
    risk_free = 2.0
    sharpe = (annualized - risk_free) / volatility if volatility > 0 else 0

    return {
        "name": strategy_name,
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "max_drawdown": max_dd,
        "bench_max_dd": bench_max_dd,
        "annualized": annualized,
        "volatility": volatility,
        "sharpe": sharpe,
        "trade_count": trade_count,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "final_assets": total_assets,
        "final_cash": account.cash,
        "final_shares": pos.total_shares,
    }


def print_comparison(results):
    """打印对比结果。"""
    print("\n" + "=" * 70)
    print("                    策 略 对 比 结 果")
    print("=" * 70)

    headers = ["指标", results[0]["name"], results[1]["name"]]
    rows = [
        ("总收益率", f"{results[0]['total_return']:.2f}%", f"{results[1]['total_return']:.2f}%"),
        ("年化收益率", f"{results[0]['annualized']:.2f}%", f"{results[1]['annualized']:.2f}%"),
        ("基准收益率", f"{results[0]['benchmark_return']:.2f}%", f"{results[1]['benchmark_return']:.2f}%"),
        ("超额收益", f"{results[0]['excess_return']:.2f}%", f"{results[1]['excess_return']:.2f}%"),
        ("最大回撤", f"{results[0]['max_drawdown']:.2f}%", f"{results[1]['max_drawdown']:.2f}%"),
        ("基准最大回撤", f"{results[0]['bench_max_dd']:.2f}%", f"{results[1]['bench_max_dd']:.2f}%"),
        ("年化波动率", f"{results[0]['volatility']:.2f}%", f"{results[1]['volatility']:.2f}%"),
        ("夏普比率", f"{results[0]['sharpe']:.2f}", f"{results[1]['sharpe']:.2f}"),
        ("交易次数", f"{results[0]['trade_count']}", f"{results[1]['trade_count']}"),
        ("买入次数", f"{results[0]['buy_count']}", f"{results[1]['buy_count']}"),
        ("卖出次数", f"{results[0]['sell_count']}", f"{results[1]['sell_count']}"),
        ("最终总资产", f"{results[0]['final_assets']:.0f}", f"{results[1]['final_assets']:.0f}"),
    ]

    print(f"\n{'指标':<14} {headers[1]:<20} {headers[2]:<20}")
    print("-" * 54)
    for label, v1, v2 in rows:
        print(f"{label:<14} {v1:<20} {v2:<20}")

    # 胜者判断
    print("\n" + "-" * 54)
    winner_return = 0 if results[0]['total_return'] > results[1]['total_return'] else 1
    winner_dd = 0 if results[0]['max_drawdown'] > results[1]['max_drawdown'] else 1
    winner_sharpe = 0 if results[0]['sharpe'] > results[1]['sharpe'] else 1

    print(f"收益率胜者:  {results[winner_return]['name']}")
    print(f"回撤控制胜者: {results[winner_dd]['name']}")
    print(f"夏普比率胜者: {results[winner_sharpe]['name']}")


def main():
    fund_code = '007343'
    start_date = '2021-01-01'
    end_date = '2026-01-01'
    initial_cash = 200000.0

    print("=" * 70)
    print(f"  回测对比: {fund_code} | {start_date} ~ {end_date} | 初始资金 {initial_cash:.0f}")
    print("=" * 70)

    loader = DataLoader()
    print("Fetching data...")
    loader.update_db(fund_code)

    print("\n--- Running V6 Strategy ---")
    r1 = run_backtest(ValuationTrendHybridStrategy, "V6估值趋势",
                      fund_code, start_date, end_date, initial_cash, loader)

    print("\n--- Running Regime-Adaptive Strategy ---")
    r2 = run_backtest(RegimeAdaptiveStrategy, "状态自适应",
                      fund_code, start_date, end_date, initial_cash, loader)

    if r1 and r2:
        print_comparison([r1, r2])
    else:
        print("ERROR: One or more strategies failed to produce results.")


if __name__ == "__main__":
    main()
