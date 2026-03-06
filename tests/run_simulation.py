import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from otc_fund_quant.data.loader import DataLoader
from otc_fund_quant.core.account import Account
from otc_fund_quant.core.engine import BacktestEngine
from otc_fund_quant.strategies.trend_strategy import ValuationTrendHybridStrategy

def main():
    fund_code = '007343' # 东方新能源汽车混合
    start_date = '2021-01-01'
    end_date = '2026-01-01'
    initial_cash = 200000.0 # Increased cash to handle 3 years of trading

    print("Initializing System...")
    loader = DataLoader()

    # Ensure we have data
    print(f"Fetching data for {fund_code}...")
    loader.update_db(fund_code)

    account = Account(initial_cash=initial_cash)
    engine = BacktestEngine(account, loader)

    # Initialize Strategy
    strategy = ValuationTrendHybridStrategy(account, fund_code)

    print(f"\n--- Starting Hybrid Valuation Trend Simulation for {fund_code} ---")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: {initial_cash}")
    print("Strategy V6: V1 minimal tuning; DCA 4%/7d/20%cheap; 8-25d cooldown; all else V1 original")

    engine.run(fund_code, start_date, end_date, strategy)

    print("\n--- Simulation Results ---")

    # Calculate Final Value
    pos = account.get_position(fund_code)
    # Get last available NAV for valuation
    last_nav = 1.0
    if not engine.nav_data.empty:
        # engine.nav_data contains all fetched data, finding the last date in simulation
        df_equity = pd.DataFrame(engine.equity_curve)
        if not df_equity.empty:
            last_nav = df_equity.iloc[-1]['nav']
            last_date = df_equity.iloc[-1]['date']
            print(f"Last Simulation Date: {last_date}, NAV: {last_nav}")
        else:
            print("No equity data generated.")

    market_value = pos.total_shares * last_nav
    total_assets = account.cash + market_value
    total_return = (total_assets - initial_cash) / initial_cash * 100

    print(f"Final Cash: {account.cash:.2f}")
    print(f"Shares Held: {pos.total_shares:.2f}")
    print(f"Market Value: {market_value:.2f}")
    print(f"Total Assets: {total_assets:.2f}")
    print(f"Total Return: {total_return:.2f}%")

    # --- Benchmark Analysis ---
    if not df_equity.empty:
        # Save results to CSV
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, 'equity.csv')
        df_equity.to_csv(csv_path, index=False)
        print(f"\n[Data] Equity curve saved to: {csv_path}")

        start_nav = df_equity.iloc[0]['nav']
        end_nav = df_equity.iloc[-1]['nav']
        benchmark_return = (end_nav - start_nav) / start_nav * 100
        print(f"\nBenchmark (Buy & Hold) Return: {benchmark_return:.2f}%")
        print(f"Strategy vs Benchmark: {total_return - benchmark_return:.2f}%")

        # --- Max Drawdown ---
        df_equity['peak'] = df_equity['total_assets'].cummax()
        df_equity['drawdown'] = (df_equity['total_assets'] - df_equity['peak']) / df_equity['peak']
        max_dd = df_equity['drawdown'].min() * 100
        print(f"Max Drawdown: {max_dd:.2f}%")

        # Benchmark Drawdown
        df_equity['nav_peak'] = df_equity['nav'].cummax()
        df_equity['nav_drawdown'] = (df_equity['nav'] - df_equity['nav_peak']) / df_equity['nav_peak']
        bench_max_dd = df_equity['nav_drawdown'].min() * 100
        print(f"Benchmark Max Drawdown: {bench_max_dd:.2f}%")

    # --- Trade Logs ---
    print("\n--- Trade History ---")
    if engine.trade_logs:
        logs_df = pd.DataFrame(engine.trade_logs)
        # 保存交易记录到 CSV
        trades_csv_path = os.path.join(results_dir, 'trades.csv')
        logs_df.to_csv(trades_csv_path, index=False)
        print(f"[Data] Trade logs saved to: {trades_csv_path}")
        # Print first 5 and last 5 if too many
        if len(logs_df) > 20:
            print(logs_df.head(10).to_string(index=False))
            print("...")
            print(logs_df.tail(10).to_string(index=False))
        else:
            print(logs_df.to_string(index=False))
    else:
        print("No trades executed.")

    if total_return > 0:
        print("\nRESULT: PROFIT")
    else:
        print("\nRESULT: LOSS")

if __name__ == "__main__":
    main()
