import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def _get_paths():
    """获取数据文件和输出文件路径。"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    return {
        'equity_csv': os.path.join(results_dir, 'equity.csv'),
        'trades_csv': os.path.join(results_dir, 'trades.csv'),
        'output_png': os.path.join(results_dir, 'equity.png'),
    }


def _load_equity(csv_path):
    """读取权益曲线数据并计算基准和回撤。"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # 基准：将 NAV 归一化到与策略相同的初始资产
    initial_assets = df['total_assets'].iloc[0]
    initial_nav = df['nav'].iloc[0]
    df['benchmark'] = (df['nav'] / initial_nav) * initial_assets

    # 策略回撤
    df['peak'] = df['total_assets'].cummax()
    df['drawdown'] = (df['total_assets'] - df['peak']) / df['peak']

    # 基准回撤
    df['nav_peak'] = df['nav'].cummax()
    df['nav_drawdown'] = (df['nav'] - df['nav_peak']) / df['nav_peak']

    return df


def _load_trades(csv_path):
    """读取交易记录，返回买入和卖出的 DataFrame。如果文件不存在返回 None。"""
    if not os.path.exists(csv_path):
        print(f"[Info] 交易记录文件不存在: {csv_path}，跳过买卖点标注。")
        return None, None

    trades = pd.read_csv(csv_path)
    trades['date'] = pd.to_datetime(trades['date'])

    buys = trades[trades['action'] == 'BUY'].copy()
    sells = trades[trades['action'] == 'SELL'].copy()

    return buys, sells



def _merge_trade_positions(df_equity, trades_df, value_col):
    """将交易记录与权益曲线合并，获取对应日期的标注位置值。

    只从 trades_df 取 date 列，避免与 equity 中同名列（如 nav）冲突。
    """
    merged = trades_df[['date']].merge(
        df_equity[['date', value_col]],
        on='date',
        how='left',
    )
    # 丢弃未匹配到权益数据的交易记录
    return merged.dropna(subset=[value_col])


def _plot_equity_with_trades(ax, df, buys_merged, sells_merged):
    """绘制上图：策略权益曲线 vs 基准，标注买卖点。"""
    ax.plot(df['date'], df['total_assets'],
            label='策略权益', color='#d62728', linewidth=2)
    ax.plot(df['date'], df['benchmark'],
            label='基准（买入持有）', color='gray', linestyle='--', alpha=0.6)

    if buys_merged is not None and not buys_merged.empty:
        ax.scatter(buys_merged['date'], buys_merged['total_assets'],
                   marker='^', color='green', s=16, alpha=0.7, zorder=5,
                   label=f'买入 ({len(buys_merged)})')

    if sells_merged is not None and not sells_merged.empty:
        ax.scatter(sells_merged['date'], sells_merged['total_assets'],
                   marker='v', color='red', s=36, alpha=0.9, zorder=5,
                   label=f'卖出 ({len(sells_merged)})')

    ax.set_title('策略 vs 基准 权益曲线', fontsize=14, fontweight='bold')
    ax.set_ylabel('总资产（元）', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


def _plot_nav_with_trades(ax, df, buys_merged, sells_merged):
    """绘制中图：NAV 走势 + 买卖点。"""
    ax.plot(df['date'], df['nav'],
            label='NAV', color='#1f77b4', linewidth=1.5)

    if buys_merged is not None and not buys_merged.empty:
        ax.scatter(buys_merged['date'], buys_merged['nav'],
                   marker='^', color='green', s=16, alpha=0.7, zorder=5,
                   label=f'买入 ({len(buys_merged)})')

    if sells_merged is not None and not sells_merged.empty:
        ax.scatter(sells_merged['date'], sells_merged['nav'],
                   marker='v', color='red', s=36, alpha=0.9, zorder=5,
                   label=f'卖出 ({len(sells_merged)})')

    ax.set_title('基金 NAV 走势与买卖点', fontsize=14, fontweight='bold')
    ax.set_ylabel('NAV', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


def _plot_drawdown(ax, df):
    """绘制下图：策略回撤 vs 基准回撤。"""
    ax.fill_between(df['date'], df['drawdown'] * 100, 0,
                    color='red', alpha=0.3, label='策略回撤')
    ax.plot(df['date'], df['nav_drawdown'] * 100,
            color='gray', linestyle='--', alpha=0.6, label='基准回撤')

    ax.set_title('回撤分析', fontsize=12)
    ax.set_ylabel('回撤 (%)', fontsize=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


def plot_equity():
    """主绘图函数：绘制权益曲线、NAV走势、回撤分析，并标注买卖点。"""
    paths = _get_paths()

    if not os.path.exists(paths['equity_csv']):
        print(f"Error: {paths['equity_csv']} 不存在，请先运行回测。")
        return

    print(f"[Info] 读取权益数据: {paths['equity_csv']}")
    df = _load_equity(paths['equity_csv'])

    # 加载交易记录
    buys, sells = _load_trades(paths['trades_csv'])
    has_trades = buys is not None and sells is not None

    # 合并买卖点与权益数据，获取标注位置
    buys_equity = None
    sells_equity = None
    buys_nav = None
    sells_nav = None

    if has_trades:
        buys_equity = _merge_trade_positions(df, buys, 'total_assets')
        sells_equity = _merge_trade_positions(df, sells, 'total_assets')
        buys_nav = _merge_trade_positions(df, buys, 'nav')
        sells_nav = _merge_trade_positions(df, sells, 'nav')

    # 绘制三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    _plot_equity_with_trades(ax1, df, buys_equity, sells_equity)
    _plot_nav_with_trades(ax2, df, buys_nav, sells_nav)
    _plot_drawdown(ax3, df)

    plt.tight_layout()
    plt.savefig(paths['output_png'], dpi=300)
    print(f"[Info] 图表已保存: {paths['output_png']}")


if __name__ == "__main__":
    # 配置中文字体
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    except Exception:
        pass

    plot_equity()
