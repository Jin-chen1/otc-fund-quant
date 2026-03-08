"""
基金分析 Web 交互界面 — Flask 后端
"""

import sys
import os
import sqlite3
import json
import math
import re
from datetime import datetime, date
from typing import Dict

from flask import Flask, render_template, request, jsonify
import pandas as pd

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_OF_PROJECT = os.path.dirname(PROJECT_ROOT)
for p in [PROJECT_ROOT, PARENT_OF_PROJECT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from otc_fund_quant.data.loader import DataLoader
from otc_fund_quant.analysis.indicators import calc_indicators
from otc_fund_quant.analysis.signals import (
    generate_recommendation,
    generate_recommendation_regime,
    generate_recommendation_index_momentum,
    generate_signal_points,
)
from otc_fund_quant.analysis.backtest import calc_period_returns
from otc_fund_quant.analysis.chart import get_chart_data
from otc_fund_quant.config.loader import load_strategy_params

app = Flask(__name__)

# 全局数据加载器
loader = DataLoader(db_path=os.environ.get("OTC_FUND_QUANT_NAV_DB_PATH"))

# 推荐记录数据库路径
REC_DB_PATH = os.environ.get(
    "OTC_FUND_QUANT_REC_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "recommendations.db")
)

DCA_STATUS_PENDING = "pending"
DCA_STATUS_CONFIRMED = "confirmed"
FUND_CODE_PATTERN = re.compile(r"^\d{6}$")


# ===== 数据库初始化 =====

def _init_rec_db():
    """初始化推荐记录数据库。"""
    conn = sqlite3.connect(REC_DB_PATH)
    # 检查是否需要迁移：旧表没有 strategy 列
    cursor = conn.execute("PRAGMA table_info(recommendations)")
    columns = [row[1] for row in cursor.fetchall()]
    if "strategy" not in columns and "date" in columns:
        # 旧表存在，需要迁移
        conn.execute("ALTER TABLE recommendations RENAME TO recommendations_old")
        conn.execute("""
            CREATE TABLE recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                fund_code TEXT NOT NULL,
                strategy TEXT NOT NULL DEFAULT 'v6',
                action TEXT NOT NULL,
                reason TEXT,
                nav_at_signal REAL,
                next_day_nav REAL,
                correct INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, fund_code, strategy)
            )
        """)
        conn.execute("""
            INSERT INTO recommendations (date, fund_code, strategy, action, reason, nav_at_signal, next_day_nav, correct, created_at)
            SELECT date, fund_code, 'v6', action, reason, nav_at_signal, next_day_nav, correct, created_at
            FROM recommendations_old
        """)
        conn.execute("DROP TABLE recommendations_old")
        conn.commit()
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                fund_code TEXT NOT NULL,
                strategy TEXT NOT NULL DEFAULT 'v6',
                action TEXT NOT NULL,
                reason TEXT,
                nav_at_signal REAL,
                next_day_nav REAL,
                correct INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, fund_code, strategy)
            )
        """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fund_code TEXT NOT NULL,
            strategy TEXT NOT NULL,
            end_date TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(fund_code, strategy, end_date)
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS dca_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fund_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('{DCA_STATUS_PENDING}', '{DCA_STATUS_CONFIRMED}')),
            confirm_nav_date TEXT,
            confirm_nav REAL,
            shares REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            confirmed_at TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_dca_records_fund_code
        ON dca_records (fund_code)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_dca_records_status_trade_date
        ON dca_records (status, trade_date)
    """)
    conn.commit()
    conn.close()


_init_rec_db()


# ===== 辅助函数 =====

# 回测缓存有效期（天）
BACKTEST_CACHE_TTL_DAYS = 7


def _get_backtest_cache(fund_code: str, strategy: str, end_date: str):
    """读取回测缓存，TTL天内直接复用旧缓存，无需精确匹配end_date。"""
    import json
    from datetime import datetime, timedelta
    conn = sqlite3.connect(REC_DB_PATH)
    cursor = conn.cursor()
    # 先精确匹配
    cursor.execute(
        "SELECT results_json FROM backtest_cache WHERE fund_code=? AND strategy=? AND end_date=?",
        (fund_code, strategy, end_date),
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return json.loads(row[0])
    # 查找最近的缓存，TTL内复用
    cursor.execute(
        "SELECT results_json, created_at FROM backtest_cache WHERE fund_code=? AND strategy=? ORDER BY end_date DESC LIMIT 1",
        (fund_code, strategy),
    )
    row = cursor.fetchone()
    conn.close()
    if row and row[1]:
        try:
            created = datetime.strptime(row[1][:19], "%Y-%m-%d %H:%M:%S")
            if datetime.now() - created < timedelta(days=BACKTEST_CACHE_TTL_DAYS):
                return json.loads(row[0])
        except (ValueError, TypeError):
            pass
    return None


def _save_backtest_cache(fund_code: str, strategy: str, end_date: str, results: list):
    """保存回测结果到缓存。"""
    import json
    conn = sqlite3.connect(REC_DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO backtest_cache (fund_code, strategy, end_date, results_json) VALUES (?, ?, ?, ?)",
        (fund_code, strategy, end_date, json.dumps(results, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def _get_fund_history(fund_code: str) -> pd.DataFrame:
    """从数据库获取基金完整历史数据。"""
    conn = sqlite3.connect(loader.db_path)
    df = pd.read_sql_query(
        "SELECT date, nav FROM fund_nav WHERE fund_code = ? ORDER BY date",
        conn,
        params=(fund_code,),
    )
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _validate_dca_payload(data: dict):
    """校验手动定投录入参数。"""
    if not isinstance(data, dict):
        raise ValueError("请求体必须是 JSON 对象")

    fund_code = str(data.get("fund_code", "")).strip()
    if not fund_code:
        raise ValueError("请输入基金代码")
    if not FUND_CODE_PATTERN.fullmatch(fund_code):
        raise ValueError("基金代码需为6位数字")

    raw_amount = data.get("amount")
    if isinstance(raw_amount, bool):
        raise ValueError("买入金额必须为正数")
    try:
        amount = float(raw_amount)
    except (TypeError, ValueError):
        raise ValueError("买入金额必须为正数")

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("买入金额必须为正数")

    raw_trade_date = data.get("trade_date", None)
    if raw_trade_date is None:
        trade_date_value = date.today()
    else:
        trade_date_str = str(raw_trade_date).strip()
        if not trade_date_str:
            raise ValueError("请选择买入日期")
        try:
            trade_date_value = datetime.strptime(trade_date_str, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("买入日期格式必须为 YYYY-MM-DD")
        if trade_date_value > date.today():
            raise ValueError("买入日期不能晚于今天")

    return fund_code, round(amount, 2), trade_date_value.isoformat()


def _sync_dca_records():
    """同步定投记录对应基金的最新净值，并确认待处理记录。"""
    conn = sqlite3.connect(REC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT fund_code FROM dca_records ORDER BY fund_code")
    fund_codes = [row[0] for row in cursor.fetchall()]
    cursor.execute("""
        SELECT id, fund_code, trade_date, amount
        FROM dca_records
        WHERE status = ?
        ORDER BY trade_date, id
    """, (DCA_STATUS_PENDING,))
    pending_records = cursor.fetchall()
    conn.close()

    if not fund_codes:
        return

    for fund_code in fund_codes:
        try:
            loader.update_db(fund_code)
        except Exception as e:
            print(f"Warning: failed to sync NAV for {fund_code}: {e}")

    if not pending_records:
        return

    nav_conn = sqlite3.connect(loader.db_path)
    rec_conn = sqlite3.connect(REC_DB_PATH)
    try:
        nav_cursor = nav_conn.cursor()
        for record_id, fund_code, trade_date_str, amount in pending_records:
            nav_cursor.execute("""
                SELECT date, nav
                FROM fund_nav
                WHERE fund_code = ? AND date >= ?
                ORDER BY date
                LIMIT 1
            """, (fund_code, trade_date_str))
            row = nav_cursor.fetchone()
            if not row:
                continue

            confirm_nav_date, confirm_nav = row
            if confirm_nav is None:
                continue

            confirm_nav = float(confirm_nav)
            if confirm_nav <= 0:
                continue

            shares = float(amount) / confirm_nav
            rec_conn.execute("""
                UPDATE dca_records
                SET status = ?, confirm_nav_date = ?, confirm_nav = ?, shares = ?, confirmed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (DCA_STATUS_CONFIRMED, confirm_nav_date, confirm_nav, shares, record_id))

        rec_conn.commit()
    finally:
        nav_conn.close()
        rec_conn.close()


def _get_latest_nav_snapshot(fund_codes):
    """读取每只基金最新净值快照。"""
    if not fund_codes:
        return {}

    conn = sqlite3.connect(loader.db_path)
    cursor = conn.cursor()
    latest_nav_map = {}
    try:
        for fund_code in fund_codes:
            cursor.execute("""
                SELECT date, nav
                FROM fund_nav
                WHERE fund_code = ?
                ORDER BY date DESC
                LIMIT 1
            """, (fund_code,))
            row = cursor.fetchone()
            if row:
                latest_nav_map[fund_code] = {
                    "latest_nav_date": row[0],
                    "latest_nav": float(row[1]),
                }
    finally:
        conn.close()

    return latest_nav_map


def _empty_dca_snapshot():
    """返回空的定投看板数据结构。"""
    return {
        "as_of_date": date.today().isoformat(),
        "portfolio": {
            "tracked_fund_count": 0,
            "total_confirmed_amount": 0.0,
            "total_pending_amount": 0.0,
            "total_value": 0.0,
            "total_profit_amount": 0.0,
            "total_profit_pct": None,
        },
        "funds": [],
        "records": [],
    }


def _get_dca_snapshot():
    """汇总定投记录、基金净值和当前收益。"""
    _sync_dca_records()

    conn = sqlite3.connect(REC_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, fund_code, trade_date, amount, status, confirm_nav_date, confirm_nav, shares, created_at
        FROM dca_records
        ORDER BY trade_date DESC, id DESC
    """).fetchall()
    conn.close()

    if not rows:
        return _empty_dca_snapshot()

    fund_codes = sorted({row["fund_code"] for row in rows})
    latest_nav_map = _get_latest_nav_snapshot(fund_codes)

    fund_metrics = {}
    records = []

    for row in rows:
        amount = float(row["amount"])
        confirm_nav = float(row["confirm_nav"]) if row["confirm_nav"] is not None else None
        shares = float(row["shares"]) if row["shares"] is not None else None

        records.append({
            "id": row["id"],
            "trade_date": row["trade_date"],
            "fund_code": row["fund_code"],
            "amount": amount,
            "status": row["status"],
            "confirm_nav_date": row["confirm_nav_date"],
            "confirm_nav": confirm_nav,
            "shares": shares,
            "created_at": row["created_at"],
        })

        fund = fund_metrics.setdefault(row["fund_code"], {
            "fund_code": row["fund_code"],
            "start_date": row["trade_date"],
            "confirmed_amount": 0.0,
            "pending_amount": 0.0,
            "total_shares": 0.0,
            "record_count": 0,
            "pending_count": 0,
        })
        fund["start_date"] = min(fund["start_date"], row["trade_date"])
        fund["record_count"] += 1

        if row["status"] == DCA_STATUS_CONFIRMED:
            fund["confirmed_amount"] += amount
            fund["total_shares"] += shares or 0.0
        else:
            fund["pending_amount"] += amount
            fund["pending_count"] += 1

    funds = []
    latest_dates = []
    total_confirmed_amount = 0.0
    total_pending_amount = 0.0
    total_value = 0.0

    for fund_code, fund in sorted(fund_metrics.items(), key=lambda item: (item[1]["start_date"], item[0])):
        latest = latest_nav_map.get(fund_code, {})
        latest_nav = latest.get("latest_nav")
        latest_nav_date = latest.get("latest_nav_date")
        if latest_nav_date:
            latest_dates.append(latest_nav_date)

        current_value = fund["total_shares"] * latest_nav if latest_nav is not None else 0.0
        confirmed_amount = fund["confirmed_amount"]
        profit_amount = current_value - confirmed_amount if confirmed_amount > 0 else 0.0
        profit_pct = (profit_amount / confirmed_amount * 100.0) if confirmed_amount > 0 else None

        funds.append({
            "fund_code": fund_code,
            "start_date": fund["start_date"],
            "latest_nav": latest_nav,
            "latest_nav_date": latest_nav_date,
            "confirmed_amount": confirmed_amount,
            "pending_amount": fund["pending_amount"],
            "total_shares": fund["total_shares"],
            "current_value": current_value,
            "profit_amount": profit_amount,
            "profit_pct": profit_pct,
            "record_count": fund["record_count"],
            "pending_count": fund["pending_count"],
        })

        total_confirmed_amount += confirmed_amount
        total_pending_amount += fund["pending_amount"]
        total_value += current_value

    total_profit_amount = total_value - total_confirmed_amount
    total_profit_pct = (
        total_profit_amount / total_confirmed_amount * 100.0
        if total_confirmed_amount > 0 else None
    )

    return {
        "as_of_date": max(latest_dates) if latest_dates else date.today().isoformat(),
        "portfolio": {
            "tracked_fund_count": len(funds),
            "total_confirmed_amount": total_confirmed_amount,
            "total_pending_amount": total_pending_amount,
            "total_value": total_value,
            "total_profit_amount": total_profit_amount,
            "total_profit_pct": total_profit_pct,
        },
        "funds": funds,
        "records": records,
    }


def _get_latest_signal_date(fund_code: str, strategy: str = "v6") -> str:
    """查询DB中该基金指定策略最新的买卖信号日期，用于增量计算。"""
    conn = sqlite3.connect(REC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(date) FROM recommendations
        WHERE fund_code = ? AND strategy = ? AND action IN ('BUY', 'SELL')
    """, (fund_code, strategy))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def _get_all_signal_points(fund_code: str, strategy: str = "v6") -> Dict[str, list]:
    """从DB读取该基金指定策略所有买卖信号点，用于图表展示。"""
    conn = sqlite3.connect(REC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, action, nav_at_signal FROM recommendations
        WHERE fund_code = ? AND strategy = ? AND action IN ('BUY', 'SELL')
        ORDER BY date
    """, (fund_code, strategy))
    rows = cursor.fetchall()
    conn.close()

    buy_dates, buy_navs, sell_dates, sell_navs = [], [], [], []
    for date, action, nav in rows:
        if action == "BUY":
            buy_dates.append(date)
            buy_navs.append(nav)
        else:
            sell_dates.append(date)
            sell_navs.append(nav)
    return {
        "buy_dates": buy_dates, "buy_navs": buy_navs,
        "sell_dates": sell_dates, "sell_navs": sell_navs,
    }


def _auto_save_reviews(fund_code, signal_records, today_rec, latest_nav, latest_date, strategy="v6"):
    """自动将新增买卖信号 + 今日建议写入数据库。"""
    conn = sqlite3.connect(REC_DB_PATH)
    try:
        # 1. 写入新增的BUY/SELL信号记录
        for r in signal_records:
            correct_val = None
            if r["correct"] is True:
                correct_val = 1
            elif r["correct"] is False:
                correct_val = 0
            conn.execute("""
                INSERT OR REPLACE INTO recommendations
                (date, fund_code, strategy, action, reason, nav_at_signal, next_day_nav, correct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (r["date"], fund_code, strategy, r["action"], r["reason"],
                  r["nav"], r["next_nav"], correct_val))

        # 2. 写入今日建议（尚无次日NAV）
        date_str = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date)
        conn.execute("""
            INSERT OR REPLACE INTO recommendations
            (date, fund_code, strategy, action, reason, nav_at_signal)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date_str, fund_code, strategy, today_rec["action"], today_rec["reason"], latest_nav))

        conn.commit()
    finally:
        conn.close()


# ===== 路由 =====

@app.route("/")
def index():
    """主页面。"""
    return render_template("index.html")


@app.route("/dca")
def dca_dashboard():
    """定投记录与收益看板。"""
    return render_template("dca.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """分析基金：拉取最新数据 + 计算指标 + 生成建议。"""
    data = request.get_json()
    fund_code = data.get("fund_code", "").strip()
    strategy = data.get("strategy", "v6").strip()
    if not fund_code:
        return jsonify({"error": "请输入基金代码"}), 400

    try:
        # 0. 加载策略配置参数
        strategy_params = load_strategy_params(fund_code, strategy)

        # 1. 更新数据库（拉取最新数据）
        loader.update_db(fund_code)

        # 2. 读取历史数据
        history_df = _get_fund_history(fund_code)
        if history_df.empty or len(history_df) < 10:
            return jsonify({"error": f"基金 {fund_code} 数据不足或不存在"}), 404

        # 3. 获取最新NAV
        latest_nav = float(history_df.iloc[-1]["nav"])
        latest_date = history_df.iloc[-1]["date"]

        # 4. 根据选择的策略生成今日建议
        if strategy == "index_momentum":
            recommendation = generate_recommendation_index_momentum(history_df, params=strategy_params)
        elif strategy == "regime_adaptive":
            recommendation = generate_recommendation_regime(history_df, params=strategy_params)
        else:
            recommendation = generate_recommendation(history_df, params=strategy_params)

        # 5. 获取图表数据
        chart_data = get_chart_data(history_df, days=500)

        # 5.5 增量计算买卖信号：只计算DB中尚未记录的新日期
        last_signal_date = _get_latest_signal_date(fund_code, strategy)
        new_signals = generate_signal_points(history_df, after_date=last_signal_date, strategy=strategy, params=strategy_params)

        # 6. 将新增信号 + 今日建议写入数据库
        _auto_save_reviews(fund_code, new_signals["records"], recommendation, latest_nav, latest_date, strategy=strategy)

        # 7. 从DB读取完整信号点用于图表展示
        signal_points = _get_all_signal_points(fund_code, strategy)

        # 8. 获取历史推荐记录
        rec_history = _get_recommendations(fund_code, strategy)

        # 构造指标白名单
        ind_keys = [
            "current_nav", "percentile", "is_cheap_zone",
            "gold_cross", "death_cross", "rsi",
            "macd_turn_positive", "macd_5d_negative",
            "above_ma20", "above_ma20_3d",
            "ma20", "ma60", "macd_hist",
            "atr", "adx", "market_regime",
            # 布林带+动量指标
            "bb_upper", "bb_lower", "bb_position", "bb_width",
            "bb_squeeze", "bb_touched_lower_3d",
            "mom_5d", "mom_10d", "mom_20d", "atr_median",
        ]

        # 9. 计算各周期策略回测收益（优先读缓存）
        cache_end_date = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date)
        period_returns = _get_backtest_cache(fund_code, strategy, cache_end_date)
        if period_returns is None:
            period_returns = calc_period_returns(history_df, strategy=strategy, params=strategy_params)
            _save_backtest_cache(fund_code, strategy, cache_end_date, period_returns)

        resp = {
            "fund_code": fund_code,
            "latest_nav": latest_nav,
            "latest_date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
            "data_count": len(history_df),
            "strategy": strategy,
            "strategy_params": strategy_params,
            "period_returns": period_returns,
            "recommendation": {
                "action": recommendation["action"],
                "reason": recommendation["reason"],
                "buy_score": recommendation["buy_score"],
                "sell_signal": recommendation["sell_signal"],
                "indicators": {
                    k: v for k, v in recommendation["indicators"].items()
                    if k in ind_keys
                },
            },
            "chart": chart_data,
            "signal_points": signal_points,
            "rec_history": rec_history,
        }

        # 新策略额外返回 regime_info
        if strategy == "regime_adaptive" and "regime_info" in recommendation:
            resp["recommendation"]["regime_info"] = recommendation["regime_info"]

        return jsonify(resp)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"分析失败: {str(e)}"}), 500


@app.route("/api/dca/portfolio")
def api_dca_portfolio():
    """获取定投看板汇总与明细。"""
    try:
        return jsonify(_get_dca_snapshot())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"读取定投看板失败: {str(e)}"}), 500


@app.route("/api/dca/records", methods=["POST"])
def api_create_dca_record():
    """新增一条手动定投记录。"""
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    try:
        fund_code, amount, trade_date_str = _validate_dca_payload(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    conn = sqlite3.connect(REC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO dca_records (fund_code, trade_date, amount, status)
        VALUES (?, ?, ?, ?)
    """, (fund_code, trade_date_str, amount, DCA_STATUS_PENDING))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        "id": record_id,
        "fund_code": fund_code,
        "trade_date": trade_date_str,
        "amount": amount,
        "status": DCA_STATUS_PENDING,
    }), 201


@app.route("/api/dca/records/<int:record_id>", methods=["DELETE"])
def api_delete_dca_record(record_id: int):
    """删除误录的手动定投记录。"""
    conn = sqlite3.connect(REC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dca_records WHERE id = ?", (record_id,))
    conn.commit()
    deleted = cursor.rowcount
    conn.close()

    if deleted == 0:
        return jsonify({"error": "记录不存在"}), 404

    return jsonify({"success": True, "id": record_id})


@app.route("/api/recommendations/<fund_code>")
def api_get_recommendations(fund_code: str):
    """获取历史推荐记录。"""
    records = _get_recommendations(fund_code)
    return jsonify(records)


def _get_recommendations(fund_code: str, strategy: str = "v6"):
    """从数据库读取指定策略的推荐历史。"""
    conn = sqlite3.connect(REC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, action, reason, nav_at_signal, next_day_nav, correct
        FROM recommendations
        WHERE fund_code = ? AND strategy = ?
        ORDER BY date DESC
        LIMIT 200
    """, (fund_code, strategy))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "date": r[0],
            "action": r[1],
            "reason": r[2],
            "nav_at_signal": r[3],
            "next_day_nav": r[4],
            "correct": r[5],
        }
        for r in rows
    ]


if __name__ == "__main__":
    print("=" * 50)
    print("  基金分析 Web 交互界面")
    print("  访问: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host="127.0.0.1", port=5000)
