"""
基金分析 Web 交互界面 — Flask 后端
"""

import sys
import os
import sqlite3
import json
import math
import hashlib
import copy
import re
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, date, timedelta
from time import perf_counter
from typing import Any, Dict

from flask import Flask, render_template, request, jsonify
import pandas as pd

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_OF_PROJECT = os.path.dirname(PROJECT_ROOT)
for p in [PROJECT_ROOT, PARENT_OF_PROJECT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from otc_fund_quant.data.loader import DataLoader
from otc_fund_quant.data.sqlite_utils import connect_sqlite, run_sqlite_write_with_retry
from otc_fund_quant.analysis.signals import (
    generate_signal_points,
    validate_recommendation_payload,
)
from otc_fund_quant.analysis.backtest import calc_period_returns, get_backtest_trades
from otc_fund_quant.analysis.chart import get_chart_data
from otc_fund_quant.analysis.strategy_registry import get_strategy_definition
from otc_fund_quant.config.loader import resolve_strategy_context
from otc_fund_quant.nav_estimator import NAVEngine
from otc_fund_quant.nav_estimator.data.fetcher.fund_fetcher import FundFetcher

app = Flask(__name__)
logger = logging.getLogger(__name__)

# 全局数据加载器
loader = DataLoader(db_path=os.environ.get("OTC_FUND_QUANT_NAV_DB_PATH"))
dca_fund_fetcher = FundFetcher()
if "pytest" not in sys.modules:
    FundFetcher.schedule_a_index_catalog_prewarm()

# 推荐记录数据库路径
REC_DB_PATH = os.environ.get(
    "OTC_FUND_QUANT_REC_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "recommendations.db")
)

DCA_STATUS_PENDING = "pending"
DCA_STATUS_CONFIRMED = "confirmed"
DCA_SNAPSHOT_STALE_TTL = timedelta(seconds=60)
FUND_CODE_PATTERN = re.compile(r"^\d{6}$")
NAV_ESTIMATOR_SUMMARY_KEYS = {
    "fund_code",
    "status",
    "fund_type",
    "estimated_nav",
    "estimated_return",
    "nav_date",
    "target_date",
    "confidence_score",
    "warnings",
}

BACKTEST_STATUS_READY = "ready"
BACKTEST_STATUS_PENDING = "pending"
BACKTEST_STATUS_ERROR = "error"
BACKTEST_EXECUTOR_WORKERS = 1
BACKTEST_TASK_KEY = tuple[str, str, str, str]
BACKTEST_PENDING_TTL = timedelta(minutes=15)
BACKTEST_FINISHED_TTL = timedelta(minutes=10)
INDICATOR_RESPONSE_KEYS = (
    "current_nav", "percentile", "is_cheap_zone",
    "gold_cross", "death_cross", "rsi",
    "macd_turn_positive", "macd_5d_negative",
    "above_ma20", "above_ma20_3d",
    "ma20", "ma60", "macd_hist",
    "atr", "adx", "market_regime",
    "bb_upper", "bb_lower", "bb_position", "bb_width",
    "bb_squeeze", "bb_touched_lower_3d",
    "mom_5d", "mom_10d", "mom_20d", "atr_median",
)

_backtest_executor: ThreadPoolExecutor | None = None
_backtest_executor_lock = threading.Lock()
_backtest_tasks: dict[BACKTEST_TASK_KEY, dict[str, Any]] = {}
_backtest_tasks_lock = threading.Lock()
_dca_snapshot_lock = threading.Lock()
_dca_snapshot_thread: threading.Thread | None = None
_dca_snapshot_state: dict[str, Any] = {
    "payload": None,
    "generated_at": None,
    "last_sync_started_at": None,
    "last_sync_completed_at": None,
    "last_sync_error": None,
    "is_syncing": False,
    "dirty": True,
}


# ===== 数据库初始化 =====

def _init_rec_db():
    """初始化推荐记录数据库。"""
    def _action():
        conn = connect_sqlite(REC_DB_PATH)
        try:
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
            backtest_cursor = conn.execute("PRAGMA table_info(backtest_cache)")
            backtest_columns = [row[1] for row in backtest_cursor.fetchall()]
            if backtest_columns and "params_hash" not in backtest_columns:
                conn.execute("ALTER TABLE backtest_cache RENAME TO backtest_cache_legacy")
                conn.execute("""
                    CREATE TABLE backtest_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fund_code TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        params_hash TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        results_json TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(fund_code, strategy, params_hash, end_date)
                    )
                """)
                conn.execute("DROP TABLE backtest_cache_legacy")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fund_code TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        params_hash TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        results_json TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(fund_code, strategy, params_hash, end_date)
                    )
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_cache_lookup
                ON backtest_cache (fund_code, strategy, params_hash, end_date DESC)
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
        finally:
            conn.close()

    run_sqlite_write_with_retry(_action, logger=logger, action_name="web_init_rec_db")


_init_rec_db()


# ===== 辅助函数 =====

# 回测缓存有效期（天）
BACKTEST_CACHE_TTL_DAYS = 7


def _get_backtest_cache(fund_code: str, strategy: str, params_hash: str, end_date: str):
    """读取回测缓存，TTL天内直接复用旧缓存，无需精确匹配end_date。"""
    conn = connect_sqlite(REC_DB_PATH)
    cursor = conn.cursor()
    try:
        # 先精确匹配
        cursor.execute(
            "SELECT results_json FROM backtest_cache WHERE fund_code=? AND strategy=? AND params_hash=? AND end_date=?",
            (fund_code, strategy, params_hash, end_date),
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        # 查找最近的缓存，TTL内复用
        cursor.execute(
            "SELECT results_json, created_at FROM backtest_cache WHERE fund_code=? AND strategy=? AND params_hash=? ORDER BY end_date DESC LIMIT 1",
            (fund_code, strategy, params_hash),
        )
        row = cursor.fetchone()
    finally:
        conn.close()
    if row and row[1]:
        try:
            created = datetime.strptime(row[1][:19], "%Y-%m-%d %H:%M:%S")
            if datetime.now() - created < timedelta(days=BACKTEST_CACHE_TTL_DAYS):
                return json.loads(row[0])
        except (ValueError, TypeError):
            pass
    return None


def _save_backtest_cache(fund_code: str, strategy: str, params_hash: str, end_date: str, results: list):
    """保存回测结果到缓存。"""
    def _action():
        conn = connect_sqlite(REC_DB_PATH)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO backtest_cache (fund_code, strategy, params_hash, end_date, results_json) VALUES (?, ?, ?, ?, ?)",
                (fund_code, strategy, params_hash, end_date, json.dumps(results, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    run_sqlite_write_with_retry(_action, logger=logger, action_name="save_backtest_cache")


def _backtest_now() -> datetime:
    return datetime.now()


def _normalize_backtest_params_for_hash(value: Any):
    if isinstance(value, dict):
        return {
            str(key): _normalize_backtest_params_for_hash(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_backtest_params_for_hash(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _stable_backtest_params_hash(params: dict[str, Any] | None) -> str:
    normalized = _normalize_backtest_params_for_hash(params or {})
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_indicator_value(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    try:
        is_na = pd.isna(value)
    except Exception:
        is_na = False

    if isinstance(is_na, bool) and is_na:
        return None
    return value


def _normalize_indicator_payload(indicators: dict[str, Any] | None) -> dict[str, Any]:
    raw_indicators = indicators if isinstance(indicators, dict) else {}
    return {
        key: _normalize_indicator_value(raw_indicators.get(key))
        for key in INDICATOR_RESPONSE_KEYS
    }


def _create_backtest_task(*, status: str, error: str | None = None, period_returns: list | None = None) -> dict[str, Any]:
    now = _backtest_now()
    task = {
        "task_id": uuid.uuid4().hex,
        "status": status,
        "error": error,
        "period_returns": period_returns,
        "created_at": now,
        "updated_at": now,
        "started_at": now if status == BACKTEST_STATUS_PENDING else None,
        "completed_at": now if status in {BACKTEST_STATUS_READY, BACKTEST_STATUS_ERROR} else None,
        "future": None,
    }
    return task


def _touch_backtest_task(
    task: dict[str, Any],
    *,
    status: str | None = None,
    error: str | None = None,
    period_returns: list | None = None,
    now: datetime | None = None,
):
    timestamp = now or _backtest_now()
    if status is not None:
        task["status"] = status
    task["error"] = error
    task["period_returns"] = period_returns
    task["updated_at"] = timestamp
    if task.get("started_at") is None and task.get("status") == BACKTEST_STATUS_PENDING:
        task["started_at"] = timestamp
    if task.get("status") in {BACKTEST_STATUS_READY, BACKTEST_STATUS_ERROR}:
        task["completed_at"] = timestamp


def _task_timestamp(task: dict[str, Any], *keys: str) -> datetime | None:
    for key in keys:
        value = task.get(key)
        if isinstance(value, datetime):
            return value
    return None


def _prune_backtest_tasks():
    now = _backtest_now()
    stale_keys: list[BACKTEST_TASK_KEY] = []
    prune_keys: list[BACKTEST_TASK_KEY] = []

    with _backtest_tasks_lock:
        for task_key, task in list(_backtest_tasks.items()):
            status = task.get("status")
            if status == BACKTEST_STATUS_PENDING:
                started_at = _task_timestamp(task, "started_at", "updated_at", "created_at")
                if started_at is not None and now - started_at > BACKTEST_PENDING_TTL:
                    _touch_backtest_task(
                        task,
                        status=BACKTEST_STATUS_ERROR,
                        error="区间收益任务已超时，请重新分析",
                        period_returns=None,
                        now=now,
                    )
                    task["future"] = None
                    stale_keys.append(task_key)
                    status = BACKTEST_STATUS_ERROR

            if status in {BACKTEST_STATUS_READY, BACKTEST_STATUS_ERROR}:
                completed_at = _task_timestamp(task, "completed_at", "updated_at", "created_at")
                if completed_at is not None and now - completed_at > BACKTEST_FINISHED_TTL:
                    prune_keys.append(task_key)

        for task_key in prune_keys:
            _backtest_tasks.pop(task_key, None)

    for task_key in stale_keys:
        logger.warning("backtest_stale task_key=%s", task_key)
    for task_key in prune_keys:
        logger.info("backtest_pruned task_key=%s", task_key)


def _should_skip_background_executor() -> bool:
    """避免在 Flask debug reloader 父进程中误建后台线程。"""
    return bool(app.debug) and not app.testing and os.environ.get("WERKZEUG_RUN_MAIN") != "true"


def _ensure_backtest_executor() -> ThreadPoolExecutor | None:
    """惰性初始化回测后台执行器。"""
    global _backtest_executor
    if _backtest_executor is not None:
        return _backtest_executor
    if _should_skip_background_executor():
        return None

    with _backtest_executor_lock:
        if _backtest_executor is None and not _should_skip_background_executor():
            _backtest_executor = ThreadPoolExecutor(
                max_workers=BACKTEST_EXECUTOR_WORKERS,
                thread_name_prefix="period-returns",
            )
    return _backtest_executor


def _backtest_task_key(fund_code: str, strategy: str, params_hash: str, end_date: str) -> BACKTEST_TASK_KEY:
    return (fund_code, strategy, params_hash, end_date)


def _snapshot_backtest_task(task_key: BACKTEST_TASK_KEY) -> dict[str, Any] | None:
    with _backtest_tasks_lock:
        task = _backtest_tasks.get(task_key)
        if task is None:
            return None
        return {
            "task_id": task.get("task_id"),
            "status": task.get("status"),
            "error": task.get("error"),
            "period_returns": task.get("period_returns"),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at"),
        }


def _finalize_backtest_task(task_key: BACKTEST_TASK_KEY, expected_task_id: str, future: Future):
    """回测后台任务完成后更新内存态。"""
    try:
        result = future.result()
    except Exception as exc:
        logger.exception("backtest_failed task_key=%s error=%s", task_key, exc)
        result = {
            "status": BACKTEST_STATUS_ERROR,
            "error": f"区间收益计算失败: {exc}",
            "period_returns": None,
        }

    with _backtest_tasks_lock:
        task = _backtest_tasks.get(task_key)
        if task is None or task.get("task_id") != expected_task_id:
            return
        _touch_backtest_task(
            task,
            status=result["status"],
            error=result["error"],
            period_returns=result["period_returns"],
        )
        task["future"] = None


def _run_backtest_task(
    fund_code: str,
    strategy: str,
    params_hash: str,
    end_date: str,
    history_df: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """后台执行区间收益回测，并将结果写入缓存。"""
    start = perf_counter()
    try:
        period_returns = calc_period_returns(history_df, strategy=strategy, params=params)
        _save_backtest_cache(fund_code, strategy, params_hash, end_date, period_returns)
        logger.info(
            "backtest_done fund_code=%s strategy=%s params_hash=%s end_date=%s elapsed=%.2fs",
            fund_code,
            strategy,
            params_hash,
            end_date,
            perf_counter() - start,
        )
        return {
            "status": BACKTEST_STATUS_READY,
            "error": None,
            "period_returns": period_returns,
        }
    except Exception as exc:
        logger.exception(
            "backtest_failed fund_code=%s strategy=%s params_hash=%s end_date=%s elapsed=%.2fs error=%s",
            fund_code,
            strategy,
            params_hash,
            end_date,
            perf_counter() - start,
            exc,
        )
        return {
            "status": BACKTEST_STATUS_ERROR,
            "error": f"区间收益计算失败: {exc}",
            "period_returns": None,
        }


def _queue_backtest_task(
    fund_code: str,
    strategy: str,
    params_hash: str,
    end_date: str,
    history_df: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """提交后台回测任务；相同 key 若已在运行则直接复用。"""
    _prune_backtest_tasks()
    task_key = _backtest_task_key(fund_code, strategy, params_hash, end_date)
    with _backtest_tasks_lock:
        existing = _backtest_tasks.get(task_key)
        if existing and existing.get("status") == BACKTEST_STATUS_PENDING:
            logger.info(
                "backtest_deduped fund_code=%s strategy=%s params_hash=%s end_date=%s",
                fund_code,
                strategy,
                params_hash,
                end_date,
            )
            return {
                "status": BACKTEST_STATUS_PENDING,
                "error": None,
                "period_returns": None,
            }
        _backtest_tasks[task_key] = _create_backtest_task(status=BACKTEST_STATUS_PENDING)

    executor = _ensure_backtest_executor()
    if executor is None:
        with _backtest_tasks_lock:
            task = _backtest_tasks.get(task_key)
            if task is not None:
                _touch_backtest_task(
                    task,
                    status=BACKTEST_STATUS_ERROR,
                    error="区间收益后台任务不可用，请稍后重试",
                    period_returns=None,
                )
        logger.error(
            "backtest_failed fund_code=%s strategy=%s end_date=%s reason=executor_unavailable",
            fund_code,
            strategy,
            end_date,
        )
        return {
            "status": BACKTEST_STATUS_ERROR,
            "error": "区间收益后台任务不可用，请稍后重试",
            "period_returns": None,
        }

    try:
        future = executor.submit(
            _run_backtest_task,
            fund_code,
            strategy,
            params_hash,
            end_date,
            history_df.copy(deep=True),
            dict(params or {}),
        )
    except Exception as exc:
        with _backtest_tasks_lock:
            task = _backtest_tasks.get(task_key)
            if task is not None:
                _touch_backtest_task(
                    task,
                    status=BACKTEST_STATUS_ERROR,
                    error=f"区间收益后台任务提交失败: {exc}",
                    period_returns=None,
                )
        logger.exception(
            "backtest_failed fund_code=%s strategy=%s params_hash=%s end_date=%s reason=submit_failed error=%s",
            fund_code,
            strategy,
            params_hash,
            end_date,
            exc,
        )
        return {
            "status": BACKTEST_STATUS_ERROR,
            "error": f"区间收益后台任务提交失败: {exc}",
            "period_returns": None,
        }
    with _backtest_tasks_lock:
        task = _backtest_tasks.get(task_key)
        if task is not None:
            task["future"] = future
            task["started_at"] = task.get("started_at") or _backtest_now()
            task["updated_at"] = _backtest_now()
            task_id = str(task.get("task_id"))
        else:
            task_id = ""

    future.add_done_callback(lambda fut, key=task_key, expected_task_id=task_id: _finalize_backtest_task(key, expected_task_id, fut))
    logger.info(
        "backtest_queued fund_code=%s strategy=%s params_hash=%s end_date=%s",
        fund_code,
        strategy,
        params_hash,
        end_date,
    )
    return {
        "status": BACKTEST_STATUS_PENDING,
        "error": None,
        "period_returns": None,
    }


def _get_period_returns_payload(fund_code: str, strategy: str, params_hash: str, end_date: str) -> dict[str, Any]:
    """读取区间收益状态：优先缓存，其次内存中的后台任务状态。"""
    _prune_backtest_tasks()
    period_returns = _get_backtest_cache(fund_code, strategy, params_hash, end_date)
    if period_returns is not None:
        with _backtest_tasks_lock:
            if _backtest_tasks.pop(_backtest_task_key(fund_code, strategy, params_hash, end_date), None) is not None:
                logger.info(
                    "backtest_pruned task_key=%s",
                    _backtest_task_key(fund_code, strategy, params_hash, end_date),
                )
        logger.info(
            "cache_hit fund_code=%s strategy=%s params_hash=%s end_date=%s",
            fund_code,
            strategy,
            params_hash,
            end_date,
        )
        return {
            "status": BACKTEST_STATUS_READY,
            "period_returns": period_returns,
            "error": None,
        }

    task = _snapshot_backtest_task(_backtest_task_key(fund_code, strategy, params_hash, end_date))
    if task is not None:
        return task

    return {
        "status": BACKTEST_STATUS_ERROR,
        "period_returns": None,
        "error": "区间收益任务不存在，请重新分析",
    }


def _ensure_period_returns(
    fund_code: str,
    strategy: str,
    params_hash: str,
    end_date: str,
    history_df: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """确保区间收益要么已缓存，要么已进入后台排队。"""
    _prune_backtest_tasks()
    cached_payload = _get_period_returns_payload(fund_code, strategy, params_hash, end_date)
    if cached_payload["status"] == BACKTEST_STATUS_READY:
        return cached_payload
    return _queue_backtest_task(fund_code, strategy, params_hash, end_date, history_df, params)


def _resolve_period_returns_params_hash(
    fund_code: str,
    strategy: str,
    raw_params_hash: str = "",
) -> str:
    params_hash = str(raw_params_hash or "").strip()
    if params_hash:
        return params_hash
    try:
        strategy_context = resolve_strategy_context(fund_code, strategy)
        return _stable_backtest_params_hash(strategy_context.strategy_params)
    except Exception:
        return _stable_backtest_params_hash({})


def _reset_backtest_runtime_state(wait: bool = False):
    """测试和重载场景下重置后台执行器与任务状态。"""
    global _backtest_executor
    with _backtest_tasks_lock:
        _backtest_tasks.clear()

    executor = _backtest_executor
    _backtest_executor = None
    if executor is not None:
        executor.shutdown(wait=wait, cancel_futures=True)


def _get_fund_history(fund_code: str) -> pd.DataFrame:
    """从数据库获取基金完整历史数据。"""
    conn = connect_sqlite(loader.db_path)
    try:
        df = pd.read_sql_query(
            "SELECT date, nav FROM fund_nav WHERE fund_code = ? ORDER BY date",
            conn,
            params=(fund_code,),
        )
    finally:
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
    conn = connect_sqlite(REC_DB_PATH)
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

    def _action():
        nav_conn = connect_sqlite(loader.db_path)
        rec_conn = connect_sqlite(REC_DB_PATH)
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

    run_sqlite_write_with_retry(_action, logger=logger, action_name="sync_dca_records")


def _get_latest_nav_snapshot(fund_codes):
    """读取每只基金最新净值快照。"""
    if not fund_codes:
        return {}

    conn = connect_sqlite(loader.db_path)
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


def _get_fund_name_map(fund_codes: list[str]) -> dict[str, str | None]:
    """读取基金名称，单只失败时返回空名称而不影响整体看板。"""
    if not fund_codes:
        return {}

    fund_name_map: dict[str, str | None] = {}
    for fund_code in fund_codes:
        try:
            fund_info = dca_fund_fetcher.get_fund_info(fund_code)
            fund_name = str(fund_info.get("name", "")).strip() or None
        except Exception as e:
            print(f"Warning: failed to fetch fund name for {fund_code}: {e}")
            fund_name = None
        fund_name_map[fund_code] = fund_name

    return fund_name_map


def _clone_dca_snapshot_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return copy.deepcopy(payload)


def _serialize_dca_sync_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(timespec="seconds")


def _mark_dca_snapshot_dirty(*, invalidate_payload: bool = False):
    with _dca_snapshot_lock:
        _dca_snapshot_state["dirty"] = True
        if invalidate_payload:
            _dca_snapshot_state["payload"] = None
            _dca_snapshot_state["generated_at"] = None


def _build_dca_sync_meta(now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now()
    with _dca_snapshot_lock:
        payload = _dca_snapshot_state["payload"]
        generated_at = _dca_snapshot_state["generated_at"]
        last_sync_completed_at = _dca_snapshot_state["last_sync_completed_at"]
        last_sync_error = _dca_snapshot_state["last_sync_error"]
        is_syncing = bool(_dca_snapshot_state["is_syncing"])
        dirty = bool(_dca_snapshot_state["dirty"])

    is_stale = payload is None or generated_at is None or dirty
    if not is_stale and now - generated_at > DCA_SNAPSHOT_STALE_TTL:
        is_stale = True

    return {
        "is_syncing": is_syncing,
        "is_stale": is_stale,
        "last_sync_completed_at": _serialize_dca_sync_time(last_sync_completed_at),
        "last_sync_error": last_sync_error,
    }


def _attach_dca_sync_meta(payload: dict[str, Any]) -> dict[str, Any]:
    response_payload = _clone_dca_snapshot_payload(payload) or _empty_dca_snapshot()
    response_payload["sync"] = _build_dca_sync_meta()
    return response_payload


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


def _build_dca_snapshot_from_db():
    """从本地数据库构建定投看板快照，不触发同步。"""
    conn = connect_sqlite(REC_DB_PATH, row_factory=sqlite3.Row)
    try:
        rows = conn.execute("""
            SELECT id, fund_code, trade_date, amount, status, confirm_nav_date, confirm_nav, shares, created_at
            FROM dca_records
            ORDER BY trade_date DESC, id DESC
        """).fetchall()
    finally:
        conn.close()

    if not rows:
        return _empty_dca_snapshot()

    fund_codes = sorted({row["fund_code"] for row in rows})
    latest_nav_map = _get_latest_nav_snapshot(fund_codes)
    fund_name_map = _get_fund_name_map(fund_codes)

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
            "fund_name": fund_name_map.get(row["fund_code"]),
            "start_date": row["trade_date"],
            "confirmed_amount": 0.0,
            "pending_amount": 0.0,
            "total_shares": 0.0,
            "record_count": 0,
            "pending_count": 0,
        })
        if fund.get("fund_name") is None:
            fund["fund_name"] = fund_name_map.get(row["fund_code"])
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
            "fund_name": fund.get("fund_name"),
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


def _cache_dca_snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cached_payload = _clone_dca_snapshot_payload(payload) or _empty_dca_snapshot()
    with _dca_snapshot_lock:
        _dca_snapshot_state["payload"] = cached_payload
        _dca_snapshot_state["generated_at"] = datetime.now()
    return _clone_dca_snapshot_payload(cached_payload) or _empty_dca_snapshot()


def _get_or_build_dca_snapshot_payload() -> dict[str, Any]:
    with _dca_snapshot_lock:
        cached_payload = _clone_dca_snapshot_payload(_dca_snapshot_state["payload"])
    if cached_payload is not None:
        return cached_payload
    payload = _build_dca_snapshot_from_db()
    return _cache_dca_snapshot_payload(payload)


def _run_dca_snapshot_sync():
    global _dca_snapshot_thread

    try:
        _sync_dca_records()
        payload = _build_dca_snapshot_from_db()
        with _dca_snapshot_lock:
            completed_at = datetime.now()
            _dca_snapshot_state["payload"] = _clone_dca_snapshot_payload(payload) or _empty_dca_snapshot()
            _dca_snapshot_state["generated_at"] = completed_at
            _dca_snapshot_state["last_sync_completed_at"] = completed_at
            _dca_snapshot_state["last_sync_error"] = None
            _dca_snapshot_state["dirty"] = False
    except Exception as error:
        logger.exception("dca_snapshot_sync_failed error=%s", error)
        with _dca_snapshot_lock:
            _dca_snapshot_state["last_sync_error"] = str(error)
    finally:
        with _dca_snapshot_lock:
            _dca_snapshot_state["is_syncing"] = False
            _dca_snapshot_thread = None


def _start_dca_snapshot_sync_thread():
    global _dca_snapshot_thread
    thread = threading.Thread(target=_run_dca_snapshot_sync, name="dca-snapshot-sync", daemon=True)
    _dca_snapshot_thread = thread
    thread.start()


def _schedule_dca_snapshot_sync_if_needed() -> bool:
    if app.config.get("TESTING"):
        return False

    now = datetime.now()
    with _dca_snapshot_lock:
        payload = _dca_snapshot_state["payload"]
        generated_at = _dca_snapshot_state["generated_at"]
        is_syncing = bool(_dca_snapshot_state["is_syncing"])
        dirty = bool(_dca_snapshot_state["dirty"])

        is_stale = payload is None or generated_at is None or dirty
        if not is_stale and now - generated_at > DCA_SNAPSHOT_STALE_TTL:
            is_stale = True
        if is_syncing or not is_stale:
            return False

        _dca_snapshot_state["is_syncing"] = True
        _dca_snapshot_state["last_sync_started_at"] = now
        _dca_snapshot_state["last_sync_error"] = None

    _start_dca_snapshot_sync_thread()
    return True


def _get_latest_signal_date(fund_code: str, strategy: str = "v6") -> str:
    """查询DB中该基金指定策略最新的买卖信号日期，用于增量计算。"""
    conn = connect_sqlite(REC_DB_PATH)
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
    conn = connect_sqlite(REC_DB_PATH)
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
    def _action():
        conn = connect_sqlite(REC_DB_PATH)
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

    run_sqlite_write_with_retry(_action, logger=logger, action_name="auto_save_reviews")


def _normalize_nav_estimator_codes(raw_codes) -> list[str]:
    """规范化净值估算请求中的基金代码。"""
    if isinstance(raw_codes, str):
        candidates = re.split(r"[\s,，;；]+", raw_codes.strip())
    elif isinstance(raw_codes, (list, tuple, set)):
        candidates = [str(item).strip() for item in raw_codes]
    else:
        raise ValueError("fund_codes 必须是字符串或数组")

    normalized_codes = []
    seen = set()
    for code in candidates:
        if not code:
            continue
        if not FUND_CODE_PATTERN.fullmatch(code):
            raise ValueError(f"基金代码格式非法: {code}")
        if code in seen:
            continue
        seen.add(code)
        normalized_codes.append(code)

    if not normalized_codes:
        raise ValueError("请至少提供一个6位基金代码")
    return normalized_codes


def _parse_nav_estimator_strict(raw_value) -> bool:
    """解析严格模式字段。"""
    if raw_value is None:
        return True
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError("strict 必须是布尔值")


def _validate_nav_estimator_payload(data: dict) -> tuple[list[str], bool]:
    """校验净值估算请求体。"""
    if not isinstance(data, dict):
        raise ValueError("请求体必须是 JSON 对象")

    fund_codes = _normalize_nav_estimator_codes(data.get("fund_codes"))
    strict = _parse_nav_estimator_strict(data.get("strict"))
    return fund_codes, strict


def _to_json_safe(value):
    """将 DataFrame 结果递归转换为可 JSON 序列化的结构。"""
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_to_json_safe(item) for item in value.to_dict(orient="records")]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _to_json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _build_nav_estimator_response(results_df: pd.DataFrame, strict: bool) -> dict:
    """将估值 DataFrame 转换为 API 响应。"""
    if results_df is None or results_df.empty:
        return {
            "summary": {
                "requested": 0,
                "succeeded": 0,
                "failed": 0,
                "strict_mode": strict,
            },
            "results": [],
        }

    records = [_to_json_safe(item) for item in results_df.to_dict(orient="records")]
    results = []
    succeeded = 0
    failed = 0
    for record in records:
        status = record.get("status")
        warnings = record.get("warnings") or []
        if status == "失败":
            failed += 1
        else:
            succeeded += 1

        summary = {key: record.get(key) for key in NAV_ESTIMATOR_SUMMARY_KEYS}
        summary["warnings"] = warnings
        summary["details"] = {
            key: value
            for key, value in record.items()
            if key not in NAV_ESTIMATOR_SUMMARY_KEYS
        }
        results.append(summary)

    return {
        "summary": {
            "requested": len(records),
            "succeeded": succeeded,
            "failed": failed,
            "strict_mode": strict,
        },
        "results": results,
    }


def _estimate_navs(fund_codes: list[str], strict: bool) -> dict:
    """执行净值估算并格式化响应。"""
    engine = NAVEngine(strict=strict)
    results_df = engine.run(fund_codes, strict=strict)
    return _build_nav_estimator_response(results_df, strict)


def _friendly_nav_estimator_error_message(error: Exception) -> str:
    """将底层异常映射为更可操作的 API 错误提示。"""
    message = str(error)
    proxy_signatures = [
        "ProxyError",
        "Unable to connect to proxy",
        "Cannot connect to proxy",
    ]
    if any(signature in message for signature in proxy_signatures):
        return (
            f"净值估算失败: {message}。检测到代理连接失败，请检查系统代理配置，"
            "或设置环境变量 OTC_FUND_QUANT_PROXY_MODE=direct 后重试。"
        )

    if "质量门槛未通过" in message:
        return f"净值估算失败: {message}"

    if "资产配置数据" in message or "股票仓位失败" in message or "股票总仓位失败" in message:
        return f"净值估算失败: {message}。若该基金为主动权益或主动QDII基金，可关闭严格模式后重试。"

    return f"净值估算失败: {message}"


def _validate_dca_estimates_payload(data: dict) -> list[str]:
    """校验定投看板实时估值请求体。"""
    if not isinstance(data, dict):
        raise ValueError("请求体必须是 JSON 对象")

    raw_codes = data.get("fund_codes", [])
    if raw_codes is None:
        return []
    if not isinstance(raw_codes, (list, tuple, set)):
        raise ValueError("fund_codes 必须是数组")

    normalized_codes = []
    seen = set()
    for raw_code in raw_codes:
        code = str(raw_code).strip()
        if not code:
            continue
        if not FUND_CODE_PATTERN.fullmatch(code):
            raise ValueError(f"基金代码格式非法: {code}")
        if code in seen:
            continue
        seen.add(code)
        normalized_codes.append(code)
    return normalized_codes


def _build_dca_estimates_response(fund_codes: list[str], strict: bool = True) -> dict:
    """将净值估算结果收敛为定投卡片专用结构。"""
    if not fund_codes:
        return {"strict_mode": strict, "results": []}

    raw_payload = _estimate_navs(fund_codes, strict)
    raw_results = raw_payload.get("results", [])

    results = []
    for item in raw_results:
        is_failed = item.get("status") == "失败"
        results.append({
            "fund_code": item.get("fund_code"),
            "status": "failed" if is_failed else "success",
            "estimated_nav": item.get("estimated_nav"),
            "estimated_return": item.get("estimated_return"),
            "nav_date": item.get("nav_date"),
            "message": "严格模式暂不可用" if is_failed else None,
            "warnings": item.get("warnings") or [],
        })

    return {
        "strict_mode": strict,
        "results": results,
    }


def _get_dev_server_run_options() -> dict[str, Any]:
    """返回本地开发服务启动参数。"""
    exclude_patterns = (
        "*/tests/*",
        "*\\tests\\*",
        "*/__pycache__/*",
        "*\\__pycache__\\*",
        "*/.pytest_cache/*",
        "*\\.pytest_cache\\*",
        "*.pyc",
    )
    return {
        "debug": True,
        "host": "127.0.0.1",
        "port": 5000,
        "use_reloader": True,
        "exclude_patterns": exclude_patterns,
    }


# ===== 路由 =====

@app.route("/")
def index():
    """主页面。"""
    return render_template("index.html")


@app.route("/dca")
def dca_dashboard():
    """定投记录与收益看板。"""
    return render_template("dca.html")


@app.route("/nav-estimator")
def nav_estimator_dashboard():
    """基金实时净值估算页面。"""
    return render_template("nav_estimator.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """分析基金：拉取最新数据 + 计算指标 + 生成建议。"""
    data = request.get_json(silent=True) or {}
    fund_code = str(data.get("fund_code", "")).strip()
    requested_strategy = str(data.get("strategy", "") or "").strip()
    if not fund_code:
        return jsonify({"error": "请输入基金代码"}), 400

    start = perf_counter()
    logger.info(
        "analyze_start fund_code=%s requested_strategy=%s",
        fund_code,
        requested_strategy or "AUTO",
    )

    try:
        # 0. 加载策略配置参数
        try:
            strategy_context = resolve_strategy_context(fund_code, requested_strategy)
        except Exception as exc:
            logger.exception(
                "analyze_strategy_context_failed fund_code=%s requested_strategy=%s error=%s",
                fund_code,
                requested_strategy or "AUTO",
                exc,
            )
            return jsonify({"error": f"策略配置解析失败: {str(exc)}"}), 500
        strategy = strategy_context.effective_strategy
        strategy_params = strategy_context.strategy_params

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
        strategy_definition = get_strategy_definition(strategy)
        recommendation = validate_recommendation_payload(
            strategy_definition.generator(history_df, params=strategy_params),
            strategy,
        )

        # 5. 获取图表数据
        chart_data = get_chart_data(history_df, days=500)

        # 5.5 增量计算买卖信号：只计算DB中尚未记录的新日期
        last_signal_date = _get_latest_signal_date(fund_code, strategy)
        new_signals = generate_signal_points(history_df, after_date=last_signal_date, strategy=strategy, params=strategy_params)

        # 6. 将新增信号 + 今日建议写入数据库
        _auto_save_reviews(fund_code, new_signals["records"], recommendation, latest_nav, latest_date, strategy=strategy)

        # 7. 从DB读取完整信号点用于图表展示
        signal_points = _get_all_signal_points(fund_code, strategy)

        # 8. 提取最近一年真实回测成交点与成交记录
        trade_payload = get_backtest_trades(
            history_df,
            strategy=strategy,
            params=strategy_params,
            days=365,
        )

        # 8.5 获取历史推荐记录（保留用于推荐审计）
        rec_history = _get_recommendations(fund_code, strategy)

        normalized_indicators = _normalize_indicator_payload(recommendation.get("indicators"))

        # 9. 计算各周期策略回测收益（优先读缓存）
        cache_end_date = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date)
        period_returns_params_hash = _stable_backtest_params_hash(strategy_params)
        period_returns_payload = _ensure_period_returns(
            fund_code,
            strategy,
            period_returns_params_hash,
            cache_end_date,
            history_df,
            strategy_params,
        )

        resp = {
            "fund_code": fund_code,
            "fund_name": strategy_context.fund_name,
            "latest_nav": latest_nav,
            "latest_date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
            "data_count": len(history_df),
            "strategy": strategy,
            "strategy_params": strategy_params,
            "strategy_context": strategy_context.to_strategy_context_payload(),
            "period_returns": period_returns_payload["period_returns"],
            "period_returns_status": period_returns_payload["status"],
            "period_returns_error": period_returns_payload["error"],
            "period_returns_end_date": cache_end_date,
            "period_returns_params_hash": period_returns_params_hash,
            "recommendation": {
                "action": recommendation["action"],
                "reason": recommendation["reason"],
                "buy_score": recommendation["buy_score"],
                "sell_signal": recommendation["sell_signal"],
                "indicators": normalized_indicators,
            },
            "chart": chart_data,
            "trade_points": trade_payload["trade_points"],
            "trade_history": trade_payload["trade_history"],
            "signal_points": signal_points,
            "rec_history": rec_history,
        }

        # 新策略额外返回 regime_info
        if strategy == "regime_adaptive" and "regime_info" in recommendation:
            resp["recommendation"]["regime_info"] = recommendation["regime_info"]

        return jsonify(resp)

    except Exception as e:
        import traceback
        logger.exception(
            "analyze_failed fund_code=%s requested_strategy=%s elapsed=%.2fs error=%s",
            fund_code,
            requested_strategy or "AUTO",
            perf_counter() - start,
            e,
        )
        traceback.print_exc()
        return jsonify({"error": f"分析失败: {str(e)}"}), 500


@app.route("/api/period-returns")
def api_period_returns():
    """查询区间收益后台计算状态。"""
    fund_code = str(request.args.get("fund_code", "")).strip()
    strategy = str(request.args.get("strategy", "")).strip()
    end_date = str(request.args.get("end_date", "")).strip()
    params_hash = _resolve_period_returns_params_hash(
        fund_code,
        strategy,
        str(request.args.get("params_hash", "")).strip(),
    )

    if not fund_code:
        return jsonify({"error": "请输入基金代码"}), 400
    if not strategy:
        return jsonify({"error": "缺少策略参数"}), 400
    if not end_date:
        return jsonify({"error": "缺少 end_date 参数"}), 400

    payload = _get_period_returns_payload(fund_code, strategy, params_hash, end_date)
    return jsonify({
        "fund_code": fund_code,
        "strategy": strategy,
        "end_date": end_date,
        "params_hash": params_hash,
        "status": payload["status"],
        "period_returns": payload["period_returns"],
        "error": payload["error"],
    })


@app.route("/api/nav-estimator/estimate", methods=["POST"])
def api_nav_estimator_estimate():
    """执行基金实时净值估算。"""
    data = request.get_json(silent=True)
    if data is None:
        data = {}

    try:
        fund_codes, strict = _validate_nav_estimator_payload(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        return jsonify(_estimate_navs(fund_codes, strict))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": _friendly_nav_estimator_error_message(e)}), 500


@app.route("/api/dca/portfolio")
def api_dca_portfolio():
    """获取定投看板汇总与明细。"""
    try:
        payload = _get_or_build_dca_snapshot_payload()
        _schedule_dca_snapshot_sync_if_needed()
        return jsonify(_attach_dca_sync_meta(payload))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"读取定投看板失败: {str(e)}"}), 500


@app.route("/api/dca/estimates", methods=["POST"])
def api_dca_estimates():
    """获取定投看板卡片使用的实时净值估算结果。"""
    start = perf_counter()
    data = request.get_json(silent=True)
    if data is None:
        data = {}

    try:
        fund_codes = _validate_dca_estimates_payload(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    logger.info("dca_estimates_start fund_count=%s", len(fund_codes))
    try:
        payload = _build_dca_estimates_response(fund_codes, strict=True)
        logger.info(
            "dca_estimates_done fund_count=%s result_count=%s elapsed=%.2fs",
            len(fund_codes),
            len(payload.get("results", [])),
            perf_counter() - start,
        )
        return jsonify(payload)
    except Exception as e:
        logger.exception(
            "dca_estimates_failed fund_count=%s elapsed=%.2fs error=%s",
            len(fund_codes),
            perf_counter() - start,
            e,
        )
        return jsonify({"error": _friendly_nav_estimator_error_message(e)}), 500


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

    record_holder: dict[str, int | None] = {"id": None}

    def _action():
        conn = connect_sqlite(REC_DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO dca_records (fund_code, trade_date, amount, status)
                VALUES (?, ?, ?, ?)
            """, (fund_code, trade_date_str, amount, DCA_STATUS_PENDING))
            record_holder["id"] = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()

    run_sqlite_write_with_retry(_action, logger=logger, action_name="create_dca_record")
    record_id = record_holder["id"]
    _mark_dca_snapshot_dirty(invalidate_payload=True)

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
    deleted_holder: dict[str, int] = {"count": 0}

    def _action():
        conn = connect_sqlite(REC_DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dca_records WHERE id = ?", (record_id,))
            deleted_holder["count"] = cursor.rowcount
            conn.commit()
        finally:
            conn.close()

    run_sqlite_write_with_retry(_action, logger=logger, action_name="delete_dca_record")
    deleted = deleted_holder["count"]

    if deleted == 0:
        return jsonify({"error": "记录不存在"}), 404

    _mark_dca_snapshot_dirty(invalidate_payload=True)
    return jsonify({"success": True, "id": record_id})


@app.route("/api/recommendations/<fund_code>")
def api_get_recommendations(fund_code: str):
    """获取历史推荐记录。"""
    records = _get_recommendations(fund_code)
    return jsonify(records)


def _get_recommendations(fund_code: str, strategy: str = "v6"):
    """从数据库读取指定策略的推荐历史。"""
    conn = connect_sqlite(REC_DB_PATH)
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
    app.run(**_get_dev_server_run_options())
