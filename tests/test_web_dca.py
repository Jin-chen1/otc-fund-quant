import importlib
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


sys.modules.setdefault("akshare", MagicMock())

PROJECT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)


class DcaWebTestCase(unittest.TestCase):
    def setUp(self):
        self._old_nav_db_path = os.environ.get("OTC_FUND_QUANT_NAV_DB_PATH")
        self._old_rec_db_path = os.environ.get("OTC_FUND_QUANT_REC_DB_PATH")

        nav_db = tempfile.NamedTemporaryFile(prefix="tmp_dca_nav_", suffix=".sqlite", delete=False)
        rec_db = tempfile.NamedTemporaryFile(prefix="tmp_dca_rec_", suffix=".sqlite", delete=False)
        nav_db.close()
        rec_db.close()

        self.nav_db_path = nav_db.name
        self.rec_db_path = rec_db.name
        os.environ["OTC_FUND_QUANT_NAV_DB_PATH"] = self.nav_db_path
        os.environ["OTC_FUND_QUANT_REC_DB_PATH"] = self.rec_db_path

        if "otc_fund_quant.web.app" in sys.modules:
            self.web_app = importlib.reload(sys.modules["otc_fund_quant.web.app"])
        else:
            self.web_app = importlib.import_module("otc_fund_quant.web.app")

        self.web_app.app.config["TESTING"] = True
        self.client = self.web_app.app.test_client()

    def tearDown(self):
        if self._old_nav_db_path is None:
            os.environ.pop("OTC_FUND_QUANT_NAV_DB_PATH", None)
        else:
            os.environ["OTC_FUND_QUANT_NAV_DB_PATH"] = self._old_nav_db_path

        if self._old_rec_db_path is None:
            os.environ.pop("OTC_FUND_QUANT_REC_DB_PATH", None)
        else:
            os.environ["OTC_FUND_QUANT_REC_DB_PATH"] = self._old_rec_db_path

        for path in (self.nav_db_path, self.rec_db_path):
            if os.path.exists(path):
                os.remove(path)

    def _seed_nav_rows(self, rows):
        conn = sqlite3.connect(self.nav_db_path)
        conn.executemany(
            """
            INSERT INTO fund_nav (date, fund_code, nav, acc_nav)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        conn.close()

    def _insert_dca_record(
        self,
        fund_code,
        trade_date,
        amount,
        *,
        trade_type=None,
        sell_input_mode=None,
        requested_shares=None,
        status,
        status_reason=None,
        confirm_nav_date=None,
        confirm_nav=None,
        shares=None,
        deficit_shares_remaining=0.0,
        offset_shares=0.0,
        fee=None,
        net_cash=None,
        realized_pnl=None,
    ):
        trade_type = trade_type or self.web_app.DCA_TRADE_TYPE_BUY
        conn = sqlite3.connect(self.rec_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO dca_records (
                fund_code, trade_date, trade_type, sell_input_mode, amount, requested_shares,
                status, status_reason, confirm_nav_date, confirm_nav, shares,
                deficit_shares_remaining, offset_shares, fee, net_cash,
                realized_pnl, confirmed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CASE WHEN ? = 'confirmed' THEN CURRENT_TIMESTAMP ELSE NULL END)
            """,
            (
                fund_code,
                trade_date,
                trade_type,
                sell_input_mode,
                amount,
                requested_shares,
                status,
                status_reason,
                confirm_nav_date,
                confirm_nav,
                shares,
                deficit_shares_remaining,
                offset_shares,
                fee,
                net_cash,
                realized_pnl,
                status,
            ),
        )
        conn.commit()
        conn.close()

    def test_portfolio_includes_fund_name_when_fetch_succeeds(self):
        self._seed_nav_rows([
            ("2026-03-06", "000001", 1.5000, 1.5000),
        ])
        self._insert_dca_record(
            "000001",
            "2026-03-05",
            150.0,
            status=self.web_app.DCA_STATUS_CONFIRMED,
            confirm_nav_date="2026-03-06",
            confirm_nav=1.5,
            shares=100.0,
        )

        with patch.object(self.web_app.loader, "update_db"), patch.object(
            self.web_app.dca_fund_fetcher,
            "get_fund_info",
            return_value={"name": "测试基金A"},
        ) as mock_get_fund_info:
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["funds"][0]["fund_name"], "测试基金A")
        mock_get_fund_info.assert_called_once_with("000001")

    def test_portfolio_keeps_loading_when_fund_name_fetch_fails(self):
        self._seed_nav_rows([
            ("2026-03-06", "000001", 1.5000, 1.5000),
        ])
        self._insert_dca_record(
            "000001",
            "2026-03-05",
            150.0,
            status=self.web_app.DCA_STATUS_CONFIRMED,
            confirm_nav_date="2026-03-06",
            confirm_nav=1.5,
            shares=100.0,
        )

        with patch.object(self.web_app.loader, "update_db"), patch.object(
            self.web_app.dca_fund_fetcher,
            "get_fund_info",
            side_effect=ValueError("name lookup failed"),
        ):
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNone(payload["funds"][0]["fund_name"])
        self.assertAlmostEqual(payload["funds"][0]["current_value"], 150.0)

    def test_portfolio_aggregates_multiple_records_under_one_fund_name(self):
        self._seed_nav_rows([
            ("2026-03-06", "000001", 1.2000, 1.2000),
            ("2026-03-07", "000001", 1.3000, 1.3000),
        ])
        self._insert_dca_record(
            "000001",
            "2026-03-05",
            120.0,
            status=self.web_app.DCA_STATUS_CONFIRMED,
            confirm_nav_date="2026-03-06",
            confirm_nav=1.2,
            shares=100.0,
        )
        self._insert_dca_record(
            "000001",
            "2026-03-06",
            130.0,
            status=self.web_app.DCA_STATUS_CONFIRMED,
            confirm_nav_date="2026-03-07",
            confirm_nav=1.3,
            shares=100.0,
        )

        with patch.object(self.web_app.loader, "update_db"), patch.object(
            self.web_app.dca_fund_fetcher,
            "get_fund_info",
            return_value={"name": "测试基金B"},
        ) as mock_get_fund_info:
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["funds"]), 1)
        self.assertEqual(payload["funds"][0]["fund_name"], "测试基金B")
        self.assertEqual(payload["funds"][0]["record_count"], 2)
        mock_get_fund_info.assert_called_once_with("000001")

    def test_portfolio_accumulates_pending_amount_for_same_fund_same_day(self):
        self._insert_dca_record(
            "018345",
            "2026-03-09",
            200.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )
        self._insert_dca_record(
            "018345",
            "2026-03-09",
            200.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )

        with patch.object(self.web_app.loader, "update_db"), patch.object(
            self.web_app.dca_fund_fetcher,
            "get_fund_info",
            return_value={"name": "测试基金C"},
        ):
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["funds"]), 1)
        self.assertEqual(payload["funds"][0]["fund_code"], "018345")
        self.assertEqual(payload["funds"][0]["pending_count"], 2)
        self.assertEqual(payload["funds"][0]["record_count"], 2)
        self.assertAlmostEqual(payload["funds"][0]["pending_amount"], 400.0)
        self.assertAlmostEqual(payload["portfolio"]["total_pending_amount"], 400.0)

    def test_dca_estimates_returns_card_friendly_success_payload(self):
        with patch.object(
            self.web_app,
            "_estimate_navs",
            return_value={
                "summary": {"strict_mode": True},
                "results": [
                    {
                        "fund_code": "000001",
                        "status": "成功",
                        "estimated_nav": 1.2345,
                        "estimated_return": 0.67,
                        "nav_date": "2026-03-09",
                        "warnings": [],
                    }
                ],
            },
        ) as mock_estimate_navs:
            response = self.client.post("/api/dca/estimates", json={"fund_codes": ["000001"]})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["strict_mode"])
        self.assertEqual(
            payload["results"],
            [
                {
                    "fund_code": "000001",
                    "status": "success",
                    "estimated_nav": 1.2345,
                    "estimated_return": 0.67,
                    "nav_date": "2026-03-09",
                    "message": None,
                    "warnings": [],
                }
            ],
        )
        mock_estimate_navs.assert_called_once_with(["000001"], True)

    def test_dev_server_run_options_keep_reloader_but_exclude_test_noise(self):
        options = self.web_app._get_dev_server_run_options()

        self.assertTrue(options["debug"])
        self.assertTrue(options["use_reloader"])
        self.assertEqual(options["host"], "127.0.0.1")
        self.assertEqual(options["port"], 5000)
        self.assertIn("*/tests/*", options["exclude_patterns"])
        self.assertIn("*/__pycache__/*", options["exclude_patterns"])
        self.assertIn("*/.pytest_cache/*", options["exclude_patterns"])
        self.assertIn("*.pyc", options["exclude_patterns"])

    def test_dca_estimates_keeps_order_and_single_fund_failure(self):
        with patch.object(
            self.web_app,
            "_estimate_navs",
            return_value={
                "summary": {"strict_mode": True},
                "results": [
                    {
                        "fund_code": "000002",
                        "status": "失败",
                        "estimated_nav": None,
                        "estimated_return": None,
                        "nav_date": None,
                        "warnings": ["质量门槛未通过"],
                    },
                    {
                        "fund_code": "000001",
                        "status": "成功",
                        "estimated_nav": 1.1111,
                        "estimated_return": -0.12,
                        "nav_date": "2026-03-09",
                        "warnings": [],
                    },
                ],
            },
        ):
            response = self.client.post("/api/dca/estimates", json={"fund_codes": ["000002", "000001"]})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(
            payload["results"],
            [
                {
                    "fund_code": "000002",
                    "status": "failed",
                    "estimated_nav": None,
                    "estimated_return": None,
                    "nav_date": None,
                    "message": "严格模式暂不可用",
                    "warnings": ["质量门槛未通过"],
                },
                {
                    "fund_code": "000001",
                    "status": "success",
                    "estimated_nav": 1.1111,
                    "estimated_return": -0.12,
                    "nav_date": "2026-03-09",
                    "message": None,
                    "warnings": [],
                },
            ],
        )

    def test_dca_estimates_empty_codes_skips_engine(self):
        with patch.object(self.web_app, "_estimate_navs", side_effect=AssertionError("should not estimate")):
            response = self.client.post("/api/dca/estimates", json={"fund_codes": []})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["strict_mode"])
        self.assertEqual(payload["results"], [])

    def test_dca_estimates_returns_friendly_error_when_request_fails(self):
        with patch.object(self.web_app, "_estimate_navs", side_effect=RuntimeError("boom")):
            response = self.client.post("/api/dca/estimates", json={"fund_codes": ["000001"]})

        self.assertEqual(response.status_code, 500)
        payload = response.get_json()
        self.assertEqual(payload["error"], "净值估算失败: boom")

    def test_dca_estimates_mixed_batch_keeps_200_and_card_states(self):
        with patch.object(
            self.web_app,
            "_estimate_navs",
            return_value={
                "summary": {"strict_mode": True},
                "results": [
                    {"fund_code": "015916", "status": "成功", "estimated_nav": 1.01, "estimated_return": -1.4, "nav_date": "2026-03-10", "warnings": []},
                    {"fund_code": "018291", "status": "成功", "estimated_nav": 1.02, "estimated_return": -3.1, "nav_date": "2026-03-10", "warnings": []},
                    {"fund_code": "018345", "status": "失败", "estimated_nav": None, "estimated_return": None, "nav_date": None, "warnings": []},
                    {"fund_code": "020989", "status": "成功", "estimated_nav": 1.03, "estimated_return": 0.02, "nav_date": "2026-03-10", "warnings": []},
                    {"fund_code": "023639", "status": "失败", "estimated_nav": None, "estimated_return": None, "nav_date": None, "warnings": []},
                ],
            },
        ):
            response = self.client.post(
                "/api/dca/estimates",
                json={"fund_codes": ["015916", "018291", "018345", "020989", "023639"]},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual([item["fund_code"] for item in payload["results"]], ["015916", "018291", "018345", "020989", "023639"])
        self.assertEqual([item["status"] for item in payload["results"]], ["success", "success", "failed", "success", "failed"])
        self.assertEqual(payload["results"][2]["message"], "严格模式暂不可用")
        self.assertEqual(payload["results"][4]["message"], "严格模式暂不可用")

    def test_portfolio_response_includes_sync_meta_and_skips_inline_sync(self):
        self._seed_nav_rows([
            ("2026-03-11", "000001", 1.5000, 1.5000),
        ])
        self._insert_dca_record(
            "000001",
            "2026-03-10",
            150.0,
            status=self.web_app.DCA_STATUS_CONFIRMED,
            confirm_nav_date="2026-03-11",
            confirm_nav=1.5,
            shares=100.0,
        )

        with patch.object(self.web_app, "_sync_dca_records", side_effect=AssertionError("should not sync inline")), \
             patch.object(self.web_app, "_schedule_dca_snapshot_sync_if_needed", return_value=False), \
             patch.object(self.web_app.dca_fund_fetcher, "get_fund_info", return_value={"name": "测试基金A"}):
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["funds"][0]["fund_name"], "测试基金A")
        self.assertIn("sync", payload)
        self.assertFalse(payload["sync"]["is_syncing"])
        self.assertTrue(payload["sync"]["is_stale"])
        self.assertIsNone(payload["sync"]["last_sync_completed_at"])
        self.assertIsNone(payload["sync"]["last_sync_error"])

    def test_portfolio_returns_cached_snapshot_when_fresh(self):
        cached_payload = {
            "as_of_date": "2026-03-12",
            "portfolio": {
                "tracked_fund_count": 1,
                "total_confirmed_amount": 100.0,
                "total_pending_amount": 0.0,
                "total_value": 101.0,
                "total_profit_amount": 1.0,
                "total_profit_pct": 1.0,
            },
            "funds": [
                {
                    "fund_code": "000001",
                    "fund_name": "缓存基金",
                    "start_date": "2026-03-10",
                    "latest_nav": 1.01,
                    "latest_nav_date": "2026-03-12",
                    "confirmed_amount": 100.0,
                    "pending_amount": 0.0,
                    "total_shares": 100.0,
                    "current_value": 101.0,
                    "profit_amount": 1.0,
                    "profit_pct": 1.0,
                    "record_count": 1,
                    "pending_count": 0,
                }
            ],
            "records": [],
        }
        with self.web_app._dca_snapshot_lock:
            self.web_app._dca_snapshot_state["payload"] = cached_payload
            self.web_app._dca_snapshot_state["generated_at"] = self.web_app.datetime.now()
            self.web_app._dca_snapshot_state["last_sync_started_at"] = None
            self.web_app._dca_snapshot_state["last_sync_completed_at"] = self.web_app.datetime.now()
            self.web_app._dca_snapshot_state["last_sync_error"] = None
            self.web_app._dca_snapshot_state["is_syncing"] = False
            self.web_app._dca_snapshot_state["dirty"] = False

        with patch.object(self.web_app, "_build_dca_snapshot_from_db", side_effect=AssertionError("should not rebuild")):
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["funds"][0]["fund_name"], "缓存基金")
        self.assertFalse(payload["sync"]["is_stale"])
        self.assertFalse(payload["sync"]["is_syncing"])

    def test_schedule_dca_snapshot_sync_if_needed_starts_thread_when_dirty(self):
        original_testing = self.web_app.app.config["TESTING"]
        self.web_app.app.config["TESTING"] = False
        try:
            with self.web_app._dca_snapshot_lock:
                self.web_app._dca_snapshot_state["payload"] = {"as_of_date": "2026-03-12", "portfolio": {}, "funds": [], "records": []}
                self.web_app._dca_snapshot_state["generated_at"] = self.web_app.datetime.now()
                self.web_app._dca_snapshot_state["last_sync_started_at"] = None
                self.web_app._dca_snapshot_state["last_sync_completed_at"] = None
                self.web_app._dca_snapshot_state["last_sync_error"] = None
                self.web_app._dca_snapshot_state["is_syncing"] = False
                self.web_app._dca_snapshot_state["dirty"] = True

            with patch.object(self.web_app, "_start_dca_snapshot_sync_thread") as mock_start:
                scheduled = self.web_app._schedule_dca_snapshot_sync_if_needed()

            self.assertTrue(scheduled)
            mock_start.assert_called_once_with()
            with self.web_app._dca_snapshot_lock:
                self.assertTrue(self.web_app._dca_snapshot_state["is_syncing"])
                self.assertIsNotNone(self.web_app._dca_snapshot_state["last_sync_started_at"])
                self.assertIsNone(self.web_app._dca_snapshot_state["last_sync_error"])
        finally:
            self.web_app.app.config["TESTING"] = original_testing

    def test_create_record_invalidates_dca_snapshot_cache(self):
        with self.web_app._dca_snapshot_lock:
            self.web_app._dca_snapshot_state["payload"] = {"as_of_date": "2026-03-12", "portfolio": {}, "funds": [], "records": []}
            self.web_app._dca_snapshot_state["generated_at"] = self.web_app.datetime.now()
            self.web_app._dca_snapshot_state["dirty"] = False

        response = self.client.post(
            "/api/dca/records",
            json={"fund_code": "000001", "trade_date": "2026-03-12", "amount": 100.0},
        )

        self.assertEqual(response.status_code, 201)
        with self.web_app._dca_snapshot_lock:
            self.assertIsNone(self.web_app._dca_snapshot_state["payload"])
            self.assertIsNone(self.web_app._dca_snapshot_state["generated_at"])
            self.assertTrue(self.web_app._dca_snapshot_state["dirty"])

    def test_delete_record_invalidates_dca_snapshot_cache(self):
        self._insert_dca_record(
            "000001",
            "2026-03-12",
            100.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )
        with self.web_app._dca_snapshot_lock:
            self.web_app._dca_snapshot_state["payload"] = {"as_of_date": "2026-03-12", "portfolio": {}, "funds": [], "records": []}
            self.web_app._dca_snapshot_state["generated_at"] = self.web_app.datetime.now()
            self.web_app._dca_snapshot_state["dirty"] = False

        response = self.client.delete("/api/dca/records/1")

        self.assertEqual(response.status_code, 200)
        with self.web_app._dca_snapshot_lock:
            self.assertIsNone(self.web_app._dca_snapshot_state["payload"])
            self.assertIsNone(self.web_app._dca_snapshot_state["generated_at"])
            self.assertTrue(self.web_app._dca_snapshot_state["dirty"])

    def test_sync_dca_records_confirms_pending_record_after_loader_update(self):
        self._insert_dca_record(
            "000001",
            "2026-03-12",
            125.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )

        def _write_latest_nav(_fund_code):
            self._seed_nav_rows([
                ("2026-03-13", "000001", 1.25, 1.25),
            ])

        with patch.object(self.web_app.loader, "update_db", side_effect=_write_latest_nav) as mock_update_db:
            self.web_app._sync_dca_records()

        self.assertEqual(mock_update_db.call_count, 1)
        self.assertEqual(mock_update_db.call_args.args[0], "000001")

        conn = sqlite3.connect(self.rec_db_path)
        row = conn.execute(
            """
            SELECT status, confirm_nav_date, confirm_nav, shares
            FROM dca_records
            WHERE fund_code = ?
            """,
            ("000001",),
        ).fetchone()
        conn.close()

        self.assertEqual(row[0], self.web_app.DCA_STATUS_CONFIRMED)
        self.assertEqual(row[1], "2026-03-13")
        self.assertAlmostEqual(row[2], 1.25)
        self.assertAlmostEqual(row[3], 100.0)

    def test_dca_records_migrates_legacy_buy_only_schema(self):
        conn = sqlite3.connect(self.rec_db_path)
        conn.execute("DROP TABLE IF EXISTS dca_records")
        conn.execute("""
            CREATE TABLE dca_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fund_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                confirm_nav_date TEXT,
                confirm_nav REAL,
                shares REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                confirmed_at TEXT
            )
        """)
        conn.execute("""
            INSERT INTO dca_records (
                fund_code, trade_date, amount, status, confirm_nav_date, confirm_nav, shares, confirmed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, ("000001", "2026-03-12", 100.0, "confirmed", "2026-03-13", 1.0, 100.0))
        conn.commit()
        conn.close()

        self.web_app = importlib.reload(sys.modules["otc_fund_quant.web.app"])
        self.web_app.app.config["TESTING"] = True
        self.client = self.web_app.app.test_client()

        conn = sqlite3.connect(self.rec_db_path)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(dca_records)").fetchall()}
        row = conn.execute("""
            SELECT trade_type, sell_input_mode, requested_shares, deficit_shares_remaining, offset_shares,
                   status, status_reason, fee, net_cash, realized_pnl
            FROM dca_records
            WHERE fund_code = ?
        """, ("000001",)).fetchone()
        conn.close()

        self.assertIn("trade_type", columns)
        self.assertIn("sell_input_mode", columns)
        self.assertIn("requested_shares", columns)
        self.assertIn("deficit_shares_remaining", columns)
        self.assertIn("offset_shares", columns)
        self.assertIn("status_reason", columns)
        self.assertIn("net_cash", columns)
        self.assertEqual(row[0], self.web_app.DCA_TRADE_TYPE_BUY)
        self.assertIsNone(row[1])
        self.assertIsNone(row[2])
        self.assertEqual(row[3], 0.0)
        self.assertEqual(row[4], 0.0)
        self.assertEqual(row[5], self.web_app.DCA_STATUS_CONFIRMED)
        self.assertIsNone(row[6])
        self.assertIsNone(row[7])
        self.assertIsNone(row[8])
        self.assertIsNone(row[9])

    def test_create_sell_record_accepts_when_no_position_exists(self):
        response = self.client.post(
            "/api/dca/records",
            json={"fund_code": "000001", "trade_type": "sell", "trade_date": "2026-03-12", "amount": 50.0},
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload["trade_type"], self.web_app.DCA_TRADE_TYPE_SELL)
        self.assertEqual(payload["status"], self.web_app.DCA_STATUS_PENDING)

    def test_create_sell_record_by_shares_accepts_when_no_position_exists(self):
        response = self.client.post(
            "/api/dca/records",
            json={
                "fund_code": "000001",
                "trade_type": "sell",
                "sell_input_mode": "shares",
                "trade_date": "2026-03-12",
                "shares": 10.0,
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload["trade_type"], self.web_app.DCA_TRADE_TYPE_SELL)
        self.assertEqual(payload["sell_input_mode"], self.web_app.DCA_SELL_INPUT_MODE_SHARES)
        self.assertEqual(payload["status"], self.web_app.DCA_STATUS_PENDING)

    def test_create_sell_record_accepts_existing_position(self):
        self._seed_nav_rows([
            ("2026-03-06", "000001", 1.5000, 1.5000),
        ])
        self._insert_dca_record(
            "000001",
            "2026-03-05",
            150.0,
            status=self.web_app.DCA_STATUS_CONFIRMED,
            confirm_nav_date="2026-03-06",
            confirm_nav=1.5,
            shares=100.0,
        )

        response = self.client.post(
            "/api/dca/records",
            json={"fund_code": "000001", "trade_type": "sell", "trade_date": "2026-03-07", "amount": 90.0},
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload["trade_type"], self.web_app.DCA_TRADE_TYPE_SELL)
        self.assertEqual(payload["status"], self.web_app.DCA_STATUS_PENDING)

        conn = sqlite3.connect(self.rec_db_path)
        row = conn.execute("""
            SELECT trade_type, status, amount
            FROM dca_records
            WHERE id = ?
        """, (payload["id"],)).fetchone()
        conn.close()

        self.assertEqual(row[0], self.web_app.DCA_TRADE_TYPE_SELL)
        self.assertEqual(row[1], self.web_app.DCA_STATUS_PENDING)
        self.assertAlmostEqual(row[2], 90.0)

    def test_create_sell_record_by_shares_accepts_when_shares_exceed_position(self):
        self._seed_nav_rows([
            ("2026-03-06", "000001", 1.5000, 1.5000),
        ])
        self._insert_dca_record(
            "000001",
            "2026-03-05",
            150.0,
            status=self.web_app.DCA_STATUS_CONFIRMED,
            confirm_nav_date="2026-03-06",
            confirm_nav=1.5,
            shares=100.0,
        )

        response = self.client.post(
            "/api/dca/records",
            json={
                "fund_code": "000001",
                "trade_type": "sell",
                "sell_input_mode": "shares",
                "trade_date": "2026-03-07",
                "shares": 120.0,
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload["trade_type"], self.web_app.DCA_TRADE_TYPE_SELL)
        self.assertEqual(payload["sell_input_mode"], self.web_app.DCA_SELL_INPUT_MODE_SHARES)
        self.assertAlmostEqual(payload["requested_shares"], 120.0)

    def test_create_sell_record_by_shares_accepts_existing_position(self):
        self._seed_nav_rows([
            ("2026-03-06", "000001", 1.5000, 1.5000),
        ])
        self._insert_dca_record(
            "000001",
            "2026-03-05",
            150.0,
            status=self.web_app.DCA_STATUS_CONFIRMED,
            confirm_nav_date="2026-03-06",
            confirm_nav=1.5,
            shares=100.0,
        )

        response = self.client.post(
            "/api/dca/records",
            json={
                "fund_code": "000001",
                "trade_type": "sell",
                "sell_input_mode": "shares",
                "trade_date": "2026-03-07",
                "shares": 40.0,
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload["trade_type"], self.web_app.DCA_TRADE_TYPE_SELL)
        self.assertEqual(payload["sell_input_mode"], self.web_app.DCA_SELL_INPUT_MODE_SHARES)
        self.assertIsNone(payload["amount"])
        self.assertAlmostEqual(payload["requested_shares"], 40.0)
        self.assertAlmostEqual(payload["shares"], 40.0)

        conn = sqlite3.connect(self.rec_db_path)
        row = conn.execute("""
            SELECT trade_type, sell_input_mode, amount, requested_shares, status
            FROM dca_records
            WHERE id = ?
        """, (payload["id"],)).fetchone()
        conn.close()

        self.assertEqual(row[0], self.web_app.DCA_TRADE_TYPE_SELL)
        self.assertEqual(row[1], self.web_app.DCA_SELL_INPUT_MODE_SHARES)
        self.assertIsNone(row[2])
        self.assertAlmostEqual(row[3], 40.0)
        self.assertEqual(row[4], self.web_app.DCA_STATUS_PENDING)

    def test_sync_dca_records_confirms_sell_and_updates_portfolio_summary(self):
        self._insert_dca_record(
            "000001",
            "2026-03-10",
            150.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )
        self._insert_dca_record(
            "000001",
            "2026-03-12",
            90.0,
            trade_type=self.web_app.DCA_TRADE_TYPE_SELL,
            status=self.web_app.DCA_STATUS_PENDING,
        )

        def _write_nav_rows(_fund_code):
            self._seed_nav_rows([
                ("2026-03-11", "000001", 1.5, 1.5),
                ("2026-03-12", "000001", 1.8, 1.8),
            ])

        with patch.object(self.web_app.loader, "update_db", side_effect=_write_nav_rows):
            self.web_app._sync_dca_records()

        conn = sqlite3.connect(self.rec_db_path)
        rows = conn.execute("""
            SELECT trade_type, status, confirm_nav_date, confirm_nav, shares, fee, net_cash, realized_pnl
            FROM dca_records
            ORDER BY id
        """).fetchall()
        conn.close()

        self.assertEqual(rows[0][0], self.web_app.DCA_TRADE_TYPE_BUY)
        self.assertEqual(rows[0][1], self.web_app.DCA_STATUS_CONFIRMED)
        self.assertEqual(rows[1][0], self.web_app.DCA_TRADE_TYPE_SELL)
        self.assertEqual(rows[1][1], self.web_app.DCA_STATUS_CONFIRMED)
        self.assertEqual(rows[1][2], "2026-03-12")
        self.assertAlmostEqual(rows[1][3], 1.8)
        self.assertAlmostEqual(rows[1][4], 50.0)
        self.assertAlmostEqual(rows[1][5], 1.35)
        self.assertAlmostEqual(rows[1][6], 88.65)
        self.assertAlmostEqual(rows[1][7], 13.65)

        with patch.object(self.web_app, "_schedule_dca_snapshot_sync_if_needed", return_value=False), patch.object(
            self.web_app.dca_fund_fetcher,
            "get_fund_info",
            return_value={"name": "测试基金卖出"},
        ):
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        fund = payload["funds"][0]
        self.assertAlmostEqual(fund["remaining_cost_basis"], 75.0)
        self.assertAlmostEqual(fund["total_shares"], 50.0)
        self.assertAlmostEqual(fund["current_value"], 90.0)
        self.assertAlmostEqual(fund["unrealized_profit_amount"], 15.0)
        self.assertAlmostEqual(fund["realized_profit_amount"], 13.65)
        self.assertEqual(payload["records"][0]["trade_type"], self.web_app.DCA_TRADE_TYPE_SELL)
        self.assertAlmostEqual(payload["records"][0]["net_cash"], 88.65)
        self.assertAlmostEqual(payload["records"][0]["realized_pnl"], 13.65)
        self.assertAlmostEqual(payload["portfolio"]["total_remaining_cost_basis"], 75.0)
        self.assertAlmostEqual(payload["portfolio"]["total_realized_profit_amount"], 13.65)

    def test_sync_dca_records_confirms_sell_by_shares_and_backfills_amount(self):
        self._insert_dca_record(
            "000001",
            "2026-03-10",
            150.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )
        self._insert_dca_record(
            "000001",
            "2026-03-12",
            None,
            trade_type=self.web_app.DCA_TRADE_TYPE_SELL,
            sell_input_mode=self.web_app.DCA_SELL_INPUT_MODE_SHARES,
            requested_shares=40.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )

        def _write_nav_rows(_fund_code):
            self._seed_nav_rows([
                ("2026-03-11", "000001", 1.5, 1.5),
                ("2026-03-12", "000001", 1.8, 1.8),
            ])

        with patch.object(self.web_app.loader, "update_db", side_effect=_write_nav_rows):
            self.web_app._sync_dca_records()

        conn = sqlite3.connect(self.rec_db_path)
        row = conn.execute("""
            SELECT sell_input_mode, amount, requested_shares, shares, fee, net_cash, realized_pnl, status
            FROM dca_records
            WHERE id = 2
        """).fetchone()
        conn.close()

        self.assertEqual(row[0], self.web_app.DCA_SELL_INPUT_MODE_SHARES)
        self.assertAlmostEqual(row[1], 72.0)
        self.assertAlmostEqual(row[2], 40.0)
        self.assertAlmostEqual(row[3], 40.0)
        self.assertAlmostEqual(row[4], 1.08)
        self.assertAlmostEqual(row[5], 70.92)
        self.assertAlmostEqual(row[6], 10.92)
        self.assertEqual(row[7], self.web_app.DCA_STATUS_CONFIRMED)

        with patch.object(self.web_app, "_schedule_dca_snapshot_sync_if_needed", return_value=False), patch.object(
            self.web_app.dca_fund_fetcher,
            "get_fund_info",
            return_value={"name": "测试基金份额卖出"},
        ):
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        sell_record = payload["records"][0]
        self.assertEqual(sell_record["sell_input_mode"], self.web_app.DCA_SELL_INPUT_MODE_SHARES)
        self.assertAlmostEqual(sell_record["requested_shares"], 40.0)
        self.assertAlmostEqual(sell_record["amount"], 72.0)
        self.assertAlmostEqual(sell_record["shares"], 40.0)

    def test_sync_dca_records_confirms_oversell_and_tracks_deficit_shares(self):
        self._insert_dca_record(
            "000001",
            "2026-03-10",
            100.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )
        self._insert_dca_record(
            "000001",
            "2026-03-11",
            150.0,
            trade_type=self.web_app.DCA_TRADE_TYPE_SELL,
            status=self.web_app.DCA_STATUS_PENDING,
        )

        def _write_nav_rows(_fund_code):
            self._seed_nav_rows([
                ("2026-03-10", "000001", 1.0, 1.0),
                ("2026-03-11", "000001", 1.0, 1.0),
            ])

        with patch.object(self.web_app.loader, "update_db", side_effect=_write_nav_rows):
            self.web_app._sync_dca_records()

        conn = sqlite3.connect(self.rec_db_path)
        row = conn.execute("""
            SELECT status, status_reason, confirm_nav_date, confirm_nav, shares,
                   deficit_shares_remaining, offset_shares, fee, net_cash, realized_pnl
            FROM dca_records
            WHERE id = 2
        """).fetchone()
        conn.close()

        self.assertEqual(row[0], self.web_app.DCA_STATUS_CONFIRMED)
        self.assertIsNone(row[1])
        self.assertEqual(row[2], "2026-03-11")
        self.assertAlmostEqual(row[3], 1.0)
        self.assertAlmostEqual(row[4], 150.0)
        self.assertAlmostEqual(row[5], 50.0)
        self.assertAlmostEqual(row[6], 0.0)
        self.assertAlmostEqual(row[7], 1.5)
        self.assertAlmostEqual(row[8], 148.5)
        self.assertAlmostEqual(row[9], -1.5)

        with patch.object(self.web_app, "_schedule_dca_snapshot_sync_if_needed", return_value=False), patch.object(
            self.web_app.dca_fund_fetcher,
            "get_fund_info",
            return_value={"name": "测试基金超卖"},
        ):
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        fund = payload["funds"][0]
        self.assertAlmostEqual(fund["deficit_shares"], 50.0)
        self.assertAlmostEqual(payload["portfolio"]["total_deficit_shares"], 50.0)
        self.assertAlmostEqual(payload["records"][0]["deficit_shares_remaining"], 50.0)
        self.assertAlmostEqual(payload["records"][0]["offset_shares"], 0.0)

    def test_sync_dca_records_offsets_oversell_with_future_buy(self):
        self._insert_dca_record(
            "000001",
            "2026-03-10",
            100.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )
        self._insert_dca_record(
            "000001",
            "2026-03-11",
            150.0,
            trade_type=self.web_app.DCA_TRADE_TYPE_SELL,
            status=self.web_app.DCA_STATUS_PENDING,
        )
        self._insert_dca_record(
            "000001",
            "2026-03-12",
            60.0,
            status=self.web_app.DCA_STATUS_PENDING,
        )

        def _write_nav_rows(_fund_code):
            self._seed_nav_rows([
                ("2026-03-10", "000001", 1.0, 1.0),
                ("2026-03-11", "000001", 1.0, 1.0),
                ("2026-03-12", "000001", 0.8, 0.8),
            ])

        with patch.object(self.web_app.loader, "update_db", side_effect=_write_nav_rows):
            self.web_app._sync_dca_records()

        conn = sqlite3.connect(self.rec_db_path)
        sell_row = conn.execute("""
            SELECT deficit_shares_remaining, realized_pnl, net_cash, fee, shares, status
            FROM dca_records
            WHERE id = 2
        """).fetchone()
        buy_row = conn.execute("""
            SELECT shares, offset_shares, status
            FROM dca_records
            WHERE id = 3
        """).fetchone()
        conn.close()

        self.assertAlmostEqual(sell_row[0], 0.0)
        self.assertAlmostEqual(sell_row[1], 8.5)
        self.assertAlmostEqual(sell_row[2], 148.5)
        self.assertAlmostEqual(sell_row[3], 1.5)
        self.assertAlmostEqual(sell_row[4], 150.0)
        self.assertEqual(sell_row[5], self.web_app.DCA_STATUS_CONFIRMED)
        self.assertAlmostEqual(buy_row[0], 75.0)
        self.assertAlmostEqual(buy_row[1], 50.0)
        self.assertEqual(buy_row[2], self.web_app.DCA_STATUS_CONFIRMED)

        with patch.object(self.web_app, "_schedule_dca_snapshot_sync_if_needed", return_value=False), patch.object(
            self.web_app.dca_fund_fetcher,
            "get_fund_info",
            return_value={"name": "测试基金自动抵扣"},
        ):
            response = self.client.get("/api/dca/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        fund = payload["funds"][0]
        self.assertAlmostEqual(fund["total_shares"], 25.0)
        self.assertAlmostEqual(fund["remaining_cost_basis"], 20.0)
        self.assertAlmostEqual(fund["deficit_shares"], 0.0)
        self.assertAlmostEqual(fund["realized_profit_amount"], 8.5)
        self.assertAlmostEqual(payload["portfolio"]["total_deficit_shares"], 0.0)
        self.assertAlmostEqual(payload["records"][0]["offset_shares"], 50.0)


if __name__ == "__main__":
    unittest.main()
