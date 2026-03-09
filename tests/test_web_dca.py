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
        status,
        confirm_nav_date=None,
        confirm_nav=None,
        shares=None,
    ):
        conn = sqlite3.connect(self.rec_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO dca_records (
                fund_code, trade_date, amount, status,
                confirm_nav_date, confirm_nav, shares, confirmed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CASE WHEN ? = 'confirmed' THEN CURRENT_TIMESTAMP ELSE NULL END)
            """,
            (fund_code, trade_date, amount, status, confirm_nav_date, confirm_nav, shares, status),
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


if __name__ == "__main__":
    unittest.main()
