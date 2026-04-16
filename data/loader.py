import logging
import os
from datetime import datetime

import pandas as pd

from .sqlite_utils import connect_sqlite, run_sqlite_write_with_retry


logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, db_path=None):
        if db_path is None:
            # Default to data/db.sqlite relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(current_dir, 'db.sqlite')
        else:
            self.db_path = db_path

        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database structure."""
        def _action():
            conn = connect_sqlite(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS fund_nav (
                    date TEXT,
                    fund_code TEXT,
                    nav REAL,
                    acc_nav REAL,
                    PRIMARY KEY (date, fund_code)
                )
                ''')
                conn.commit()
            finally:
                conn.close()

        run_sqlite_write_with_retry(_action, logger=logger, action_name="loader_init_db")

    def fetch_nav(self, fund_code, start_date, end_date):
        """
        Fetch NAV data from pingzhongdata.js and normalize it for local storage.

        Args:
            fund_code (str): Fund code (e.g., '005658')
            start_date (str): Start date in 'YYYYMMDD' format
            end_date (str): End date in 'YYYYMMDD' format

        Returns:
            pd.DataFrame: DataFrame with columns ['date', 'nav', 'acc_nav']
        """
        try:
            try:
                from otc_fund_quant.nav_estimator.data.fetcher.fund_fetcher import FundFetcher
            except ImportError:
                from nav_estimator.data.fetcher.fund_fetcher import FundFetcher

            fund_data = FundFetcher.get_pingzhongdata_nav_history_df(fund_code)
            fund_data = fund_data.copy()
            fund_data["净值日期"] = pd.to_datetime(fund_data["净值日期"], errors="coerce")

            mask = (fund_data["净值日期"] >= pd.to_datetime(start_date)) & (
                fund_data["净值日期"] <= pd.to_datetime(end_date)
            )
            filtered_data = fund_data.loc[mask].copy()
            if filtered_data.empty:
                return pd.DataFrame(columns=["date", "fund_code", "nav", "acc_nav"])

            result = pd.DataFrame()
            result["date"] = filtered_data["净值日期"].dt.strftime("%Y-%m-%d")
            result["fund_code"] = fund_code
            result["nav"] = pd.to_numeric(filtered_data["单位净值"], errors="coerce").astype(float)
            result["acc_nav"] = pd.to_numeric(filtered_data["累计净值"], errors="coerce").fillna(result["nav"]).astype(float)
            return result[["date", "fund_code", "nav", "acc_nav"]]

        except Exception as e:
            print(f"Error fetching data for {fund_code}: {e}")
            return pd.DataFrame()

    def update_db(self, fund_code):
        """
        Fetch latest data and update local SQLite database.

        Args:
            fund_code (str): Fund code
        """
        # Determine start date: find last date in DB or default to a distant past
        conn = connect_sqlite(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT MAX(date) FROM fund_nav WHERE fund_code = ?', (fund_code,))
        last_date_row = cursor.fetchone()

        today = datetime.now().strftime('%Y%m%d')

        if last_date_row and last_date_row[0]:
            last_date = last_date_row[0]
            # Start from next day? Or just overlap to be safe.
            # Akshare usually takes YYYYMMDD. DB has YYYY-MM-DD.
            # Convert DB date to YYYYMMDD
            start_date = last_date.replace('-', '')
        else:
            start_date = '20000101' # Default start

        conn.close()

        # Fetch data
        df = self.fetch_nav(fund_code, start_date, today)

        if df.empty:
            print(f"No new data found for {fund_code}")
            return

        # Save to DB
        def _action():
            conn = connect_sqlite(self.db_path)
            try:
                # Use 'replace' to handle potential overlaps/updates
                # However, pandas to_sql 'replace' drops the table. We want 'append'.
                # But we need to handle duplicates (Primary Key).
                # SQLite doesn't have "INSERT OR UPDATE" standard in standard SQL without UPSERT syntax (newer sqlite).
                # We can delete existing records in the range and insert new ones.
                dates = df['date'].tolist()
                if dates:
                    placeholders = ','.join(['?'] * len(dates))
                    # Delete existing records for these dates to avoid PK constraint failure on re-run
                    query = f"DELETE FROM fund_nav WHERE fund_code = ? AND date IN ({placeholders})"
                    params = [fund_code] + dates
                    conn.execute(query, params)

                df.to_sql('fund_nav', conn, if_exists='append', index=False)
                conn.commit()
                print(f"Updated {len(df)} records for {fund_code}")
            finally:
                conn.close()

        try:
            run_sqlite_write_with_retry(_action, logger=logger, action_name="loader_update_db")
        except Exception as e:
            print(f"Error updating database: {e}")

    def load_nav(self, fund_code, date):
        """
        Read NAV from local DB for a specific date.

        Args:
            fund_code (str): Fund code
            date (str): Date in 'YYYY-MM-DD' format

        Returns:
            dict: {'nav': float, 'acc_nav': float} or None if not found
        """
        conn = connect_sqlite(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT nav, acc_nav FROM fund_nav
            WHERE fund_code = ? AND date = ?
        ''', (fund_code, date))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {'nav': row[0], 'acc_nav': row[1]}
        return None

if __name__ == "__main__":
    # Simple test
    loader = DataLoader()
    print(f"Database initialized at: {loader.db_path}")

    # Example usage (commented out to avoid auto-execution during import)
    # fund = "005658"
    # loader.update_db(fund)
    # print(loader.load_nav(fund, "2023-01-01"))
