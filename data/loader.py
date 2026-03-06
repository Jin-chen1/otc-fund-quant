import akshare as ak
import sqlite3
import pandas as pd
from datetime import datetime
import os

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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create fund_nav table if not exists
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
        conn.close()

    def fetch_nav(self, fund_code, start_date, end_date):
        """
        Fetch NAV data using akshare.

        Args:
            fund_code (str): Fund code (e.g., '005658')
            start_date (str): Start date in 'YYYYMMDD' format
            end_date (str): End date in 'YYYYMMDD' format

        Returns:
            pd.DataFrame: DataFrame with columns ['date', 'nav', 'acc_nav']
        """
        try:
            # fund_open_fund_info_em returns date, net_value, accumulated_net_value, etc.
            # adjusting parameters to match akshare's API
            fund_data = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")

            # Filter by date
            fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
            mask = (fund_data['净值日期'] >= pd.to_datetime(start_date)) & \
                   (fund_data['净值日期'] <= pd.to_datetime(end_date))
            filtered_data = fund_data.loc[mask].copy()

            # Rename columns to match our schema
            # akshare output columns: 净值日期, 单位净值, 日增长率, etc.
            # We need to ensure we get accumulated nav if available,
            # but fund_open_fund_info_em mainly gives unit nav.
            # Let's check if we need another API for accumulated NAV or if it's included.
            # Usually fund_open_fund_info_em gives basic history.

            # Mapping
            # 净值日期 -> date
            # 单位净值 -> nav
            # akshare '单位净值走势' usually just returns unit nav.
            # '累计净值走势' is a different indicator?
            # Let's try to get both or assume standard structure.

            # Actually ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势") gets acc nav.
            # Getting both might require two calls or a different API.
            # Let's use fund_etf_hist_em for ETFs or fund_open_fund_info_em for open funds.
            # Assuming open funds given the context.

            # To get both Unit NAV and Accumulated NAV efficiently, we might need:
            # fund_open_fund_info_em defaults to unit nav.

            # Let's try a more comprehensive API if available, or just map what we have.
            # If we only get Unit NAV, we might leave Acc NAV as None or fetch separately.
            # For simplicity and robustness, let's fetch Unit NAV first.

            result = pd.DataFrame()
            result['date'] = filtered_data['净值日期'].dt.strftime('%Y-%m-%d')
            result['fund_code'] = fund_code
            result['nav'] = filtered_data['单位净值'].astype(float)

            # Attempt to get Accumulated NAV if possible, otherwise set to equal NAV or 0
            # For accurate Acc NAV, we might need a separate call.
            # Let's try to fetch acc nav separately and merge.
            try:
                acc_data = ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势")
                acc_data['净值日期'] = pd.to_datetime(acc_data['净值日期'])
                acc_mask = (acc_data['净值日期'] >= pd.to_datetime(start_date)) & \
                           (acc_data['净值日期'] <= pd.to_datetime(end_date))
                acc_filtered = acc_data.loc[acc_mask].copy()

                # Merge
                # acc_filtered has '净值日期' and '累计净值'
                acc_filtered['date_str'] = acc_filtered['净值日期'].dt.strftime('%Y-%m-%d')

                # Create a mapping dict
                acc_map = dict(zip(acc_filtered['date_str'], acc_filtered['累计净值']))

                result['acc_nav'] = result['date'].map(acc_map)

                # Fill missing acc_nav with nav if missing (fallback)
                result['acc_nav'] = result['acc_nav'].fillna(result['nav'])

            except Exception as e:
                print(f"Warning: Could not fetch accumulated NAV: {e}")
                result['acc_nav'] = result['nav']

            return result[['date', 'fund_code', 'nav', 'acc_nav']]

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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
            print(f"Updated {len(df)} records for {fund_code}")

        except Exception as e:
            print(f"Error updating database: {e}")
        finally:
            conn.commit()
            conn.close()

    def load_nav(self, fund_code, date):
        """
        Read NAV from local DB for a specific date.

        Args:
            fund_code (str): Fund code
            date (str): Date in 'YYYY-MM-DD' format

        Returns:
            dict: {'nav': float, 'acc_nav': float} or None if not found
        """
        conn = sqlite3.connect(self.db_path)
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
