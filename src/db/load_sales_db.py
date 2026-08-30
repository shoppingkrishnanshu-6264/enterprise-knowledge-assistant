"""
Loads data/structured/sales_data.csv into a local SQLite database.
Run this once (or whenever the CSV changes) before using the SQL tool.
"""

import pandas as pd
import sqlite3
import os

CSV_PATH = "data/structured/sales_data.csv"
DB_PATH = "data/structured/sales.db"
TABLE_NAME = "sales"


def load_csv_to_sqlite():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Check the path.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print(f"Stored as table '{TABLE_NAME}' in SQLite DB at: {DB_PATH}")


if __name__ == "__main__":
    load_csv_to_sqlite()
