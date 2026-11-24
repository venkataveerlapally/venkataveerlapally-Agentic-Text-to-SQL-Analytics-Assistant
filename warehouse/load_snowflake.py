"""Create the 12 warehouse tables in Snowflake and load the same synthetic data.

Usage (after filling SNOWFLAKE_* values in .env):
    python -m warehouse.load_snowflake

Requires: pip install "snowflake-connector-python[pandas]"
The connecting role needs CREATE TABLE on the target schema. For the agent
itself, use a separate READ-ONLY role (see README).
"""

from __future__ import annotations

import pandas as pd

from config import settings
from warehouse.catalog import CATALOG, TABLE_ORDER, column_names, ddl
from warehouse.seed import generate


def main() -> None:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas

    conn = snowflake.connector.connect(**settings.snowflake_params(admin=True))
    cur = conn.cursor()
    try:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {settings.snowflake_database}")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {settings.snowflake_database}.{settings.snowflake_schema}")
        cur.execute(f"USE SCHEMA {settings.snowflake_database}.{settings.snowflake_schema}")
        data = generate()
        # drop in reverse dependency order
        for table in reversed(TABLE_ORDER):
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        for table in TABLE_ORDER:
            cur.execute(ddl(table, "snowflake"))
            df = pd.DataFrame(data[table], columns=[c.upper() for c in column_names(table)])
            for name, typ, _d, _fk in CATALOG[table]["columns"]:
                if typ == "DATE":
                    df[name.upper()] = pd.to_datetime(df[name.upper()]).dt.date
            ok, _chunks, nrows, _ = write_pandas(conn, df, table.upper(), quote_identifiers=False)
            print(f"  {table:<12} {nrows:>7,} rows {'ok' if ok else 'FAILED'}")
            # Attach business descriptions as Snowflake comments
            cur.execute(f"COMMENT ON TABLE {table} IS $${CATALOG[table]['description']}$$")
            for name, _t, desc, _fk in CATALOG[table]["columns"]:
                cur.execute(f"COMMENT ON COLUMN {table}.{name} IS $${desc}$$")
        print("Snowflake warehouse loaded.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
