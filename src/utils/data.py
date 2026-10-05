"""Shared data access for the EDA notebooks: DuckDB connection, column lists and cached intermediate tables."""
import uuid
from pathlib import Path

import duckdb as ddb
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
TRAIN = (DATA_DIR / "train_data.parquet").as_posix()
TEST = (DATA_DIR / "test_data.parquet").as_posix()

ID_COLS = ['customer_ID', 'S_2', '__index_level_0__']
TARGET = 'target'
CATEGORICAL_FEATURES = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']


def get_connection(train: str = TRAIN, test: str = TEST) -> ddb.DuckDBPyConnection:
    """In-memory DuckDB connection with `train` and `test` views over the parquet files.

    In-memory (not a .duckdb file) so several notebooks can be open at once without fighting over a file lock.
    `train` / `test` override the parquet paths (e.g. a copy on Colab's local disk, faster to read than Drive).
    A view whose file does not exist is skipped (e.g. test not uploaded to Colab).
    """
    con = ddb.connect()
    for name, path in [('train', train), ('test', test)]:
        if Path(path).exists():
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{Path(path).as_posix()}')")
    return con


def feature_columns(con: ddb.DuckDBPyConnection) -> list[str]:
    """All feature columns of train, in file order (no IDs, date or target)."""
    return [c for c in con.sql("DESCRIBE train").df()['column_name'] if c not in ID_COLS + [TARGET]]


def _write_cache(con: ddb.DuckDBPyConnection, name: str, query: str, refresh: bool) -> Path:
    """Path of INTERIM_DIR/<name>.parquet, running `query` into it first if needed.

    Writes to a .tmp file and renames it at the end, so an interrupted or still-running query never leaves a
    half-written parquet that another notebook would take for a valid cache.
    """
    path = INTERIM_DIR / f"{name}.parquet"
    if refresh or not path.exists() or path.stat().st_size == 0:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        try:
            con.execute(f"COPY ({query}) TO '{tmp.as_posix()}' (FORMAT parquet)")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    return path


def _cached_table(con: ddb.DuckDBPyConnection, name: str, query: str, refresh: bool) -> None:
    """Create table `name` from INTERIM_DIR/<name>.parquet, running `query` and writing the parquet first if needed."""
    path = _write_cache(con, name, query, refresh)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")


def load_s2_tables(con: ddb.DuckDBPyConnection, refresh: bool = False) -> None:
    """Create the statement-date tables `s2` and `s2_cust` (train + test), cached as parquet in data/interim.

    - `s2`: one row per statement: split, customer_ID, s2 (S_2 as DATE), stmt_idx (1 = first statement of the
      customer) and gap_days (days since the customer's previous statement, NULL for the first one)
    - `s2_cust`: one row per customer: first/last date, number of rows and distinct dates, span in days,
      median and max gap
    """
    _cached_table(con, "s2", """
        SELECT
            split, customer_ID, s2,
            ROW_NUMBER() OVER w AS stmt_idx,
            DATEDIFF('day', LAG(s2) OVER w, s2) AS gap_days
        FROM (
            SELECT 'train' AS split, customer_ID, S_2::DATE AS s2 FROM train
            UNION ALL
            SELECT 'test'  AS split, customer_ID, S_2::DATE AS s2 FROM test
        )
        WINDOW w AS (PARTITION BY split, customer_ID ORDER BY s2)
    """, refresh)

    _cached_table(con, "s2_cust", """
        SELECT
            split, customer_ID,
            MIN(s2) AS first_s2,
            MAX(s2) AS last_s2,
            COUNT(*) AS n_rows,
            COUNT(DISTINCT s2) AS n_dates,
            DATEDIFF('day', MIN(s2), MAX(s2)) AS span_days,
            MEDIAN(gap_days) AS median_gap,
            MAX(gap_days) AS max_gap
        FROM s2
        GROUP BY split, customer_ID
    """, refresh)


def load_cached(con: ddb.DuckDBPyConnection, name: str, query: str, refresh: bool = False) -> pd.DataFrame:
    """Result of `query` as a DataFrame, cached as INTERIM_DIR/<name>.parquet (the query only runs on the first call)."""
    path = _write_cache(con, name, query, refresh)
    return con.sql(f"SELECT * FROM read_parquet('{path.as_posix()}')").df()


def load_sample(con: ddb.DuckDBPyConnection, table: str, n: int = 300_000, seed: int = 42, refresh: bool = False) -> pd.DataFrame:
    """Random sample of `n` rows of `table` ('train' or 'test'), cached as parquet so every notebook gets the exact same rows."""
    return load_cached(con, f"{table}_sample_{n}_{seed}",
                       f"SELECT * FROM {table} USING SAMPLE {n} ROWS (reservoir, {seed})", refresh)


def load_train_sample(con: ddb.DuckDBPyConnection, n: int = 300_000, seed: int = 42, refresh: bool = False) -> pd.DataFrame:
    """Random sample of `n` train rows, cached as parquet so every notebook gets the exact same rows."""
    return load_sample(con, "train", n, seed, refresh)
