from __future__ import annotations

"""
DuckDB ingestion layer. Manages schema creation and bulk inserts.
"""

import os
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

DEFAULT_DB = os.getenv("DB_PATH", "data/markets.duckdb")

DDL = """
CREATE TABLE IF NOT EXISTS markets (
    id          VARCHAR PRIMARY KEY,
    condition_id VARCHAR,
    question    TEXT,
    category    VARCHAR,
    end_date    VARCHAR,
    resolution  VARCHAR,
    closing_price DOUBLE,
    volume      DOUBLE,
    liquidity   DOUBLE,
    yes_token_id VARCHAR,
    source      VARCHAR
);

CREATE TABLE IF NOT EXISTS prices (
    market_id   VARCHAR,
    timestamp   BIGINT,
    price       DOUBLE
);

CREATE TABLE IF NOT EXISTS snapshots (
    market_id       VARCHAR,
    hours_to_close  INTEGER,
    price           DOUBLE
);
"""


def get_conn(db_path: str = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    conn.execute(DDL)
    return conn


def insert_markets(conn: duckdb.DuckDBPyConnection, markets: list[dict]) -> int:
    if not markets:
        return 0
    df = pd.DataFrame(markets)
    existing = set(
        conn.execute("SELECT id FROM markets").fetchdf()["id"].tolist()
    )
    new = df[~df["id"].isin(existing)]
    if new.empty:
        logger.info("No new markets to insert")
        return 0
    conn.execute("INSERT INTO markets SELECT * FROM new")
    logger.info(f"Inserted {len(new)} markets")
    return len(new)


def insert_prices(conn: duckdb.DuckDBPyConnection, prices: list[dict]) -> int:
    if not prices:
        return 0
    df = pd.DataFrame(prices)
    conn.execute("INSERT INTO prices SELECT * FROM df")
    logger.info(f"Inserted {len(df)} price records")
    return len(df)


def insert_snapshots(conn: duckdb.DuckDBPyConnection, snapshots: list[dict]) -> int:
    if not snapshots:
        return 0
    df = pd.DataFrame(snapshots)
    conn.execute("INSERT INTO snapshots SELECT * FROM df")
    logger.info(f"Inserted {len(df)} snapshots")
    return len(df)


def load_analysis_df(conn: duckdb.DuckDBPyConnection, hours_to_close: int = 24) -> pd.DataFrame:
    """
    Join markets with snapshots at a specific hours_to_close.
    Returns DataFrame ready for analysis with columns:
        market_id, question, category, source, resolution,
        volume, liquidity, predicted_prob, outcome
    """
    query = f"""
    SELECT
        m.id            AS market_id,
        m.question,
        m.category,
        m.source,
        m.resolution,
        m.volume,
        m.liquidity,
        m.closing_price,
        s.price         AS predicted_prob,
        CASE m.resolution WHEN 'YES' THEN 1 ELSE 0 END AS outcome
    FROM markets m
    JOIN snapshots s
        ON m.id = s.market_id AND s.hours_to_close = {hours_to_close}
    WHERE s.price > 0.01 AND s.price < 0.99
    """
    df = conn.execute(query).fetchdf()
    logger.info(f"Loaded {len(df)} markets for analysis (T-{hours_to_close}h snapshot)")
    return df


def db_summary(conn: duckdb.DuckDBPyConnection) -> dict:
    return {
        "markets": conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0],
        "prices": conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0],
        "snapshots": conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0],
        "categories": conn.execute(
            "SELECT category, COUNT(*) as n FROM markets GROUP BY category ORDER BY n DESC"
        ).fetchdf().to_dict("records"),
        "sources": conn.execute(
            "SELECT source, COUNT(*) as n FROM markets GROUP BY source"
        ).fetchdf().to_dict("records"),
        "resolution_rate": conn.execute(
            "SELECT resolution, COUNT(*) as n FROM markets GROUP BY resolution"
        ).fetchdf().to_dict("records"),
    }
