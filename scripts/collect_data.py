"""
Entry point for data collection. Scrapes Polymarket (and optionally Kalshi)
and stores results in DuckDB.

Usage:
    python scripts/collect_data.py --markets 2000
    python scripts/collect_data.py --markets 500 --kalshi
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from loguru import logger

from src.scrapers import polymarket, kalshi
from src.pipeline.ingest import get_conn, insert_markets, insert_prices, insert_snapshots, db_summary

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Collect prediction market data")
    parser.add_argument("--markets", type=int, default=2000, help="Number of markets to fetch from Polymarket")
    parser.add_argument("--kalshi", action="store_true", help="Also scrape Kalshi (requires .env credentials)")
    parser.add_argument("--kalshi-only", action="store_true", help="Skip Polymarket, only scrape Kalshi")
    parser.add_argument("--db", default=os.getenv("DB_PATH", "data/markets.duckdb"), help="Path to DuckDB file")
    args = parser.parse_args()

    logger.info(f"Starting data collection → {args.db}")
    conn = get_conn(args.db)

    # Polymarket
    if not args.kalshi_only:
        logger.info("=== Polymarket ===")
        pm_markets, pm_prices, pm_snapshots = polymarket.scrape(n_markets=args.markets)
        insert_markets(conn, pm_markets)
        insert_prices(conn, pm_prices)
        insert_snapshots(conn, pm_snapshots)

    # Kalshi (optional)
    if args.kalshi or args.kalshi_only:
        logger.info("=== Kalshi ===")
        email = os.getenv("KALSHI_EMAIL")
        password = os.getenv("KALSHI_PASSWORD")
        k_markets, k_snapshots = kalshi.scrape(n_markets=500, email=email, password=password)
        insert_markets(conn, k_markets)
        insert_snapshots(conn, k_snapshots)

    summary = db_summary(conn)
    logger.info("=== DB Summary ===")
    logger.info(f"Markets: {summary['markets']}")
    logger.info(f"Price records: {summary['prices']}")
    logger.info(f"Snapshots: {summary['snapshots']}")
    for row in summary["categories"]:
        logger.info(f"  {row['category']}: {row['n']}")

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
