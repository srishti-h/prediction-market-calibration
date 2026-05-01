"""
Polymarket scraper using the public Gamma API and CLOB API.
No API key required for read access.
"""
from __future__ import annotations

import time
import json
from datetime import datetime, timezone
from typing import Optional

import requests
import urllib3
from loguru import logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

SNAPSHOT_HOURS = [168, 72, 24, 1]  # hours before close to sample price

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://polymarket.com",
    "Referer": "https://polymarket.com/",
})
SESSION.verify = False


def _get(url: str, params: dict = None, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"Request failed ({e}), retrying in {wait}s...")
            time.sleep(wait)


def fetch_resolved_markets(limit: int = 500, max_pages: int = 40) -> list[dict]:
    """
    Fetch resolved (closed) markets from Polymarket Gamma API.
    Returns raw market dicts with outcome prices and resolution info.
    """
    markets = []
    offset = 0
    page_size = min(limit, 100)

    logger.info("Fetching resolved Polymarket markets...")
    while len(markets) < limit:
        params = {
            "active": "false",
            "closed": "true",
            "limit": page_size,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        }
        try:
            page = _get(f"{GAMMA_BASE}/markets", params=params)
        except Exception as e:
            logger.error(f"Failed at offset {offset}: {e}")
            break

        if not page:
            break

        markets.extend(page)
        offset += len(page)

        if len(page) < page_size or len(markets) >= limit:
            break

        time.sleep(0.3)

        if offset // page_size >= max_pages:
            break

    logger.info(f"Fetched {len(markets)} raw markets")
    return markets[:limit]


_CATEGORY_KEYWORDS = {
    "crypto": ["bitcoin", "btc", "eth", "ethereum", "crypto", "defi", "nft", "solana", "doge", "coin"],
    "sports": ["nfl", "nba", "mlb", "nhl", "soccer", "tennis", "ufc", "boxing", "esport", "cs2", "league-of", "dota", "football", "basketball", "baseball"],
    "politics": ["election", "president", "congress", "senate", "trump", "biden", "harris", "democrat", "republican", "vote", "poll"],
    "economics": ["fed", "inflation", "gdp", "unemployment", "interest-rate", "recession", "jobs", "cpi"],
    "entertainment": ["oscar", "grammy", "emmy", "movie", "celebrity", "award", "netflix"],
}


def _infer_category(raw: dict) -> str:
    text = " ".join([
        raw.get("slug", ""),
        raw.get("question", ""),
        *[e.get("slug", "") + " " + e.get("ticker", "") for e in (raw.get("events") or [])],
    ]).lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return "other"


def parse_market(raw: dict) -> Optional[dict]:
    """
    Parse a raw Gamma API market dict into a clean record.
    Works with any binary market (YES/NO, Over/Under, Team A/Team B).
    Treats outcome[0] as the reference outcome; outcomePrices[0] == "1" means it won.
    """
    try:
        outcomes_raw = raw.get("outcomes", "[]")
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        if len(outcomes) != 2:
            return None

        prices_raw = raw.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        if not prices or len(prices) != 2:
            return None

        closing_price = float(prices[0])  # price of outcome[0]

        # Infer resolution from closing price
        if closing_price >= 0.99:
            resolution = "YES"
        elif closing_price <= 0.01:
            resolution = "NO"
        else:
            return None  # not resolved or ambiguous

        end_date = raw.get("endDateIso") or raw.get("endDate")
        if not end_date:
            return None

        # clobTokenIds is a JSON array string; index 0 = outcome[0] token
        token_ids_raw = raw.get("clobTokenIds", "[]")
        token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else (token_ids_raw or [])
        yes_token_id = token_ids[0] if token_ids else None

        return {
            "id": str(raw["id"]),
            "condition_id": raw.get("conditionId", ""),
            "question": raw.get("question", ""),
            "category": _infer_category(raw),
            "end_date": str(end_date),
            "resolution": resolution,
            "closing_price": closing_price,
            "volume": float(raw.get("volumeClob") or raw.get("volume") or 0),
            "liquidity": float(raw.get("liquidityClob") or raw.get("liquidity") or 0),
            "yes_token_id": yes_token_id,
            "source": "polymarket",
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"Skipping malformed market {raw.get('id')}: {e}")
        return None


def fetch_price_history(token_id: str, fidelity: int = 60) -> list[dict]:
    """
    Fetch YES-token price history from the CLOB API.
    fidelity: candle size in minutes.
    Returns list of {timestamp, price} dicts sorted ascending.
    """
    if not token_id:
        return []
    try:
        data = _get(
            f"{CLOB_BASE}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": fidelity},
        )
        history = data.get("history", [])
        return [
            {"timestamp": int(pt["t"]), "price": float(pt["p"])}
            for pt in history
            if "t" in pt and "p" in pt
        ]
    except Exception as e:
        logger.debug(f"Price history fetch failed for token {token_id}: {e}")
        return []


def compute_snapshots(history: list[dict], end_ts: int) -> dict[int, float]:
    """
    Given price history and market close timestamp, return the YES price
    at each of SNAPSHOT_HOURS hours before close.
    Returns {hours_to_close: price}.
    """
    if not history:
        return {}

    snapshots = {}
    sorted_history = sorted(history, key=lambda x: x["timestamp"])

    for hours in SNAPSHOT_HOURS:
        target_ts = end_ts - hours * 3600
        # find latest price at or before target_ts
        candidates = [pt for pt in sorted_history if pt["timestamp"] <= target_ts]
        if candidates:
            snapshots[hours] = candidates[-1]["price"]

    return snapshots


def scrape(n_markets: int = 2000) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Full scrape pipeline. Returns (markets, prices, snapshots).
    """
    raw_markets = fetch_resolved_markets(limit=n_markets)
    parsed = [parse_market(r) for r in raw_markets]
    markets = [m for m in parsed if m is not None]
    logger.info(f"Parsed {len(markets)} valid binary markets")

    all_prices = []
    all_snapshots = []

    for i, market in enumerate(markets):
        if i % 100 == 0:
            logger.info(f"Fetching price history {i}/{len(markets)}...")

        token_id = market.get("yes_token_id")
        if not token_id:
            continue

        try:
            end_ts = int(
                datetime.fromisoformat(
                    market["end_date"].replace("Z", "+00:00")
                ).timestamp()
            )
        except Exception:
            continue

        history = fetch_price_history(token_id)
        time.sleep(0.1)

        for pt in history:
            all_prices.append({
                "market_id": market["id"],
                "timestamp": pt["timestamp"],
                "price": pt["price"],
            })

        snaps = compute_snapshots(history, end_ts)
        for hours, price in snaps.items():
            all_snapshots.append({
                "market_id": market["id"],
                "hours_to_close": hours,
                "price": price,
            })

    logger.info(
        f"Done. {len(markets)} markets, {len(all_prices)} price points, "
        f"{len(all_snapshots)} snapshots"
    )
    return markets, all_prices, all_snapshots
