"""
Kalshi scraper using the public v2 trading API.
Requires free account credentials for some endpoints; anonymous access for others.
"""

import time
from typing import Optional

import requests
from loguru import logger

BASE = "https://trading-api.kalshi.com/trade-api/v2"

SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})


def login(email: str, password: str) -> bool:
    try:
        r = SESSION.post(
            f"{BASE}/log_in",
            json={"email": email, "password": password},
            timeout=10,
        )
        r.raise_for_status()
        token = r.json().get("token", "")
        SESSION.headers["Authorization"] = f"Bearer {token}"
        logger.info("Kalshi login successful")
        return True
    except Exception as e:
        logger.warning(f"Kalshi login failed: {e}. Proceeding without auth.")
        return False


def _get(path: str, params: dict = None, retries: int = 3) -> dict:
    url = f"{BASE}{path}"
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if r.status_code == 401:
                logger.warning("Kalshi auth required for this endpoint, skipping")
                return {}
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def fetch_resolved_markets(limit: int = 500) -> list[dict]:
    """
    Fetch resolved Kalshi markets. Returns raw market dicts.
    """
    markets = []
    cursor = None
    page_size = 100

    logger.info("Fetching resolved Kalshi markets...")
    while len(markets) < limit:
        params = {"limit": page_size, "status": "finalized"}
        if cursor:
            params["cursor"] = cursor

        try:
            data = _get("/markets", params=params)
        except Exception as e:
            logger.error(f"Kalshi fetch failed: {e}")
            break

        page = data.get("markets", [])
        if not page:
            break

        markets.extend(page)
        cursor = data.get("cursor")
        if not cursor:
            break

        time.sleep(0.3)

    logger.info(f"Fetched {len(markets)} raw Kalshi markets")
    return markets[:limit]


def parse_market(raw: dict) -> Optional[dict]:
    """
    Parse a raw Kalshi market dict into a clean record.
    Kalshi markets are binary (Yes/No contracts).
    """
    try:
        result = raw.get("result", "")
        if result not in ("yes", "no"):
            return None

        resolution = result.upper()

        # last_price is the closing YES price in cents (0-100)
        last_price_cents = raw.get("last_price")
        if last_price_cents is None:
            return None
        closing_price = float(last_price_cents) / 100.0

        close_time = raw.get("close_time") or raw.get("expected_expiration_time")
        if not close_time:
            return None

        category = (raw.get("category") or "uncategorized").lower()

        return {
            "id": f"kalshi_{raw['ticker']}",
            "condition_id": raw.get("ticker", ""),
            "question": raw.get("title", ""),
            "category": category,
            "end_date": close_time,
            "resolution": resolution,
            "closing_price": closing_price,
            "volume": float(raw.get("volume", 0) or 0),
            "liquidity": float(raw.get("open_interest", 0) or 0),
            "yes_token_id": None,
            "source": "kalshi",
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"Skipping malformed Kalshi market {raw.get('ticker')}: {e}")
        return None


def fetch_price_history(ticker: str) -> list[dict]:
    """
    Fetch candlestick price history for a Kalshi market.
    Returns list of {timestamp, price} dicts.
    """
    try:
        data = _get(f"/markets/{ticker}/candlesticks", params={"period_interval": 1440})
        candles = data.get("candlesticks", [])
        result = []
        for c in candles:
            ts = c.get("end_period_ts")
            price = c.get("close", {}).get("yes_ask")
            if ts and price is not None:
                result.append({"timestamp": ts, "price": float(price) / 100.0})
        return sorted(result, key=lambda x: x["timestamp"])
    except Exception as e:
        logger.debug(f"Kalshi price history failed for {ticker}: {e}")
        return []


def scrape(n_markets: int = 500, email: str = None, password: str = None) -> tuple[list[dict], list[dict]]:
    """
    Full Kalshi scrape. Returns (markets, snapshots).
    Price history requires auth; markets are public.
    """
    if email and password:
        login(email, password)

    raw_markets = fetch_resolved_markets(limit=n_markets)
    parsed = [parse_market(r) for r in raw_markets]
    markets = [m for m in parsed if m is not None]
    logger.info(f"Parsed {len(markets)} valid Kalshi markets")
    return markets, []
