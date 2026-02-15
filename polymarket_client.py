"""
Polymarket API Client
=====================
Read-only client for Gamma Markets API and CLOB API.
No authentication required for market data retrieval.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Rate limit: ~1000 requests/hour => ~1 req/3.6s to be safe
RATE_LIMIT_DELAY = 0.5  # seconds between requests (conservative)

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
RETRY_BACKOFF_FACTOR = 2.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class Market:
    """A single prediction market with outcome prices."""
    id: str
    question: str
    slug: str
    outcome_prices: list[float]  # [yes_price, no_price] where price = probability
    volume: float
    liquidity: float
    active: bool
    closed: bool
    end_date: Optional[str] = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    token_ids: list[str] = field(default_factory=list)

    @property
    def probability(self) -> float:
        """Implied probability of the 'Yes' outcome."""
        if self.outcome_prices:
            return self.outcome_prices[0]
        return 0.0


@dataclass
class Event:
    """A prediction event containing one or more markets."""
    id: str
    title: str
    slug: str
    description: str
    markets: list[Market]
    tags: list[dict]

    @property
    def tag_names(self) -> list[str]:
        return [t.get("label", t.get("slug", "")) for t in self.tags]


@dataclass
class PricePoint:
    """A single historical price observation."""
    timestamp: int  # unix seconds
    price: float    # 0.0 - 1.0


class PolymarketClient:
    """
    Read-only client for Polymarket APIs.

    Usage:
        client = PolymarketClient()
        tags = client.get_tags(limit=100)
        events = client.get_events(tag_id=100265)  # Geopolitics
        history = client.get_price_history(token_id="xxx", interval="1d")
    """

    def __init__(self, rate_limit_delay: float | None = None):
        # Load from config if available, fall back to module-level defaults
        if rate_limit_delay is None:
            try:
                from .config_loader import get_config
                cfg = get_config().api
                rate_limit_delay = cfg.rate_limit_delay
            except Exception:
                rate_limit_delay = RATE_LIMIT_DELAY

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PROVIDER-ScenarioGenerator/0.1"
        })
        self._rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0

    def _throttle(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _execute_with_retry(self, url: str, params: Optional[dict] = None) -> requests.Response:
        """Execute a GET request with exponential backoff retry."""
        from .metrics import MetricsCollector
        metrics = MetricsCollector()

        delay = INITIAL_RETRY_DELAY
        last_exception = None
        retried = False

        for attempt in range(MAX_RETRIES + 1):
            self._throttle()
            start = time.time()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                latency = time.time() - start

                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    metrics.track_request(latency, success=False, retried=True)
                    retried = True
                    logger.warning(
                        f"HTTP {resp.status_code} for {url} (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= RETRY_BACKOFF_FACTOR
                    continue

                resp.raise_for_status()
                metrics.track_request(latency, success=True, retried=retried)
                return resp
            except requests.exceptions.ConnectionError as e:
                latency = time.time() - start
                last_exception = e
                if attempt < MAX_RETRIES:
                    metrics.track_request(latency, success=False, retried=True)
                    retried = True
                    logger.warning(
                        f"Connection error for {url} (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= RETRY_BACKOFF_FACTOR
                    continue
                metrics.track_request(latency, success=False)
                raise
            except requests.exceptions.Timeout as e:
                latency = time.time() - start
                last_exception = e
                if attempt < MAX_RETRIES:
                    metrics.track_request(latency, success=False, retried=True)
                    retried = True
                    logger.warning(
                        f"Timeout for {url} (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= RETRY_BACKOFF_FACTOR
                    continue
                metrics.track_request(latency, success=False)
                raise

        raise last_exception  # should not reach here, but safety net

    def _get(self, base: str, path: str, params: Optional[dict] = None) -> dict | list:
        """Execute a throttled GET request with retry logic."""
        url = f"{base}{path}"
        logger.debug(f"GET {url} params={params}")
        try:
            resp = self._execute_with_retry(url, params)
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

    # ─── Gamma API: Tags ────────────────────────────────────────

    def get_tags(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """
        List available tags/categories.
        Returns: [{"id": "100265", "label": "Geopolitics", "slug": "geopolitics"}, ...]
        """
        data = self._get(GAMMA_BASE, "/tags", {"limit": limit, "offset": offset})
        if isinstance(data, list):
            return data
        return data.get("tags", data.get("data", []))

    def search_tags(self, query: str, limit: int = 50) -> list[dict]:
        """Search tags by keyword."""
        all_tags = []
        offset = 0
        while offset < 500:  # scan up to 500 tags
            batch = self.get_tags(limit=100, offset=offset)
            if not batch:
                break
            all_tags.extend(batch)
            offset += 100

        query_lower = query.lower()
        return [
            t for t in all_tags
            if query_lower in t.get("label", "").lower()
            or query_lower in t.get("slug", "").lower()
        ]

    # ─── Gamma API: Events & Markets ────────────────────────────

    def get_events(
        self,
        tag_id: Optional[int] = None,
        slug: Optional[str] = None,
        closed: bool = False,
        active: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Event]:
        """
        Fetch events (each containing 1+ markets).
        Filter by tag_id for category-based queries.
        """
        params = {"limit": limit, "offset": offset, "closed": str(closed).lower()}
        if tag_id is not None:
            params["tag_id"] = tag_id
        if slug:
            params["slug"] = slug
        if active:
            params["active"] = "true"

        raw = self._get(GAMMA_BASE, "/events", params)
        items = raw if isinstance(raw, list) else raw.get("data", raw.get("events", []))

        events = []
        for item in items:
            markets = []
            for m in item.get("markets", []):
                prices = self._parse_outcome_prices(m)
                token_ids = self._parse_token_ids(m)
                markets.append(Market(
                    id=str(m.get("id", "")),
                    question=m.get("question", ""),
                    slug=m.get("slug", m.get("conditionId", "")),
                    outcome_prices=prices,
                    volume=float(m.get("volume", 0) or 0),
                    liquidity=float(m.get("liquidity", 0) or 0),
                    active=m.get("active", True),
                    closed=m.get("closed", False),
                    end_date=m.get("endDate", m.get("end_date_iso")),
                    description=m.get("description", ""),
                    token_ids=token_ids,
                ))
            events.append(Event(
                id=str(item.get("id", "")),
                title=item.get("title", ""),
                slug=item.get("slug", ""),
                description=item.get("description", ""),
                markets=markets,
                tags=item.get("tags", []),
            ))
        return events

    def get_markets(
        self,
        tag_id: Optional[int] = None,
        closed: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Market]:
        """Fetch individual markets, optionally filtered by tag."""
        params = {"limit": limit, "offset": offset, "closed": str(closed).lower()}
        if tag_id is not None:
            params["tag_id"] = tag_id

        raw = self._get(GAMMA_BASE, "/markets", params)
        items = raw if isinstance(raw, list) else raw.get("data", raw.get("markets", []))

        markets = []
        for m in items:
            prices = self._parse_outcome_prices(m)
            token_ids = self._parse_token_ids(m)
            tags_raw = m.get("tags", [])
            tag_labels = []
            if isinstance(tags_raw, list):
                for t in tags_raw:
                    if isinstance(t, dict):
                        tag_labels.append(t.get("label", t.get("slug", "")))
                    elif isinstance(t, str):
                        tag_labels.append(t)

            markets.append(Market(
                id=str(m.get("id", "")),
                question=m.get("question", ""),
                slug=m.get("slug", m.get("conditionId", "")),
                outcome_prices=prices,
                volume=float(m.get("volume", 0) or 0),
                liquidity=float(m.get("liquidity", 0) or 0),
                active=m.get("active", True),
                closed=m.get("closed", False),
                end_date=m.get("endDate", m.get("end_date_iso")),
                description=m.get("description", ""),
                tags=tag_labels,
                token_ids=token_ids,
            ))
        return markets

    def search_markets(self, query: str, limit: int = 20) -> list[Market]:
        """Free-text search across all markets."""
        raw = self._get(GAMMA_BASE, "/markets", {"_q": query, "limit": limit})
        items = raw if isinstance(raw, list) else raw.get("data", raw.get("markets", []))

        markets = []
        for m in items:
            prices = self._parse_outcome_prices(m)
            token_ids = self._parse_token_ids(m)
            markets.append(Market(
                id=str(m.get("id", "")),
                question=m.get("question", ""),
                slug=m.get("slug", ""),
                outcome_prices=prices,
                volume=float(m.get("volume", 0) or 0),
                liquidity=float(m.get("liquidity", 0) or 0),
                active=m.get("active", True),
                closed=m.get("closed", False),
                description=m.get("description", ""),
                token_ids=token_ids,
            ))
        return markets

    # ─── CLOB API: Prices & History ─────────────────────────────

    def get_price(self, token_id: str) -> Optional[float]:
        """Get current price (probability) for a token."""
        try:
            data = self._get(CLOB_BASE, "/price", {"token_id": token_id})
            return float(data.get("price", 0))
        except Exception as e:
            logger.warning(f"Could not get price for token {token_id}: {e}")
            return None

    def get_price_history(
        self,
        token_id: str,
        interval: str = "1d",
        fidelity: int = 60,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> list[PricePoint]:
        """
        Fetch historical price time series for a token.

        Args:
            token_id: The CLOB token ID
            interval: Candle interval - "1m", "5m", "15m", "1h", "4h", "1d", "1w"
            fidelity: Data resolution in minutes (default 60)
            start_ts: Start timestamp (unix seconds)
            end_ts: End timestamp (unix seconds)

        Returns:
            List of PricePoint(timestamp, price) sorted chronologically
        """
        params = {"market": token_id, "interval": interval, "fidelity": fidelity}
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts

        try:
            data = self._get(CLOB_BASE, "/prices-history", params)
            history = data.get("history", data) if isinstance(data, dict) else data
            points = []
            if isinstance(history, list):
                for point in history:
                    if isinstance(point, dict):
                        ts = int(point.get("t", 0))
                        price = float(point.get("p", 0))
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        ts = int(point[0])
                        price = float(point[1])
                    else:
                        continue
                    points.append(PricePoint(timestamp=ts, price=price))
            return sorted(points, key=lambda p: p.timestamp)
        except Exception as e:
            logger.warning(f"Could not get price history for token {token_id}: {e}")
            return []

    def get_orderbook(self, token_id: str) -> dict:
        """
        Get current orderbook. Useful for spread analysis (confidence indicator).
        Returns: {"bids": [...], "asks": [...], "spread": float}
        """
        try:
            data = self._get(CLOB_BASE, "/book", {"token_id": token_id})
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            spread = None
            if bids and asks:
                best_bid = max(float(b.get("price", 0)) for b in bids) if bids else 0
                best_ask = min(float(a.get("price", 1)) for a in asks) if asks else 1
                spread = best_ask - best_bid
            return {"bids": bids, "asks": asks, "spread": spread}
        except Exception as e:
            logger.warning(f"Could not get orderbook for token {token_id}: {e}")
            return {"bids": [], "asks": [], "spread": None}

    # ─── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _parse_outcome_prices(market_data: dict) -> list[float]:
        """Parse outcome prices from various API response formats."""
        prices = market_data.get("outcomePrices", market_data.get("outcome_prices"))
        if prices is None:
            return []
        if isinstance(prices, str):
            # Format: "[0.65, 0.35]" or "0.65,0.35"
            prices = prices.strip("[]").split(",")
            return [float(p.strip().strip('"')) for p in prices if p.strip()]
        if isinstance(prices, list):
            return [float(p) for p in prices]
        return []

    @staticmethod
    def _parse_token_ids(market_data: dict) -> list[str]:
        """Parse CLOB token IDs from market data."""
        tokens = market_data.get("clobTokenIds", market_data.get("clob_token_ids"))
        if tokens is None:
            return []
        if isinstance(tokens, str):
            tokens = tokens.strip("[]").split(",")
            return [t.strip().strip('"') for t in tokens if t.strip()]
        if isinstance(tokens, list):
            return [str(t) for t in tokens]
        return []
