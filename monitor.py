#!/usr/bin/env python3
"""
PROVIDER Early Warning Monitor
================================
Polls Polymarket markets at configurable intervals, detects threshold
crossings and rapid probability shifts, and writes alert state to JSON
for the live dashboard.

Alert types:
    THRESHOLD_CROSSED   - Probability crossed a configured threshold
    RAPID_SHIFT         - Probability changed >X% in the polling window
    NEW_HIGH            - All-time high probability observed
    CASCADE_RISK        - Multiple correlated events spiking simultaneously
    VOLUME_SPIKE        - Trading volume surge (market attention indicator)

Usage:
    # Live monitoring (polls Polymarket every 5 minutes)
    python -m terminal.monitor

    # Demo mode with simulated probability walks
    python -m terminal.monitor --demo --interval 3

    # Custom thresholds
    python -m terminal.monitor --demo \
        --threshold-warn 0.30 --threshold-critical 0.60
"""

import argparse
import json
import logging
import time
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Alert:
    """A single early warning alert."""
    id: str
    timestamp: str
    level: str           # "info", "warning", "critical"
    type: str            # THRESHOLD_CROSSED, RAPID_SHIFT, etc.
    market_id: str
    question: str
    disruption_type: str
    message: str
    current_probability: float
    previous_probability: Optional[float] = None
    change_pct: Optional[float] = None
    acknowledged: bool = False


@dataclass
class MarketState:
    """Tracked state for a single monitored market."""
    market_id: str
    question: str
    disruption_type: str
    eu_relevance: float
    current_probability: float
    previous_probability: float
    probability_history: list[float] = field(default_factory=list)
    timestamp_history: list[str] = field(default_factory=list)
    all_time_high: float = 0.0
    all_time_low: float = 1.0
    change_1h: float = 0.0
    change_24h: float = 0.0
    trend: str = "stable"  # "rising", "falling", "stable", "volatile"
    alert_level: str = "normal"  # "normal", "watch", "warning", "critical"
    current_volume: float = 0.0
    volume_history: list[float] = field(default_factory=list)
    volume_baseline: float = 0.0


@dataclass
class MonitorState:
    """Complete monitor state written to JSON for the dashboard."""
    last_updated: str
    poll_count: int
    markets: list[dict]
    active_alerts: list[dict]
    alert_history: list[dict]
    statistics: dict


# ═══════════════════════════════════════════════════════════════
# ALERT ENGINE
# ═══════════════════════════════════════════════════════════════

class AlertEngine:
    """
    Evaluates market states and generates alerts based on configurable rules.
    """

    def __init__(
        self,
        threshold_watch: float = 0.25,
        threshold_warn: float = 0.40,
        threshold_critical: float = 0.60,
        rapid_shift_pct: float = 0.10,    # 10% change triggers alert
        cascade_threshold: int = 3,        # N correlated markets spiking
    ):
        self.threshold_watch = threshold_watch
        self.threshold_warn = threshold_warn
        self.threshold_critical = threshold_critical
        self.rapid_shift_pct = rapid_shift_pct
        self.cascade_threshold = cascade_threshold
        self._alert_counter = 0

    def evaluate(
        self,
        markets: dict[str, MarketState],
        previous_alerts: list[Alert],
    ) -> list[Alert]:
        """Evaluate all markets and return new alerts."""
        new_alerts = []
        now = datetime.now(timezone.utc).isoformat()

        # Track which alerts already exist to avoid duplicates
        existing_keys = {
            (a.type, a.market_id)
            for a in previous_alerts
            if not a.acknowledged
        }

        for mid, state in markets.items():
            # ── Threshold crossing ──
            prev_level = self._classify_level(state.previous_probability)
            curr_level = self._classify_level(state.current_probability)

            if curr_level != prev_level and curr_level in ("warning", "critical"):
                key = ("THRESHOLD_CROSSED", mid)
                if key not in existing_keys:
                    self._alert_counter += 1
                    new_alerts.append(Alert(
                        id=f"alert_{self._alert_counter:04d}",
                        timestamp=now,
                        level=curr_level,
                        type="THRESHOLD_CROSSED",
                        market_id=mid,
                        question=state.question,
                        disruption_type=state.disruption_type,
                        message=f"Wahrscheinlichkeit hat {self.threshold_warn*100:.0f}%-Schwelle überschritten"
                            if curr_level == "warning"
                            else f"KRITISCH: Wahrscheinlichkeit über {self.threshold_critical*100:.0f}%",
                        current_probability=state.current_probability,
                        previous_probability=state.previous_probability,
                    ))

            # ── Rapid shift ──
            if len(state.probability_history) >= 2:
                change = abs(state.current_probability - state.previous_probability)
                if change >= self.rapid_shift_pct:
                    key = ("RAPID_SHIFT", mid)
                    if key not in existing_keys:
                        direction = "gestiegen" if state.current_probability > state.previous_probability else "gefallen"
                        self._alert_counter += 1
                        new_alerts.append(Alert(
                            id=f"alert_{self._alert_counter:04d}",
                            timestamp=now,
                            level="warning",
                            type="RAPID_SHIFT",
                            market_id=mid,
                            question=state.question,
                            disruption_type=state.disruption_type,
                            message=f"Wahrscheinlichkeit um {change*100:.1f}% {direction} in letztem Intervall",
                            current_probability=state.current_probability,
                            previous_probability=state.previous_probability,
                            change_pct=change * 100,
                        ))

            # ── New all-time high ──
            if state.current_probability > state.all_time_high and len(state.probability_history) > 5:
                key = ("NEW_HIGH", mid)
                if key not in existing_keys:
                    self._alert_counter += 1
                    new_alerts.append(Alert(
                        id=f"alert_{self._alert_counter:04d}",
                        timestamp=now,
                        level="info",
                        type="NEW_HIGH",
                        market_id=mid,
                        question=state.question,
                        disruption_type=state.disruption_type,
                        message=f"Neues Allzeithoch: {state.current_probability*100:.1f}%",
                        current_probability=state.current_probability,
                    ))

            # Update alert level on market state
            state.alert_level = curr_level

        # ── Volume spike ──
        for mid, state in markets.items():
            if (state.volume_baseline > 0 and state.current_volume > 0
                    and state.current_volume >= state.volume_baseline * 3):
                key = ("VOLUME_SPIKE", mid)
                if key not in existing_keys:
                    self._alert_counter += 1
                    spike_factor = state.current_volume / state.volume_baseline
                    new_alerts.append(Alert(
                        id=f"alert_{self._alert_counter:04d}",
                        timestamp=now,
                        level="warning",
                        type="VOLUME_SPIKE",
                        market_id=mid,
                        question=state.question,
                        disruption_type=state.disruption_type,
                        message=f"Volumen-Spike: {spike_factor:.1f}x ueber Baseline",
                        current_probability=state.current_probability,
                    ))

        # ── Cascade risk ──
        # Group rising markets by disruption type
        rising_by_type: dict[str, list[str]] = {}
        for mid, state in markets.items():
            if state.trend == "rising" and state.current_probability > self.threshold_watch:
                rising_by_type.setdefault(state.disruption_type, []).append(mid)

        for dtype, mids in rising_by_type.items():
            if len(mids) >= self.cascade_threshold:
                key = ("CASCADE_RISK", dtype)
                if key not in {(a.type, a.market_id) for a in previous_alerts if not a.acknowledged}:
                    self._alert_counter += 1
                    new_alerts.append(Alert(
                        id=f"alert_{self._alert_counter:04d}",
                        timestamp=now,
                        level="critical",
                        type="CASCADE_RISK",
                        market_id=dtype,
                        question=f"{len(mids)} korrelierte {dtype}-Events steigen gleichzeitig",
                        disruption_type=dtype,
                        message=f"Kaskadenrisiko: {len(mids)} Märkte im Bereich '{dtype}' steigen gleichzeitig",
                        current_probability=max(markets[m].current_probability for m in mids),
                    ))

        return new_alerts

    def _classify_level(self, prob: float) -> str:
        if prob >= self.threshold_critical:
            return "critical"
        if prob >= self.threshold_warn:
            return "warning"
        if prob >= self.threshold_watch:
            return "watch"
        return "normal"


# ═══════════════════════════════════════════════════════════════
# MARKET MONITOR
# ═══════════════════════════════════════════════════════════════

class MarketMonitor:
    """
    Polls markets and maintains tracked state with trend analysis.
    """

    def __init__(self, max_history: int = 288):  # 288 * 5min = 24h
        self.markets: dict[str, MarketState] = {}
        self.max_history = max_history

    def update(self, market_id: str, probability: float, question: str,
               disruption_type: str, eu_relevance: float, volume: float = 0.0):
        """Update a market with a new probability observation."""
        now = datetime.now(timezone.utc).isoformat()

        if market_id not in self.markets:
            self.markets[market_id] = MarketState(
                market_id=market_id,
                question=question,
                disruption_type=disruption_type,
                eu_relevance=eu_relevance,
                current_probability=probability,
                previous_probability=probability,
            )

        state = self.markets[market_id]
        state.previous_probability = state.current_probability
        state.current_probability = probability
        state.probability_history.append(probability)
        state.timestamp_history.append(now)

        # Volume tracking
        if volume > 0:
            state.current_volume = volume
            state.volume_history.append(volume)
            if len(state.volume_history) > self.max_history:
                state.volume_history = state.volume_history[-self.max_history:]
            if len(state.volume_history) >= 5:
                state.volume_baseline = float(np.mean(state.volume_history[:-1]))

        # Trim history
        if len(state.probability_history) > self.max_history:
            state.probability_history = state.probability_history[-self.max_history:]
            state.timestamp_history = state.timestamp_history[-self.max_history:]

        # Update extremes
        state.all_time_high = max(state.all_time_high, probability)
        state.all_time_low = min(state.all_time_low, probability)

        # Compute changes
        h = state.probability_history
        if len(h) >= 2:
            state.change_1h = h[-1] - h[-2]
        if len(h) >= 12:  # ~1h at 5min intervals
            state.change_24h = h[-1] - h[-12]
        elif len(h) >= 2:
            state.change_24h = h[-1] - h[0]

        # Trend classification
        state.trend = self._classify_trend(h)

    def _classify_trend(self, history: list[float]) -> str:
        if len(history) < 3:
            return "stable"

        recent = history[-5:] if len(history) >= 5 else history
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]

        avg_diff = sum(diffs) / len(diffs)
        volatility = (sum(d**2 for d in diffs) / len(diffs)) ** 0.5

        if volatility > 0.05:
            return "volatile"
        if avg_diff > 0.01:
            return "rising"
        if avg_diff < -0.01:
            return "falling"
        return "stable"


# ═══════════════════════════════════════════════════════════════
# DEMO SIMULATOR
# ═══════════════════════════════════════════════════════════════

class DemoSimulator:
    """Simulates realistic probability movements for demo mode."""

    DEMO_MARKETS = [
        ("ew_ukraine", "Russia-Ukraine ceasefire by end of 2026?", "geopolitical_conflict", 1.0, 0.44),
        ("ew_china_taiwan", "Will China invade Taiwan before 2027?", "geopolitical_conflict", 0.9, 0.13),
        ("ew_eu_tariff", "30% EU tariff in effect by August 2026?", "trade_restriction", 1.0, 0.35),
        ("ew_recession", "US recession by end of 2026?", "financial_crisis", 0.9, 0.25),
        ("ew_pandemic", "New WHO-declared pandemic in 2026?", "pandemic", 0.9, 0.08),
        ("ew_oil", "Oil price above $80/barrel by June 2026?", "energy_supply", 0.9, 0.40),
        ("ew_disaster", "Major natural disaster ($50B+) in 2026?", "climate_event", 0.7, 0.30),
        ("ew_rare_earth", "Ukraine rare earth deal before April 2026?", "tech_disruption", 0.8, 0.55),
        ("ew_brazil", "Brazil agricultural export restrictions 2026?", "agricultural_shock", 0.9, 0.12),
        ("ew_red_sea", "Red Sea shipping disruptions through 2026?", "transport_disruption", 0.9, 0.60),
        ("ew_ecb", "ECB cuts rates below 2% by end of 2026?", "financial_crisis", 1.0, 0.35),
        ("ew_argentina", "Argentina sovereign default in 2026?", "political_instability", 0.7, 0.15),
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.current_probs = {m[0]: m[4] for m in self.DEMO_MARKETS}
        self.step = 0

    def tick(self) -> list[tuple[str, str, str, float, float]]:
        """
        Simulate one time step. Returns list of
        (market_id, question, disruption_type, eu_relevance, new_probability).
        """
        self.step += 1
        results = []

        for mid, question, dtype, relevance, base_prob in self.DEMO_MARKETS:
            prob = self.current_probs[mid]

            # Mean-reverting random walk with occasional jumps
            drift = (base_prob - prob) * 0.02  # mean reversion
            noise = self.rng.normal(0, 0.015)  # normal noise

            # Occasional jump (5% chance per tick)
            if self.rng.random() < 0.05:
                noise += self.rng.choice([-1, 1]) * self.rng.uniform(0.05, 0.15)

            # Correlated shock: if geopolitical conflict rises, energy rises too
            if dtype == "energy_supply" and "ew_ukraine" in self.current_probs:
                ukraine_change = self.current_probs["ew_ukraine"] - base_prob
                noise += ukraine_change * 0.3 * self.rng.normal(0, 0.5)

            prob = np.clip(prob + drift + noise, 0.01, 0.99)
            self.current_probs[mid] = float(prob)
            results.append((mid, question, dtype, relevance, float(prob)))

        return results


# ═══════════════════════════════════════════════════════════════
# MAIN MONITORING LOOP
# ═══════════════════════════════════════════════════════════════

def run_monitor(
    demo: bool = True,
    interval: int = 5,
    output_file: str = "monitor_state.json",
    threshold_warn: float = 0.40,
    threshold_critical: float = 0.60,
    max_ticks: int = 0,
):
    """
    Main monitoring loop.

    Args:
        demo: Use simulated data instead of live API
        interval: Seconds between polls (demo) or minutes (live)
        output_file: Path to write monitor state JSON
        threshold_warn: Warning threshold
        threshold_critical: Critical threshold
        max_ticks: Stop after N ticks (0 = infinite)
    """
    monitor = MarketMonitor()
    alert_engine = AlertEngine(
        threshold_warn=threshold_warn,
        threshold_critical=threshold_critical,
    )
    simulator = DemoSimulator() if demo else None

    all_alerts: list[Alert] = []
    active_alerts: list[Alert] = []
    poll_count = 0
    live_mapper = None  # persistent EventMapper for live mode caching

    logger.info(f"PROVIDER Early Warning Monitor started")
    logger.info(f"  Mode: {'DEMO' if demo else 'LIVE'}")
    logger.info(f"  Interval: {interval}s")
    logger.info(f"  Thresholds: warn={threshold_warn:.0%}, critical={threshold_critical:.0%}")
    logger.info(f"  Output: {output_file}")

    try:
        while True:
            poll_count += 1

            # ── Fetch data ──
            if demo:
                updates = simulator.tick()
                for mid, question, dtype, relevance, prob in updates:
                    monitor.update(mid, prob, question, dtype, relevance)
            else:
                # Live mode: import and use PolymarketClient with caching
                try:
                    if live_mapper is None:
                        from .polymarket_client import PolymarketClient
                        from .event_mapper import EventMapper
                        live_mapper = EventMapper(PolymarketClient())

                    # Full discovery on first tick and every 12 ticks, otherwise price-only refresh
                    if poll_count == 1 or poll_count % 12 == 0 or not live_mapper.cache_valid:
                        logger.info(f"  Full Discovery (Tick #{poll_count})")
                        mapped = live_mapper.discover_and_map(use_search=False)
                    else:
                        logger.debug(f"  Price Refresh (Tick #{poll_count})")
                        mapped = live_mapper.refresh_prices()

                    for event in mapped:
                        monitor.update(
                            event.market_id, event.probability,
                            event.question, event.disruption_type,
                            event.eu_relevance,
                        )
                except requests.exceptions.RequestException as e:
                    logger.warning(f"API-Fehler bei Poll #{poll_count}, nutze bestehende Daten: {e}")
                except Exception as e:
                    logger.warning(f"Unerwarteter Fehler bei Poll #{poll_count}: {e}")

            # ── Evaluate alerts ──
            new_alerts = alert_engine.evaluate(monitor.markets, active_alerts)
            all_alerts.extend(new_alerts)
            active_alerts.extend(new_alerts)

            # Auto-dismiss old info alerts after 20 ticks
            active_alerts = [
                a for a in active_alerts
                if not a.acknowledged and (
                    a.level != "info" or
                    (datetime.now(timezone.utc) - datetime.fromisoformat(a.timestamp)).seconds < interval * 20
                )
            ]

            # ── Log alerts ──
            for alert in new_alerts:
                icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}.get(alert.level, "")
                logger.info(f"  {icon} [{alert.level.upper()}] {alert.message} ({alert.question})")

            # ── Write state ──
            state = MonitorState(
                last_updated=datetime.now(timezone.utc).isoformat(),
                poll_count=poll_count,
                markets=[asdict(s) for s in monitor.markets.values()],
                active_alerts=[asdict(a) for a in active_alerts],
                alert_history=[asdict(a) for a in all_alerts[-100:]],  # last 100
                statistics={
                    "total_alerts": len(all_alerts),
                    "active_critical": sum(1 for a in active_alerts if a.level == "critical"),
                    "active_warning": sum(1 for a in active_alerts if a.level == "warning"),
                    "markets_monitored": len(monitor.markets),
                    "avg_probability": float(np.mean([
                        s.current_probability for s in monitor.markets.values()
                    ])) if monitor.markets else 0,
                    "max_probability": float(max(
                        (s.current_probability for s in monitor.markets.values()), default=0
                    )),
                    "rising_count": sum(1 for s in monitor.markets.values() if s.trend == "rising"),
                    "falling_count": sum(1 for s in monitor.markets.values() if s.trend == "falling"),
                },
            )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(asdict(state), f, indent=2, ensure_ascii=False)

            # Print compact status line
            n_crit = state.statistics["active_critical"]
            n_warn = state.statistics["active_warning"]
            status = f"Poll #{poll_count:4d}  |  "
            status += f"Markets: {len(monitor.markets)}  |  "
            status += f"Alerts: {n_crit}C {n_warn}W  |  "
            status += f"Rising: {state.statistics['rising_count']}  "
            status += f"Falling: {state.statistics['falling_count']}"
            print(f"\r{status}", end="", flush=True)

            # Periodic health check log (every 20 ticks)
            if not demo and poll_count % 20 == 0:
                try:
                    from .metrics import MetricsCollector
                    MetricsCollector().log_health_check()
                except Exception:
                    pass

            if max_ticks and poll_count >= max_ticks:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("\nMonitor stopped by user")

    print(f"\n\nFinal state written to {output_file}")
    print(f"Total alerts generated: {len(all_alerts)}")
    return state


def main():
    parser = argparse.ArgumentParser(
        description="PROVIDER Early Warning Monitor",
    )
    parser.add_argument("--demo", action="store_true", help="Use simulated data")
    parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds (demo) or minutes (live)")
    parser.add_argument("--output", type=str, default="monitor_state.json", help="Output state file")
    parser.add_argument("--threshold-warn", type=float, default=0.40)
    parser.add_argument("--threshold-critical", type=float, default=0.60)
    parser.add_argument("--max-ticks", type=int, default=0, help="Stop after N polls (0=infinite)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run_monitor(
        demo=args.demo,
        interval=args.interval,
        output_file=args.output,
        threshold_warn=args.threshold_warn,
        threshold_critical=args.threshold_critical,
        max_ticks=args.max_ticks,
    )


if __name__ == "__main__":
    main()
