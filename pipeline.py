#!/usr/bin/env python3
"""
PROVIDER Integrated Pipeline
==============================
Connects the Early Warning Monitor with the Scenario Generator.

Architecture:
    ┌─────────────────┐
    │  Polymarket API  │  (or Demo Simulator)
    └────────┬────────┘
             │  poll every N seconds/minutes
             ▼
    ┌─────────────────┐
    │  Market Monitor  │  tracks probabilities, trends, extremes
    └────────┬────────┘
             │  on every tick
             ▼
    ┌─────────────────┐
    │  Alert Engine    │  threshold crossings, rapid shifts, cascades
    └────────┬────────┘
             │  always + on trigger events
             ▼
    ┌─────────────────┐
    │  Scenario Bridge │  converts monitor state → MappedEvents
    └────────┬────────┘
             │  re-samples when probabilities shift significantly
             ▼
    ┌─────────────────┐
    │ Scenario Sampler │  Monte Carlo with updated probabilities
    └────────┬────────┘
             │  writes combined state JSON
             ▼
    ┌─────────────────┐
    │  Dashboard JSON  │  consumed by HTML dashboard via fetch/reload
    └─────────────────┘

Trigger logic for re-sampling:
    - ALWAYS: lightweight scenario stats updated every tick
    - FULL RESAMPLE when:
        - Any market crosses a threshold (warn/critical)
        - Any market shifts >10% in one tick
        - Cascade risk detected
        - Every N ticks (configurable, default 10)

Usage:
    python -m terminal.pipeline --demo --interval 3
    python -m terminal.pipeline --demo --interval 1 --max-ticks 30
"""

import argparse
import json
import logging
import time
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .monitor import MarketMonitor, AlertEngine, DemoSimulator, Alert
from .event_mapper import MappedEvent, DISRUPTION_PROFILES
from .scenario_sampler import ScenarioSampler

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SCENARIO BRIDGE
# ═══════════════════════════════════════════════════════════════

class ScenarioBridge:
    """
    Converts live MarketMonitor state into MappedEvents that the
    ScenarioSampler can consume. This is the glue between real-time
    monitoring and scenario generation.
    """

    def __init__(self):
        self._last_sample_probs: dict[str, float] = {}
        self._resample_count = 0

    def monitor_to_events(self, monitor: MarketMonitor) -> list[MappedEvent]:
        """Convert current monitor state to MappedEvents for sampling."""
        events = []
        for mid, state in monitor.markets.items():
            # Build parameters from disruption profiles
            dtype = state.disruption_type
            prob = state.current_probability
            profile = DISRUPTION_PROFILES.get(dtype, [])
            params = []
            for p in profile:
                if p.unit == "multiplier":
                    scaled = 1.0 + (p.base_impact - 1.0) * prob
                else:
                    scaled = p.base_impact * prob
                params.append({
                    "name": p.name,
                    "unit": p.unit,
                    "base_impact": p.base_impact,
                    "scaled_impact": round(scaled, 4),
                    "probability": round(prob, 4),
                    "description": p.description,
                    "affected_sectors": p.affected_sectors,
                })

            events.append(MappedEvent(
                market_id=mid,
                question=state.question,
                probability=prob,
                volume=1_000_000,  # placeholder for demo
                token_id=None,
                disruption_type=dtype,
                eu_relevance=state.eu_relevance,
                parameters=params,
            ))
        return events

    def should_resample(
        self,
        monitor: MarketMonitor,
        new_alerts: list[Alert],
        tick: int,
        resample_interval: int = 10,
        shift_threshold: float = 0.05,
    ) -> tuple[bool, str]:
        """
        Decide whether to trigger a full scenario resample.

        Returns: (should_resample, reason)
        """
        # Trigger: critical or warning alert
        for alert in new_alerts:
            if alert.level in ("critical", "warning") and alert.type in (
                "THRESHOLD_CROSSED", "RAPID_SHIFT", "CASCADE_RISK"
            ):
                return True, f"Alert: {alert.type} ({alert.level})"

        # Trigger: significant cumulative drift since last sample
        if self._last_sample_probs:
            max_drift = 0.0
            for mid, state in monitor.markets.items():
                if mid in self._last_sample_probs:
                    drift = abs(state.current_probability - self._last_sample_probs[mid])
                    max_drift = max(max_drift, drift)
            if max_drift >= shift_threshold:
                return True, f"Kumulative Drift: {max_drift:.1%}"

        # Trigger: periodic resample
        if tick % resample_interval == 0 and tick > 0:
            return True, f"Periodisch (alle {resample_interval} Ticks)"

        return False, ""

    def record_sample(self, monitor: MarketMonitor):
        """Record probabilities at time of sampling for drift detection."""
        self._last_sample_probs = {
            mid: state.current_probability
            for mid, state in monitor.markets.items()
        }
        self._resample_count += 1


# ═══════════════════════════════════════════════════════════════
# COMBINED STATE (written as JSON for the dashboard)
# ═══════════════════════════════════════════════════════════════

def build_combined_state(
    monitor: MarketMonitor,
    alert_engine: AlertEngine,
    active_alerts: list[Alert],
    all_alerts: list[Alert],
    scenario_stats: Optional[dict],
    stress_scenarios: Optional[list[dict]],
    poll_count: int,
    resample_count: int,
    last_resample_reason: str,
) -> dict:
    """Build the combined state JSON consumed by the integrated dashboard."""
    markets = []
    for mid, state in monitor.markets.items():
        markets.append({
            "market_id": state.market_id,
            "question": state.question,
            "disruption_type": state.disruption_type,
            "eu_relevance": state.eu_relevance,
            "current_probability": round(state.current_probability, 4),
            "previous_probability": round(state.previous_probability, 4),
            "probability_history": [round(p, 4) for p in state.probability_history],
            "all_time_high": round(state.all_time_high, 4),
            "all_time_low": round(state.all_time_low, 4),
            "change_1h": round(state.change_1h, 4),
            "trend": state.trend,
            "alert_level": state.alert_level,
        })

    return {
        "meta": {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "poll_count": poll_count,
            "resample_count": resample_count,
            "last_resample_reason": last_resample_reason,
            "markets_monitored": len(monitor.markets),
        },
        "markets": sorted(markets, key=lambda m: -m["current_probability"]),
        "alerts": {
            "active": [asdict(a) for a in active_alerts[-20:]],
            "history": [asdict(a) for a in all_alerts[-50:]],
            "total_count": len(all_alerts),
            "active_critical": sum(1 for a in active_alerts if a.level == "critical"),
            "active_warning": sum(1 for a in active_alerts if a.level == "warning"),
        },
        "scenarios": scenario_stats or {},
        "stress_scenarios": stress_scenarios or [],
        "statistics": {
            "avg_probability": round(float(np.mean([
                s.current_probability for s in monitor.markets.values()
            ])), 4) if monitor.markets else 0,
            "max_probability": round(float(max(
                (s.current_probability for s in monitor.markets.values()), default=0
            )), 4),
            "rising_count": sum(1 for s in monitor.markets.values() if s.trend == "rising"),
            "falling_count": sum(1 for s in monitor.markets.values() if s.trend == "falling"),
            "volatile_count": sum(1 for s in monitor.markets.values() if s.trend == "volatile"),
        },
    }


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(
    demo: bool = True,
    interval: int = 3,
    output_file: str = "pipeline_state.json",
    n_scenarios: int = 500,
    n_stress: int = 20,
    threshold_warn: float = 0.40,
    threshold_critical: float = 0.60,
    resample_interval: int = 10,
    max_ticks: int = 0,
    seed: int = 42,
):
    """
    Integrated pipeline: Monitor → Bridge → Sampler → Dashboard JSON.
    """
    monitor = MarketMonitor()
    alert_engine = AlertEngine(
        threshold_warn=threshold_warn,
        threshold_critical=threshold_critical,
    )
    bridge = ScenarioBridge()
    simulator = DemoSimulator(seed=seed) if demo else None

    all_alerts: list[Alert] = []
    active_alerts: list[Alert] = []
    poll_count = 0
    resample_count = 0
    last_resample_reason = "Initialisierung"
    live_mapper = None  # persistent EventMapper for live mode caching

    # Current scenario results
    scenario_stats: Optional[dict] = None
    stress_list: Optional[list[dict]] = None

    logger.info("=" * 60)
    logger.info("PROVIDER Integrated Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Mode:       {'DEMO' if demo else 'LIVE'}")
    logger.info(f"  Interval:   {interval}s")
    logger.info(f"  Scenarios:  {n_scenarios} (+ {n_stress} stress)")
    logger.info(f"  Resample:   every {resample_interval} ticks or on alert")
    logger.info(f"  Output:     {output_file}")
    logger.info("=" * 60)

    try:
        while True:
            poll_count += 1

            # ── 1. Poll markets ──
            if demo:
                updates = simulator.tick()
                for mid, question, dtype, relevance, prob in updates:
                    monitor.update(mid, prob, question, dtype, relevance)
            else:
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

            # ── 2. Evaluate alerts ──
            new_alerts = alert_engine.evaluate(monitor.markets, active_alerts)
            all_alerts.extend(new_alerts)
            active_alerts.extend(new_alerts)
            active_alerts = [a for a in active_alerts if not a.acknowledged]

            # ── 3. Decide whether to resample ──
            should_resample, reason = bridge.should_resample(
                monitor, new_alerts, poll_count, resample_interval
            )

            if should_resample or scenario_stats is None:
                if reason:
                    last_resample_reason = reason
                elif scenario_stats is None:
                    last_resample_reason = "Erster Lauf"

                # Convert monitor state to events
                events = bridge.monitor_to_events(monitor)

                # Run Monte Carlo
                sampler = ScenarioSampler(events, seed=seed + resample_count)
                scenario_set = sampler.sample(n_scenarios=n_scenarios)
                stress_set = sampler.sample_stress_scenarios(
                    n_scenarios=n_stress, min_severity=0.4
                )

                # Extract stats and top stress scenarios
                full_dict = scenario_set.to_dict()
                scenario_stats = {
                    "n_samples": n_scenarios,
                    "n_unique": full_dict["metadata"]["n_unique_scenarios"],
                    "severity": full_dict["statistics"].get("severity", {}),
                    "disruption_frequency": full_dict["statistics"].get("disruption_type_frequency", {}),
                    "parameter_distributions": full_dict["statistics"].get("parameter_distributions", {}),
                    "resample_tick": poll_count,
                    "resample_reason": last_resample_reason,
                }

                stress_list = [s.to_dict() for s in stress_set.scenarios[:10]]

                bridge.record_sample(monitor)
                resample_count += 1

                logger.info(
                    f"\n  [RESAMPLE #{resample_count}] Grund: {last_resample_reason} | "
                    f"Severity: mean={scenario_stats['severity'].get('mean', 0):.3f} "
                    f"p95={scenario_stats['severity'].get('p95', 0):.3f}"
                )

            # ── 4. Write combined state ──
            combined = build_combined_state(
                monitor, alert_engine, active_alerts, all_alerts,
                scenario_stats, stress_list,
                poll_count, resample_count, last_resample_reason,
            )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)

            # ── 5. Status line ──
            n_crit = combined["alerts"]["active_critical"]
            n_warn = combined["alerts"]["active_warning"]
            sev_mean = scenario_stats.get("severity", {}).get("mean", 0) if scenario_stats else 0
            status = (
                f"\rPoll #{poll_count:4d} | "
                f"Alerts: {n_crit}C {n_warn}W | "
                f"Resamples: {resample_count} | "
                f"Severity: {sev_mean:.3f} | "
                f"Rising: {combined['statistics']['rising_count']}"
            )
            print(status, end="", flush=True)

            for alert in new_alerts:
                icon = {"info": "i", "warning": "!", "critical": "X"}[alert.level]
                print(f"\n  [{icon}] {alert.message}", end="")

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
        logger.info("\nPipeline stopped.")

    print(f"\n\nFinal state: {output_file}")
    print(f"Polls: {poll_count} | Resamples: {resample_count} | Alerts: {len(all_alerts)}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PROVIDER Integrated Pipeline")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--interval", type=int, default=3)
    parser.add_argument("--output", type=str, default="pipeline_state.json")
    parser.add_argument("--scenarios", type=int, default=500)
    parser.add_argument("--stress", type=int, default=20)
    parser.add_argument("--threshold-warn", type=float, default=0.40)
    parser.add_argument("--threshold-critical", type=float, default=0.60)
    parser.add_argument("--resample-interval", type=int, default=10)
    parser.add_argument("--max-ticks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    run_pipeline(
        demo=args.demo, interval=args.interval, output_file=args.output,
        n_scenarios=args.scenarios, n_stress=args.stress,
        threshold_warn=args.threshold_warn,
        threshold_critical=args.threshold_critical,
        resample_interval=args.resample_interval,
        max_ticks=args.max_ticks, seed=args.seed,
    )


if __name__ == "__main__":
    main()
