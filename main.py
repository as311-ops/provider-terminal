#!/usr/bin/env python3
"""
PROVIDER Scenario Generator - Main Entry Point
================================================
Fetches Polymarket prediction data and generates supply chain
crisis scenarios for ORCA simulations.

Usage:
    # Full pipeline: discover events, map, sample, export
    python -m terminal.main

    # Quick test with mock data (no API calls)
    python -m terminal.main --demo

    # Custom parameters
    python -m terminal.main \
        --scenarios 5000 \
        --seed 123 \
        --min-volume 50000 \
        --output scenarios_output
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from .polymarket_client import PolymarketClient
from .event_mapper import EventMapper, MappedEvent, DISRUPTION_PROFILES
from .scenario_sampler import ScenarioSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("provider")


# ═══════════════════════════════════════════════════════════════
# DEMO DATA (for testing without API access)
# ═══════════════════════════════════════════════════════════════

def create_demo_events() -> list[MappedEvent]:
    """
    Create realistic demo events based on known Polymarket markets
    for testing the pipeline without API calls.
    """
    demo_markets = [
        {
            "market_id": "demo_ukraine_ceasefire",
            "question": "Russia-Ukraine ceasefire by end of 2026?",
            "probability": 0.44,
            "volume": 9_000_000,
            "disruption_type": "geopolitical_conflict",
            "eu_relevance": 1.0,
        },
        {
            "market_id": "demo_china_taiwan",
            "question": "Will China invade Taiwan before 2027?",
            "probability": 0.13,
            "volume": 5_000_000,
            "disruption_type": "geopolitical_conflict",
            "eu_relevance": 0.9,
        },
        {
            "market_id": "demo_eu_tariff_30",
            "question": "30% EU tariff in effect by August 2026?",
            "probability": 0.35,
            "volume": 3_000_000,
            "disruption_type": "trade_restriction",
            "eu_relevance": 1.0,
        },
        {
            "market_id": "demo_us_recession",
            "question": "US recession by end of 2026?",
            "probability": 0.25,
            "volume": 8_000_000,
            "disruption_type": "financial_crisis",
            "eu_relevance": 0.9,
        },
        {
            "market_id": "demo_pandemic_2026",
            "question": "New WHO-declared pandemic in 2026?",
            "probability": 0.08,
            "volume": 2_000_000,
            "disruption_type": "pandemic",
            "eu_relevance": 0.9,
        },
        {
            "market_id": "demo_oil_above_80",
            "question": "Oil price above $80/barrel by June 2026?",
            "probability": 0.40,
            "volume": 4_000_000,
            "disruption_type": "energy_supply",
            "eu_relevance": 0.9,
        },
        {
            "market_id": "demo_natural_disaster",
            "question": "Major natural disaster causes $50B+ damage in 2026?",
            "probability": 0.30,
            "volume": 1_500_000,
            "disruption_type": "climate_event",
            "eu_relevance": 0.7,
        },
        {
            "market_id": "demo_rare_earth_deal",
            "question": "Ukraine rare earth deal with US before April 2026?",
            "probability": 0.55,
            "volume": 2_000_000,
            "disruption_type": "tech_disruption",
            "eu_relevance": 0.8,
        },
        {
            "market_id": "demo_brazil_export_ban",
            "question": "Brazil imposes agricultural export restrictions in 2026?",
            "probability": 0.12,
            "volume": 500_000,
            "disruption_type": "agricultural_shock",
            "eu_relevance": 0.9,
        },
        {
            "market_id": "demo_red_sea_disruption",
            "question": "Red Sea shipping disruptions continue through 2026?",
            "probability": 0.60,
            "volume": 3_000_000,
            "disruption_type": "transport_disruption",
            "eu_relevance": 0.9,
        },
        {
            "market_id": "demo_ecb_rate_cut",
            "question": "ECB cuts rates below 2% by end of 2026?",
            "probability": 0.35,
            "volume": 2_500_000,
            "disruption_type": "financial_crisis",
            "eu_relevance": 1.0,
        },
        {
            "market_id": "demo_argentina_crisis",
            "question": "Argentina sovereign default in 2026?",
            "probability": 0.15,
            "volume": 1_000_000,
            "disruption_type": "political_instability",
            "eu_relevance": 0.7,
        },
    ]

    events = []
    for m in demo_markets:
        # Compute parameters from disruption profiles
        dtype = m["disruption_type"]
        prob = m["probability"]
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
                "probability": prob,
                "description": p.description,
                "affected_sectors": p.affected_sectors,
            })

        events.append(MappedEvent(
            market_id=m["market_id"],
            question=m["question"],
            probability=prob,
            volume=m["volume"],
            token_id=None,
            disruption_type=dtype,
            eu_relevance=m["eu_relevance"],
            parameters=params,
        ))

    return events


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(
    demo: bool = False,
    n_scenarios: int = 1000,
    n_stress: int = 100,
    seed: int = 42,
    min_volume: float = 10_000,
    min_eu_relevance: float = 0.5,
    output_dir: str = "scenarios_output",
    skip_search: bool = False,
):
    """Run the full PROVIDER scenario generation pipeline."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Discover and map events ──────────────────────
    if demo:
        logger.info("=== DEMO MODE: Using mock Polymarket data ===")
        mapped_events = create_demo_events()
    else:
        logger.info("=== Connecting to Polymarket API ===")
        try:
            client = PolymarketClient()
            mapper = EventMapper(
                client,
                min_volume=min_volume,
                min_eu_relevance=min_eu_relevance,
            )
            mapped_events = mapper.discover_and_map(
                use_categories=True,
                use_search=not skip_search,
            )
            if not mapped_events:
                logger.warning("Keine Events von API erhalten")
                logger.info("Fallback auf Demo-Events...")
                mapped_events = create_demo_events()
        except requests.exceptions.RequestException as e:
            logger.warning(f"API nicht erreichbar: {e}")
            logger.info("Fallback auf Demo-Events...")
            mapped_events = create_demo_events()

    logger.info(f"  {len(mapped_events)} PROVIDER-relevant events discovered")

    # Print event summary
    print("\n" + "=" * 70)
    print("PROVIDER EVENT INVENTORY")
    print("=" * 70)
    for i, event in enumerate(mapped_events, 1):
        print(f"\n  [{i:2d}] {event.question}")
        print(f"       Probability: {event.probability:.0%}  |  "
              f"Type: {event.disruption_type}  |  "
              f"EU-Relevanz: {event.eu_relevance:.0%}")
        print(f"       Volume: ${event.volume:,.0f}")
        if event.parameters:
            top_param = max(event.parameters, key=lambda p: abs(p["scaled_impact"]))
            print(f"       Top Impact: {top_param['name']} = "
                  f"{top_param['scaled_impact']:.2f} {top_param['unit']}")

    # Save event inventory
    events_file = out_path / f"events_{timestamp}.json"
    with open(events_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": timestamp,
                "mode": "demo" if demo else "live",
                "n_events": len(mapped_events),
                "events": [e.to_dict() for e in mapped_events],
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info(f"  Event inventory saved to {events_file}")

    # ── Phase 2: Monte Carlo sampling ─────────────────────────
    logger.info(f"\n=== Sampling {n_scenarios} scenarios (seed={seed}) ===")
    sampler = ScenarioSampler(mapped_events, seed=seed)

    # Regular scenarios (full probability distribution)
    scenario_set = sampler.sample(n_scenarios=n_scenarios)

    # Stress scenarios (worst-case filter)
    stress_set = sampler.sample_stress_scenarios(
        n_scenarios=n_stress,
        min_severity=0.4,
    )

    # ── Phase 3: Export results ───────────────────────────────
    scenarios_json = out_path / f"scenarios_{timestamp}.json"
    scenarios_csv = out_path / f"scenarios_{timestamp}.csv"
    stress_json = out_path / f"stress_scenarios_{timestamp}.json"

    scenario_set.save_json(scenarios_json)
    scenario_set.save_csv(scenarios_csv)
    stress_set.save_json(stress_json)

    # ── Phase 4: Print summary ────────────────────────────────
    stats = scenario_set.to_dict()["statistics"]

    print("\n" + "=" * 70)
    print(f"SCENARIO GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n  Total scenarios:  {n_scenarios}")
    print(f"  Stress scenarios: {len(stress_set.scenarios)}")
    print(f"  Input events:     {len(mapped_events)}")

    print(f"\n  Severity Distribution:")
    sev = stats.get("severity", {})
    print(f"    Mean:  {sev.get('mean', 0):.3f}")
    print(f"    P5:    {sev.get('p5', 0):.3f}  (optimistic)")
    print(f"    P50:   {sev.get('p50', 0):.3f}  (median)")
    print(f"    P95:   {sev.get('p95', 0):.3f}  (severe)")

    print(f"\n  Disruption Type Frequency (% of scenarios):")
    for dtype, freq in stats.get("disruption_type_frequency", {}).items():
        label = dtype.replace("_", " ").title()
        bar = "#" * int(freq * 40)
        print(f"    {label:30s} {freq:5.1%}  {bar}")

    print(f"\n  Key Parameter Statistics:")
    for param_name, pstats in list(stats.get("parameter_distributions", {}).items())[:8]:
        print(f"    {param_name:35s}  "
              f"mean={pstats['mean']:8.3f}  "
              f"p5={pstats['p5']:8.3f}  "
              f"p95={pstats['p95']:8.3f}")

    print(f"\n  Output Files:")
    print(f"    {scenarios_json}")
    print(f"    {scenarios_csv}")
    print(f"    {stress_json}")
    print("=" * 70)

    return scenario_set, stress_set


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PROVIDER Scenario Generator for ORCA Simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m terminal.main --demo
  python -m terminal.main --scenarios 5000 --seed 123
  python -m terminal.main --min-volume 100000 --output /data/scenarios
        """,
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use demo data (no API calls required)",
    )
    parser.add_argument(
        "--scenarios", type=int, default=1000,
        help="Number of Monte Carlo scenarios to generate (default: 1000)",
    )
    parser.add_argument(
        "--stress", type=int, default=100,
        help="Number of stress scenarios to extract (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--min-volume", type=float, default=10_000,
        help="Minimum market volume in USD to include (default: 10000)",
    )
    parser.add_argument(
        "--min-relevance", type=float, default=0.5,
        help="Minimum EU relevance score 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--output", type=str, default="scenarios_output",
        help="Output directory for generated files (default: scenarios_output)",
    )
    parser.add_argument(
        "--skip-search", action="store_true",
        help="Skip free-text search queries (faster, category-only)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        run_pipeline(
            demo=args.demo,
            n_scenarios=args.scenarios,
            n_stress=args.stress,
            seed=args.seed,
            min_volume=args.min_volume,
            min_eu_relevance=args.min_relevance,
            output_dir=args.output,
            skip_search=args.skip_search,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Netzwerkfehler: {e}\n"
            "  Hinweis: Nutzen Sie --demo fuer den Offline-Modus."
        )
        sys.exit(2)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
