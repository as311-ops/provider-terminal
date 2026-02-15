"""
PROVIDER Scenario Sampler
=========================
Monte Carlo scenario generation with correlation modeling.

Generates N scenario vectors where each scenario is a joint realization
of Polymarket events, respecting correlations between disruption types.

Key concepts:
    - Each mapped event has a probability p_i from Polymarket
    - Events within the same disruption cluster are positively correlated
    - A Gaussian copula models joint occurrence probabilities
    - Each scenario sample is a binary vector (event occurs / doesn't occur)
    - Parameters are then aggregated across triggered events
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from .event_mapper import MappedEvent, DISRUPTION_TYPES

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CORRELATION STRUCTURE
# ═══════════════════════════════════════════════════════════════

# Pairwise correlation between disruption types (symmetric).
# These represent "if type A occurs, how likely is type B to co-occur?"
# Based on historical supply chain crisis analysis.
DISRUPTION_CORRELATIONS = {
    ("geopolitical_conflict", "energy_supply"):        0.7,
    ("geopolitical_conflict", "trade_restriction"):    0.6,
    ("geopolitical_conflict", "transport_disruption"): 0.5,
    ("geopolitical_conflict", "financial_crisis"):     0.4,
    ("geopolitical_conflict", "agricultural_shock"):   0.4,

    ("energy_supply", "trade_restriction"):            0.3,
    ("energy_supply", "financial_crisis"):             0.5,
    ("energy_supply", "transport_disruption"):         0.3,

    ("trade_restriction", "financial_crisis"):         0.5,
    ("trade_restriction", "agricultural_shock"):       0.3,

    ("pandemic", "transport_disruption"):              0.6,
    ("pandemic", "financial_crisis"):                  0.5,
    ("pandemic", "agricultural_shock"):                0.3,

    ("climate_event", "agricultural_shock"):           0.7,
    ("climate_event", "transport_disruption"):         0.4,
    ("climate_event", "energy_supply"):                0.3,

    ("tech_disruption", "trade_restriction"):          0.4,
    ("tech_disruption", "geopolitical_conflict"):      0.3,

    ("political_instability", "trade_restriction"):    0.5,
    ("political_instability", "agricultural_shock"):   0.4,
    ("political_instability", "energy_supply"):        0.3,

    ("financial_crisis", "agricultural_shock"):        0.2,
}


def _get_correlation(type_a: str, type_b: str) -> float:
    """Get correlation between two disruption types (order-independent)."""
    if type_a == type_b:
        return 1.0

    # Try config-based correlations first
    try:
        from .config_loader import get_config
        cfg_corr = get_config().correlations
        if cfg_corr:
            val = cfg_corr.get((type_a, type_b), cfg_corr.get((type_b, type_a)))
            if val is not None:
                return val
    except Exception:
        pass

    return DISRUPTION_CORRELATIONS.get(
        (type_a, type_b),
        DISRUPTION_CORRELATIONS.get((type_b, type_a), 0.05)  # minimal baseline
    )


# ═══════════════════════════════════════════════════════════════
# SCENARIO DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """A single sampled scenario for ORCA simulation."""
    id: int
    triggered_events: list[str]       # market IDs of events that occur
    triggered_questions: list[str]    # human-readable questions
    disruption_types: list[str]       # active disruption types
    parameters: dict[str, float]      # aggregated parameter name -> value
    severity_score: float             # 0-1 composite severity
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.id,
            "triggered_events": self.triggered_events,
            "triggered_questions": self.triggered_questions,
            "disruption_types": self.disruption_types,
            "parameters": {k: round(v, 4) for k, v in self.parameters.items()},
            "severity_score": round(self.severity_score, 4),
            "description": self.description,
        }


@dataclass
class ScenarioSet:
    """A complete set of sampled scenarios with metadata."""
    scenarios: list[Scenario]
    n_samples: int
    n_events: int
    seed: int
    event_summary: list[dict]  # input events used for generation

    def to_dict(self) -> dict:
        return {
            "metadata": {
                "n_samples": self.n_samples,
                "n_events": self.n_events,
                "seed": self.seed,
                "n_unique_scenarios": len(set(
                    tuple(s.triggered_events) for s in self.scenarios
                )),
            },
            "event_inputs": self.event_summary,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "statistics": self._compute_statistics(),
        }

    def _compute_statistics(self) -> dict:
        """Compute aggregate statistics across all scenarios."""
        if not self.scenarios:
            return {}

        severities = [s.severity_score for s in self.scenarios]

        # How often does each disruption type appear?
        dtype_counts: dict[str, int] = {}
        for s in self.scenarios:
            for dt in s.disruption_types:
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

        # Parameter distributions
        param_values: dict[str, list[float]] = {}
        for s in self.scenarios:
            for name, value in s.parameters.items():
                param_values.setdefault(name, []).append(value)

        param_stats = {}
        for name, values in param_values.items():
            arr = np.array(values)
            param_stats[name] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "p5": round(float(np.percentile(arr, 5)), 4),
                "p25": round(float(np.percentile(arr, 25)), 4),
                "p50": round(float(np.percentile(arr, 50)), 4),
                "p75": round(float(np.percentile(arr, 75)), 4),
                "p95": round(float(np.percentile(arr, 95)), 4),
            }

        return {
            "severity": {
                "mean": round(float(np.mean(severities)), 4),
                "std": round(float(np.std(severities)), 4),
                "p5": round(float(np.percentile(severities, 5)), 4),
                "p50": round(float(np.percentile(severities, 50)), 4),
                "p95": round(float(np.percentile(severities, 95)), 4),
            },
            "disruption_type_frequency": {
                dt: round(count / len(self.scenarios), 4)
                for dt, count in sorted(
                    dtype_counts.items(), key=lambda x: -x[1]
                )
            },
            "parameter_distributions": param_stats,
        }

    def save_json(self, path: str | Path):
        """Save scenario set to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.scenarios)} scenarios to {path}")

    def save_csv(self, path: str | Path):
        """Save scenario parameters as CSV (one row per scenario)."""
        import pandas as pd
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for s in self.scenarios:
            row = {
                "scenario_id": s.id,
                "severity_score": s.severity_score,
                "n_triggered": len(s.triggered_events),
                "disruption_types": "|".join(s.disruption_types),
            }
            row.update(s.parameters)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info(f"Saved scenario CSV to {path}")


# ═══════════════════════════════════════════════════════════════
# SCENARIO SAMPLER
# ═══════════════════════════════════════════════════════════════

class ScenarioSampler:
    """
    Monte Carlo scenario generator using Gaussian copula for
    correlated event sampling.

    Usage:
        sampler = ScenarioSampler(mapped_events, seed=42)
        scenario_set = sampler.sample(n_scenarios=1000)
        scenario_set.save_json("scenarios.json")
    """

    def __init__(
        self,
        events: list[MappedEvent],
        seed: int = 42,
    ):
        self.events = events
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Build correlation matrix
        self.n_events = len(events)
        self.corr_matrix = self._build_correlation_matrix()

    def _build_correlation_matrix(self) -> np.ndarray:
        """
        Build NxN correlation matrix for the Gaussian copula.
        Events of the same disruption type get intra-type correlation 0.8.
        Cross-type correlations come from DISRUPTION_CORRELATIONS.
        """
        n = self.n_events
        corr = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                type_i = self.events[i].disruption_type
                type_j = self.events[j].disruption_type

                if type_i == type_j:
                    rho = 0.8  # same-type events are highly correlated
                else:
                    rho = _get_correlation(type_i, type_j)

                corr[i, j] = rho
                corr[j, i] = rho

        # Ensure positive semi-definite (numerical stability)
        eigvals = np.linalg.eigvalsh(corr)
        if np.min(eigvals) < 0:
            corr += np.eye(n) * (abs(np.min(eigvals)) + 1e-6)
            # Re-normalize diagonal
            d = np.sqrt(np.diag(corr))
            corr = corr / np.outer(d, d)

        return corr

    def sample(self, n_scenarios: int = 1000) -> ScenarioSet:
        """
        Generate n_scenarios using Gaussian copula sampling.

        Process:
        1. Draw correlated normal samples Z ~ N(0, Sigma)
        2. Transform to uniform via Phi(Z)
        3. Event i triggers if U_i < p_i (Polymarket probability)
        4. Aggregate triggered event parameters
        """
        if self.n_events == 0:
            logger.warning("No events to sample from")
            return ScenarioSet(
                scenarios=[], n_samples=n_scenarios,
                n_events=0, seed=self.seed, event_summary=[]
            )

        # Step 1: Draw correlated normals
        mean = np.zeros(self.n_events)
        try:
            Z = self.rng.multivariate_normal(mean, self.corr_matrix, size=n_scenarios)
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not PSD, falling back to independent sampling")
            Z = self.rng.standard_normal(size=(n_scenarios, self.n_events))

        # Step 2: Transform to uniform [0, 1]
        U = stats.norm.cdf(Z)

        # Step 3: Threshold against probabilities
        probabilities = np.array([e.probability for e in self.events])
        triggered = U < probabilities  # shape: (n_scenarios, n_events)

        # Step 4: Build scenario objects
        scenarios = []
        for i in range(n_scenarios):
            active_indices = np.where(triggered[i])[0]
            scenario = self._build_scenario(i, active_indices)
            scenarios.append(scenario)

        event_summary = [e.to_dict() for e in self.events]

        return ScenarioSet(
            scenarios=scenarios,
            n_samples=n_scenarios,
            n_events=self.n_events,
            seed=self.seed,
            event_summary=event_summary,
        )

    def _build_scenario(self, scenario_id: int, active_indices: np.ndarray) -> Scenario:
        """Build a Scenario from a set of triggered event indices."""
        triggered_events = []
        triggered_questions = []
        disruption_types = set()
        aggregated_params: dict[str, list[float]] = {}

        for idx in active_indices:
            event = self.events[idx]
            triggered_events.append(event.market_id)
            triggered_questions.append(event.question)
            disruption_types.add(event.disruption_type)

            # Use base_impact (full effect) since the event DID trigger
            for param in event.parameters:
                name = param["name"]
                base = param["base_impact"]
                unit = param["unit"]
                aggregated_params.setdefault(name, []).append((base, unit))

        # Aggregate parameters: worst-case for multipliers, sum for additive
        final_params: dict[str, float] = {}
        for name, values in aggregated_params.items():
            impacts = [v[0] for v in values]
            unit = values[0][1]
            if unit == "multiplier":
                # Compound multipliers (multiplicative stacking)
                compound = 1.0
                for imp in impacts:
                    compound *= imp
                final_params[name] = compound
            elif unit == "percent_change":
                # Additive stacking, capped at -95%
                final_params[name] = max(sum(impacts), -95.0)
            elif unit == "days":
                # Take maximum delay
                final_params[name] = max(impacts)
            else:
                final_params[name] = sum(impacts)

        # Severity score: normalized measure of overall disruption
        severity = self._compute_severity(final_params, len(active_indices))

        return Scenario(
            id=scenario_id,
            triggered_events=triggered_events,
            triggered_questions=triggered_questions,
            disruption_types=sorted(disruption_types),
            parameters=final_params,
            severity_score=severity,
        )

    def _compute_severity(self, params: dict[str, float], n_triggered: int) -> float:
        """
        Compute composite severity score [0, 1].
        Combines parameter magnitudes with number of simultaneous disruptions.
        """
        if not params:
            return 0.0

        scores = []
        for name, value in params.items():
            if "multiplier" in name or name.endswith("_factor"):
                # Multiplier: 1.0=no effect, 3.0=severe
                score = min((value - 1.0) / 2.0, 1.0)
            elif "percent_change" in name or name.startswith("percent"):
                # Percent change: 0=no effect, -50+=severe
                score = min(abs(value) / 50.0, 1.0)
            elif "days" in name:
                score = min(value / 30.0, 1.0)
            else:
                score = min(abs(value) / 50.0, 1.0)
            scores.append(max(score, 0.0))

        if not scores:
            return 0.0

        # Average parameter severity + bonus for simultaneous disruptions
        base_severity = float(np.mean(scores))
        cascade_bonus = min(n_triggered * 0.05, 0.3)  # up to +0.3 for many simultaneous events
        return min(base_severity + cascade_bonus, 1.0)

    def sample_stress_scenarios(
        self,
        n_scenarios: int = 100,
        min_severity: float = 0.5,
    ) -> ScenarioSet:
        """
        Sample only high-severity stress scenarios.
        Useful for ATTACKER agent training in ORCA.
        Oversamples by 10x and filters to keep worst cases.
        """
        full_set = self.sample(n_scenarios=n_scenarios * 10)
        stress = [s for s in full_set.scenarios if s.severity_score >= min_severity]
        stress.sort(key=lambda s: s.severity_score, reverse=True)
        stress = stress[:n_scenarios]

        # Re-number
        for i, s in enumerate(stress):
            s.id = i

        return ScenarioSet(
            scenarios=stress,
            n_samples=n_scenarios,
            n_events=self.n_events,
            seed=self.seed,
            event_summary=full_set.event_summary,
        )
