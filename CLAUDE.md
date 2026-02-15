# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sprache

Kommunikation auf Deutsch, Code und Variablen auf Englisch.

## Projektueberblick

Monte-Carlo-basierter Szenario-Generator fuer Lieferketten-Disruptionsszenarien. Nutzt Polymarket-Prediction-Markets als Echtzeit-Wahrscheinlichkeitsquelle fuer geopolitische, wirtschaftliche und klimatische Risiken. Teil des PROVIDER-Verbundprojekts (AP3/AP4).

## Commands

```bash
# Installation (venv liegt in terminal/.venv)
source terminal/.venv/bin/activate  # vom 04_Apps-Verzeichnis aus
pip install -r requirements.txt

# Demo-Modus (keine API, 12 Dummy-Maerkte)
python -m terminal.main --demo

# Live-Modus (Polymarket API, kein Key noetig)
python -m terminal.main

# Monitor (Echtzeit-Polling mit Alerts)
python -m terminal.monitor --demo --max-ticks 10 --interval 1

# Pipeline (Monitor + Szenario-Generator kombiniert)
python -m terminal.pipeline --demo --max-ticks 10 --interval 1

# Dashboard testen (Server aus terminal/ starten!)
python3 -m http.server 8080
# -> http://localhost:8080/provider_dashboard.html

# Vollstaendige Optionen
python -m terminal.main --demo --scenarios 5000 --stress 200 --seed 42 \
    --min-volume 50000 --min-relevance 0.7 --output /custom/path --skip-search -v
```

**Wichtig:** Alle `python -m terminal.*` Befehle muessen vom `04_Apps/`-Verzeichnis aus ausgefuehrt werden (nicht aus `terminal/`), da `terminal` ein Package ist.

## Abhaengigkeiten

Python 3.10+ (Union-Syntax `X | Y` in Type Hints). Keine API-Keys oder Umgebungsvariablen noetig.

```
requests>=2.28    # Polymarket API
numpy>=1.24       # Korrelationsmatrizen, Sampling
scipy>=1.10       # Gaussian Copula, CDF
pandas>=2.0       # CSV-Export
pyyaml>=6.0       # YAML-Konfiguration
```

## Architektur

### Drei Betriebsmodi

1. **main.py** -- Einmal-Lauf: Events discovern, Monte Carlo samplen, JSON/CSV exportieren
2. **monitor.py** -- Polling-Schleife: Maerkte ueberwachen, Alerts generieren, `monitor_state.json` schreiben
3. **pipeline.py** -- Kombiniert Monitor + Sampler: Resampling bei Alerts/Drift, schreibt `pipeline_state.json` fuer Dashboard

### Daten-Pipeline

```
PolymarketClient (API)  --->  EventMapper (Klassifikation + Parameter-Scaling)
        |                            |
        v                            v
  Market/Event               MappedEvent (mit Cache + Price-Refresh)
                                     |
                                     v
                           ScenarioSampler (Gaussian Copula)
                                     |
                                     v
                              ScenarioSet (JSON/CSV Export)
```

### Kern-Datenstrukturen

- **MappedEvent** (event_mapper.py): Polymarket-Event -> Disruption-Typ + skalierte Supply-Chain-Parameter
- **Scenario** (scenario_sampler.py): Kombination getriggerter Events mit aggregierten Parametern und Severity-Score (0-1)
- **MarketState** (monitor.py): Getrackte Wahrscheinlichkeit mit History, Trend, Alert-Level, Volume-Tracking
- **Config** (config_loader.py): Dataclass-basiert, Singleton via `get_config()`

### 10 Disruption-Typen

`energy_supply`, `trade_restriction`, `transport_disruption`, `agricultural_shock`, `financial_crisis`, `geopolitical_conflict`, `pandemic`, `climate_event`, `tech_disruption`, `political_instability`

Jeder Typ hat 3-5 spezifische Supply-Chain-Parameter und Korrelationen zu anderen Typen.

### Alert-Typen (monitor.py)

`THRESHOLD_CROSSED`, `RAPID_SHIFT`, `NEW_HIGH`, `CASCADE_RISK`, `VOLUME_SPIKE`

## Konfiguration

`config.yaml` ist optional -- alle Werte haben Hardcoded-Defaults. Sektionen: `api`, `categories`, `search_queries`, `filtering`, `monitoring`, `sampling`, `correlations`. Der Config-Loader (`config_loader.py`) funktioniert auch ohne PyYAML.

Nicht in config.yaml konfigurierbar:
- **KEYWORD_RULES** (event_mapper.py): ~70 Regex-Patterns fuer Markt-Klassifikation
- **DISRUPTION_PROFILES** (event_mapper.py): Parameter-Definitionen pro Disruptionstyp

### Market Discovery

38 Polymarket-Kategorien (Tag-IDs) und 80 Suchbegriffe decken folgende Themenbereiche ab:

| Bereich | Kategorien | Beispiel-Tags |
|---------|-----------|---------------|
| Kern | 4 | Geopolitics, Finance, Politics, Tech |
| Energie | 5 | Crude Oil, Electricity, Uranium, Oil Futures |
| Handel & Zoelle | 3 | Trade, Tariff, Tariffs |
| Transport | 4 | Ports, ILA, Strikes, Houthi |
| Agrar | 1 | Food |
| Pandemie | 3 | H5N1, Pandemics, Diseases |
| Klima | 2 | Global Temp, Volcanic Eruption |
| Infrastruktur | 3 | Mining, CrowdStrike, Outage |
| Makro | 8 | GDP, ECB, FOMC, Rates, Gold/FX Futures |
| Regionen | 5 | EU, Germany, Panama, Japan, Sudan |

Live-Ergebnis (Feb 2025): ~1.200 Raw Markets -> ~194 gemappte Events nach Filterung.

## Fehlerbehandlung

- **Exponential Backoff** (polymarket_client.py): 3 Retries, 1s/2s/4s, fuer HTTP 429/5xx und Connection-Errors
- **Graceful Degradation**: Monitor/Pipeline arbeiten bei API-Fehlern mit bestehendem State weiter
- **Auto-Fallback**: `main.py` faellt bei 0 API-Events automatisch auf 12 Demo-Events zurueck
- **Event-Cache** (event_mapper.py): TTL-basiert (Standard 5 Min), Price-Only-Refresh zwischen Full Discoveries

## Observability

- **MetricsCollector** (metrics.py): Singleton, trackt Requests, Latenz (avg/p95), Fehlerrate
- **Dashboard** (provider_dashboard.html): Laedt `pipeline_state.json` per Fetch, Control-Bar mit Auto-Refresh

## Tests

Keine Unit-Tests. Validierung ueber Demo-Modus (`--demo --seed 42` ist deterministisch reproduzierbar).

## Output

Standard-Verzeichnis: `scenarios_output/`. Dateien: `events_*.json`, `scenarios_*.json`, `scenarios_*.csv`, `stress_scenarios_*.json`. Monitor/Pipeline schreiben `monitor_state.json` bzw. `pipeline_state.json` ins Arbeitsverzeichnis.
