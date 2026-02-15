# PROVIDER Terminal

Monte-Carlo-basierter Szenario-Generator fuer Lieferketten-Disruption. Nutzt [Polymarket](https://polymarket.com)-Prediction-Markets als Echtzeit-Wahrscheinlichkeitsquelle fuer geopolitische, wirtschaftliche und klimatische Risiken.

Teil des [PROVIDER-Verbundprojekts](https://github.com/as311-ops) zur proaktiven Versorgungssicherheit (AP3/AP4).

## Features

- **Automatische Event-Klassifikation** -- Polymarket-Events werden ueber ~40 Regex-Patterns in 10 Disruptions-Typen eingeordnet (Energie, Handel, Transport, Agrar, Klima u.a.)
- **Gaussian-Copula Monte-Carlo-Sampling** -- Korrelierte Risikoverteilungen mit konfigurierbarer Korrelationsmatrix
- **Echtzeit-Monitor** -- Polling-basierte Marktueberwachung mit 5 Alert-Typen (Schwellenwerte, Rapid Shifts, Kaskaden-Risiken, Volume Spikes)
- **Pipeline-Modus** -- Kombiniert Monitor + Sampler: automatisches Resampling bei Marktveraenderungen
- **Web-Dashboard** -- Live-Visualisierung des Pipeline-Status
- **Demo-Modus** -- Vollstaendig offline mit 12 Dummy-Maerkten, deterministisch reproduzierbar

## Voraussetzungen

- Python 3.10+
- Keine API-Keys oder Umgebungsvariablen noetig

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verwendung

Alle Befehle werden vom uebergeordneten Verzeichnis (`04_Apps/`) ausgefuehrt:

```bash
# Demo-Modus (keine API, 12 Dummy-Maerkte)
python -m terminal.main --demo

# Live-Modus (Polymarket API)
python -m terminal.main

# Monitor (Echtzeit-Polling mit Alerts)
python -m terminal.monitor --demo --max-ticks 10 --interval 1

# Pipeline (Monitor + Szenario-Generator kombiniert)
python -m terminal.pipeline --demo --max-ticks 10 --interval 1

# Dashboard
cd terminal && python3 -m http.server 8080
# -> http://localhost:8080/provider_dashboard.html
```

### Erweiterte Optionen

```bash
python -m terminal.main --demo \
    --scenarios 5000 \    # Anzahl Monte-Carlo-Szenarien
    --stress 200 \        # Anzahl Stress-Szenarien
    --seed 42 \           # Deterministischer RNG-Seed
    --min-volume 50000 \  # Mindest-Handelsvolumen
    --min-relevance 0.7 \ # Mindest-Relevanz-Score
    --output ./out \      # Output-Verzeichnis
    --skip-search \       # Keyword-Suche ueberspringen
    -v                    # Verbose-Logging
```

## Architektur

```
PolymarketClient (API)  --->  EventMapper (Klassifikation + Skalierung)
        |                            |
        v                            v
  Market/Event               MappedEvent (Cache + Price-Refresh)
                                     |
                                     v
                           ScenarioSampler (Gaussian Copula)
                                     |
                                     v
                              ScenarioSet (JSON/CSV Export)
```

### Module

| Modul | Beschreibung |
|---|---|
| `main.py` | CLI-Einstiegspunkt, Einmal-Lauf |
| `polymarket_client.py` | API-Client mit Exponential Backoff |
| `event_mapper.py` | Klassifikation + Parameter-Scaling |
| `scenario_sampler.py` | Gaussian-Copula Monte-Carlo-Sampling |
| `monitor.py` | Echtzeit-Polling mit Alert-System |
| `pipeline.py` | Monitor + Sampler kombiniert |
| `config_loader.py` | YAML/Default-Konfiguration |
| `metrics.py` | Request-Metriken (Latenz, Fehlerrate) |

### Disruptions-Typen

`energy_supply` | `trade_restriction` | `transport_disruption` | `agricultural_shock` | `financial_crisis` | `geopolitical_conflict` | `pandemic` | `climate_event` | `tech_disruption` | `political_instability`

## Output

Standard-Verzeichnis: `scenarios_output/`

| Datei | Inhalt |
|---|---|
| `events_*.json` | Erkannte und klassifizierte Events |
| `scenarios_*.json` | Monte-Carlo-Szenario-Set |
| `scenarios_*.csv` | Tabellarischer Export |
| `stress_scenarios_*.json` | Extremszenarien |

Monitor und Pipeline schreiben zusaetzlich `monitor_state.json` bzw. `pipeline_state.json` ins Arbeitsverzeichnis.

## Konfiguration

`config.yaml` ist optional -- alle Werte haben sinnvolle Defaults. Sektionen: `api`, `categories`, `search_queries`, `filtering`, `monitoring`, `sampling`, `correlations`.

## Lizenz

Dieses Projekt ist Teil des BMFTR-gefoerderten Verbundprojekts PROVIDER.
