# PROVIDER Scenario Generator — Prototyp, Architektur & Umsetzung

**Autor:** Andreas Schaefer (andreas.schaefer@implisense.com)
**Stand:** Februar 2026
**Version:** 3.0 (Interactive Risk Terminal)

---

## 1. Executive Summary

Der PROVIDER Scenario Generator ist ein System zur Quantifizierung und Simulation von Lieferkettenrisiken für die deutsche und europäische Wirtschaft. Er nutzt Echtzeit-Wahrscheinlichkeiten aus Prediction Markets (Polymarket) als Eingangssignal und generiert mittels Monte-Carlo-Simulation mit Gaussian Copula korrelierte Krisenszenarien, die direkt als Input für agentenbasierte Supply-Chain-Simulationen dienen können.

Das System besteht aus drei Schichten: einem Python-Backend für Datenerfassung und Szenariogenerierung (2.678 LOC), einer Echtzeit-Monitoring-Pipeline mit Alert-Engine, und einem interaktiven Browser-Terminal mit Command-Line-Interface, Scenario Builder und Korrelationsanalyse (5.038 LOC HTML/JS). Alle Komponenten sind als Self-Contained-Prototyp lauffähig — ohne externe Infrastruktur.

---

## 2. Motivation & Problemstellung

Lieferketten der deutschen Wirtschaft sind zunehmend geopolitischen, ökonomischen und ökologischen Schocks ausgesetzt. Bisherige Simulationsmodelle arbeiten meist mit statischen Szenarien oder Experteneinschätzungen. Der PROVIDER-Ansatz adressiert drei zentrale Defizite:

**Marktbasierte Wahrscheinlichkeiten statt Expertenmeinung:** Prediction Markets aggregieren die Einschätzungen tausender Teilnehmer mit echtem Geld. Polymarket verarbeitet Handelsvolumina von mehreren Millionen Dollar pro Markt — eine empirisch validierte Grundlage für Eintrittswahrscheinlichkeiten.

**Korrelierte Szenarien statt isolierter Events:** Ein geopolitischer Konflikt erhöht gleichzeitig Energiepreise, Handelsbarrieren und Transportkosten. Die Gaussian-Copula-Methode bildet diese Abhängigkeiten ab, ohne die individuellen Marginalverteilungen zu verzerren.

**Echtzeit-Adaptivität statt Einmal-Analyse:** Die Monitoring-Pipeline erkennt Schwellenüberschreitungen, Rapid Shifts und Kaskadenrisiken und löst automatische Resamplings aus — das Szenario-Set bleibt stets aktuell.

---

## 3. Systemarchitektur

### 3.1 Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATENQUELLEN                                  │
│  Polymarket Gamma API ──── Polymarket CLOB API ──── Demo-Modus  │
└──────────────┬──────────────────────┬────────────────────────────┘
               │                      │
               ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PYTHON BACKEND (2.678 LOC)                      │
│                                                                  │
│  ┌──────────────────┐   ┌──────────────────┐                    │
│  │ PolymarketClient │──▶│   EventMapper    │                    │
│  │  (API-Zugriff)   │   │ (Klassifikation) │                    │
│  └──────────────────┘   └────────┬─────────┘                    │
│                                   │                              │
│                                   ▼                              │
│                    ┌──────────────────────────┐                  │
│                    │   ScenarioSampler        │                  │
│                    │ (Gaussian Copula MC)     │                  │
│                    └──────────┬───────────────┘                  │
│                               │                                  │
│  ┌──────────────────┐        │        ┌──────────────────┐      │
│  │  MarketMonitor   │◀───────┘───────▶│  ScenarioBridge  │      │
│  │  (Alert Engine)  │                 │ (Resample-Logic) │      │
│  └──────────────────┘                 └──────────────────┘      │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼  JSON State
┌─────────────────────────────────────────────────────────────────┐
│              BROWSER FRONTEND (5.038 LOC)                        │
│                                                                  │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────────────┐    │
│  │  Dashboard   │  │ Early Warning │  │ Interactive Terminal│    │
│  │  (statisch)  │  │  (live sim.)  │  │ (Command-Line+     │    │
│  │             │  │              │  │  Scenario Builder)  │    │
│  └─────────────┘  └───────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Datenfluss

Der Datenfluss folgt einer fünfstufigen Pipeline:

**Phase 1 — Event Harvesting:** Der PolymarketClient fragt die Gamma Markets API (Marktentdeckung) und CLOB API (Preise/Historie) ab. Beide APIs sind read-only ohne Authentifizierung nutzbar. Rate-Limiting (0.5s/Request) und automatisches Retry bei HTTP 429 schützen vor Sperrung.

**Phase 2 — Risk Mapping:** Der EventMapper klassifiziert entdeckte Märkte mittels 55 Regex-Regeln in 10 Disruptions-Typen und berechnet EU-Relevanz-Scores. Ein dualer Discovery-Ansatz kombiniert kategorie-basierte Suche (4 Polymarket-Tags) mit 33 freien Suchanfragen.

**Phase 3 — Scenario Sampling:** Der ScenarioSampler erzeugt korrelierte Szenarien via Gaussian Copula. Pro Resample werden 500–1.000 Samples generiert, Stress-Szenarien durch 10-faches Oversampling extrahiert.

**Phase 4 — Monitoring & Triggering:** Die Pipeline überwacht kontinuierlich Marktbewegungen und löst Resamplings aus bei kritischen Alerts, Drift >5% oder periodisch alle 8–10 Ticks.

**Phase 5 — Visualisierung:** Der kombinierte State (Märkte + Alerts + Szenarien) wird als JSON exportiert und von den Browser-Dashboards konsumiert — oder direkt im Browser per In-Browser Monte Carlo simuliert.

---

## 4. Python Backend — Module im Detail

### 4.1 polymarket_client.py (373 LOC)

Abstrakter HTTP-Client für zwei Polymarket-API-Schichten:

| API | Base-URL | Funktion | Auth |
|-----|----------|----------|------|
| Gamma Markets | `https://gamma-api.polymarket.com` | Markt-Discovery, Tags, Events | Keine |
| CLOB | `https://clob.polymarket.com` | Preise, Historie, Orderbook | Keine (Lesen) |

Kernmethoden: `get_tags()`, `get_events(tag_id)`, `search_markets(query)`, `get_price_history(token_id, interval)`. Die `_throttle()`-Methode erzwingt ein Minimum von 0.5 Sekunden zwischen API-Calls (ca. 7.200 Requests/Tag).

### 4.2 event_mapper.py (521 LOC)

Die zentrale Wissensbasis des Systems. Drei Datenschichten:

**Kategorien (PROVIDER_CATEGORIES):** 4 Polymarket-Tag-IDs — Geopolitics (100265), Finance (120), Politics (2), Tech (1401).

**Suchterme (PROVIDER_SEARCH_QUERIES):** 33 freie Suchanfragen, z.B. "EU tariff", "oil price shock", "semiconductor shortage", "Suez canal", "rare earth minerals".

**Klassifikationsregeln (KEYWORD_RULES):** 55 Regex-Patterns, die Marktfragen auf 10 Disruptions-Typen abbilden:

| Disruptions-Typ | Beispiel-Keywords | EU-Relevanz |
|-----------------|-------------------|-------------|
| Geopolitischer Konflikt | russia, ukraine, nato, invasion | 0.9–1.0 |
| Energieversorgung | oil, gas, opec, pipeline, energy | 0.8–1.0 |
| Handelsrestriktionen | tariff, sanction, trade war, export ban | 0.8–1.0 |
| Transportstörung | shipping, suez, red sea, port | 0.7–0.9 |
| Agrarschock | crop, grain, soy, drought, food | 0.7–0.9 |
| Finanzkrise | recession, default, bank, credit | 0.8–1.0 |
| Pandemie | pandemic, virus, WHO, lockdown | 0.9 |
| Klimaereignis | hurricane, flood, wildfire, climate | 0.6–0.8 |
| Technologie-Disruption | semiconductor, rare earth, chip, AI | 0.7–0.9 |
| Politische Instabilität | coup, election, protest, regime | 0.5–0.8 |

**Disruptions-Profile (DISRUPTION_PROFILES):** Jeder Typ ist mit konkreten Supply-Chain-Parametern hinterlegt — z.B. verursacht ein geopolitischer Konflikt: trade_corridor -50%, sanctions_impact -25%, uncertainty_index 2.0×, energy_supply_risk -30%.

### 4.3 scenario_sampler.py (427 LOC)

Implementiert die Gaussian-Copula-Methode für korrelierte Szenariogenerierung:

**Korrelationsmatrix:** 20 paarweise Korrelationen, z.B. Geopolitik↔Energie (ρ=0.7), Klima↔Agrar (ρ=0.7), Pandemie↔Transport (ρ=0.6). Unmapped-Paare erhalten ρ=0.05. Events gleichen Typs korrelieren mit ρ=0.8.

**Sampling-Algorithmus:**

1. Konstruiere NxN Korrelationsmatrix Σ
2. Cholesky-Zerlegung: Σ = LLᵀ (mit numerischer Stabilisierung via Eigenvalue-Check)
3. Ziehe unabhängige Standardnormalvariablen: Z ~ N(0, I)
4. Korreliere: Z_corr = L · Z
5. Transformiere in Uniform: U = Φ(Z_corr)
6. Threshold: Event i tritt ein wenn U_i < P_i (Markt-Wahrscheinlichkeit)
7. Aggregiere Parameter der getriggerten Events

**Parameter-Aggregation** differenziert nach Einheitentyp: Multiplikatoren (×) werden multipliziert, Prozentänderungen (%) addiert (Cap bei -95%), Tage (d) per Maximum.

**Severity-Score:** Komposit-Metrik [0, 1] aus normalisierten Parameter-Magnitudes + Kaskaden-Bonus (bis +0.3 für Multi-Typ-Szenarien).

### 4.4 monitor.py (529 LOC)

Echtzeit-Marktüberwachung mit 5 Alert-Typen:

| Alert-Typ | Auslöser | Level |
|-----------|----------|-------|
| THRESHOLD_CROSSED | P > 40% (Warn) oder P > 60% (Krit.) | Warning / Critical |
| RAPID_SHIFT | ΔP > 10% in einem Polling-Intervall | Warning |
| NEW_HIGH | Neues Allzeithoch (nach 5+ Beobachtungen) | Info |
| CASCADE_RISK | ≥3 korrelierte Märkte im gleichen Typ steigen | Critical |
| VOLUME_SPIKE | Handelsvolumen-Sprung | Warning |

Der DemoSimulator erzeugt realistische Wahrscheinlichkeitsbewegungen via Mean-Reverting Random Walk mit 5% Jump-Chance und korrelierten Schocks (z.B. Ukraine-Bewegung → Energiemarkt-Reaktion).

### 4.5 pipeline.py (422 LOC)

Die integrierte Pipeline verbindet Monitor und Sampler:

**ScenarioBridge:** Konvertiert MarketMonitor-State → MappedEvents für den ScenarioSampler. Entscheidet autonom über Resample-Zeitpunkte.

**Resample-Trigger:** Kritischer Alert → sofort. Kumulativer Drift ≥5% → sofort. Periodisch alle 10 Ticks. Jeder Resample erhält einen Reason-String für Audit-Trail.

**Output:** Kombinierter JSON-State mit Markets, Alerts, Severity-Verteilung, Parameter-Distributions und Top-10 Stress-Szenarien — konsumierbar durch alle Dashboard-Varianten.

### 4.6 main.py (392 LOC)

CLI-Einstiegspunkt für Batch-Generierung. Demo-Modus mit 12 realistischen Mock-Events:

| Event | P(Ja) | Typ | EU-Rel. |
|-------|-------|-----|---------|
| Russia-Ukraine Ceasefire 2026 | 44% | Geopolitik | 1.0 |
| China-Taiwan Invasion vor 2027 | 13% | Geopolitik | 0.9 |
| 30% EU-Zoll bis Aug. 2026 | 35% | Handel | 1.0 |
| US-Rezession bis Ende 2026 | 25% | Finanzen | 0.9 |
| Neue WHO-Pandemie 2026 | 8% | Pandemie | 0.9 |
| Ölpreis > $80 bis Juni 2026 | 40% | Energie | 0.9 |
| Naturkatastrophe > $50B 2026 | 30% | Klima | 0.7 |
| Ukraine Rare-Earth-Deal vor April | 55% | Tech | 0.8 |
| Brasilien Agrar-Exportrestriktionen | 12% | Agrar | 0.9 |
| Red Sea Shipping Disruptions 2026 | 60% | Transport | 0.9 |
| EZB-Zinsen unter 2% bis Ende 2026 | 35% | Finanzen | 1.0 |
| Argentinien Sovereign Default 2026 | 15% | Polit. Stab. | 0.7 |

---

## 5. Browser Frontend — Dashboard-Evolution

Das Frontend durchlief fünf Iterationsstufen, die jeweils auf Nutzerfeedback basierten:

### 5.1 Statisches Szenario-Dashboard (provider_dashboard.html)

**916 KB | 1.126 Zeilen | 22 JS-Funktionen**

Erste Visualisierung der Batch-generierten Szenarien. Enthält die vollständigen Ergebnisdaten (1.000 Szenarien, 430 Kombinationen) als eingebetteten JSON-Block. Vier Chart.js-Diagramme (Severity-Histogramm, Disruptions-Häufigkeit, Event-Wahrscheinlichkeiten, Parameter-Impact), ein Scenario Explorer mit Keyboard-Navigation (Pfeiltasten, R für Random, W für Worst-Case) und eine sortierbare Event-Inventar-Tabelle.

### 5.2 Live Early Warning Dashboard (provider_early_warning.html)

**47 KB | 1.397 Zeilen | 20 JS-Funktionen**

Übergang zu Echtzeit: Marktbewegungen werden per DemoSimulator live generiert, Alerts erscheinen im Feed, Sparklines zeigen Trendverläufe. Erste Integration des Scenario Panels mit Severity-Gauge, Impact-Bars und Stress-Strip. Der Tick-Override-Pattern (`const originalTick = tick; tick = function() { ... }`) verbindet Monitor und Scenario Engine.

### 5.3 Kompaktes Dashboard (provider_early_warning_compact.html)

**36 KB | 643 Zeilen | 18 JS-Funktionen**

Redesign auf Informationsdichte: Markt-Karten werden durch eine Datentabelle ersetzt, KPIs wandern inline in den Header, Alert-Banner wird zum schmalen Ticker, Disruptions werden als Chips statt Balken dargestellt. 54% weniger Code bei gleicher Funktionalität.

### 5.4 Deutsche Bank Terminal (provider_terminal_db.html)

**43 KB | 843 Zeilen | 18 JS-Funktionen**

Minimalistisches Terminal-Design inspiriert von institutionellen Trading-Terminals. Farbpalette: Deep Navy (#000a18 → #001e3c), DB-Blau (#0038a8), Cyan-Akzent (#00d2ff). Scanline-Overlay für Terminal-Ästhetik. Dreistufiger Critical-Fokus: Pulsierendes Banner (oben), glühende Tabellenzeilen (Mitte), Stress-Cards mit Box-Shadow-Glow (unten).

### 5.5 Interactive Risk Terminal (provider_terminal_interactive.html)

**64 KB | 1.030 Zeilen | 47 JS-Funktionen**

Finales Terminal mit vier interaktiven Subsystemen:

**Command-Line (Bloomberg-Style):** Goldene Eingabeleiste mit Tab-Autocomplete, Pfeil-↑↓ History und 13 Befehlen:

| Befehl | Funktion |
|--------|----------|
| `/help` | Befehlsübersicht |
| `/focus <markt>` | Drill-Down auf einzelnen Markt |
| `/whatif <markt> <prob>` | What-If-Analyse: setzt Wahrscheinlichkeit und öffnet Builder |
| `/scenario` | Scenario Builder öffnen |
| `/stress` | Stress-Test: alle Märkte +20% |
| `/corr` | Korrelationsmatrix als Heatmap |
| `/filter crit\|warn\|all` | Alert-Feed filtern |
| `/speed 1\|3\|10` | Simulationsgeschwindigkeit |
| `/export` | JSON-Export aller Szenariodaten |
| `/pause` / `/resume` | Simulation pausieren/fortsetzen |
| `/clear` | Alert-Feed leeren |
| `/reset` | Simulation zurücksetzen |
| `/<markt>` | Direkt als Markt-Shortcut (z.B. `/ukraine`) |

Markt-Aliase: `ukraine`, `taiwan`, `oil`, `ecb`, `red`, `brazil`, `pandemic`, `recession`, `tariff`, `disaster`, `rare`, `argentina`.

**Scenario Builder:** Interaktive What-If-Analyse. Nutzer wählt Events per Checkbox, passt Wahrscheinlichkeiten per Slider (1–99%), klickt "SIMULATE" — eigenständiger Monte-Carlo-Lauf (n=500) liefert sofortige Ergebnisse. Vergleich "Builder vs. Live-Baseline" zeigt Severity-Delta farbcodiert (rot = verschlechtert, grün = verbessert).

**Market Drill-Down:** Klick auf Tabellenzeile oder `/focus` öffnet Detail-Panel mit Chart.js-Liniendiagramm (History mit Warn/Crit-Schwellenlinien), Supply-Chain-Parametern des Disruptions-Typs und Korrelationsstärken zu allen anderen Typen als Mini-Bars.

**Korrelations-Heatmap:** 10×10 Matrix aller Disruptions-Typen als Fullscreen-Overlay. Farbintensität proportional zur Korrelationsstärke. Hover zeigt exakte Werte.

---

## 6. Mathematisches Modell

### 6.1 Gaussian Copula für korrelierte Events

Gegeben: N Events mit individuellen Eintrittswahrscheinlichkeiten p_1, ..., p_N aus Polymarket.

Gesucht: Korrelierte binäre Realisierungen (Event tritt ein / tritt nicht ein), die sowohl die individuellen Wahrscheinlichkeiten als auch die Abhängigkeitsstruktur respektieren.

Methode:

1. Definiere Korrelationsmatrix Σ ∈ R^(N×N) mit Σ_ij = ρ(type_i, type_j)
2. Cholesky-Zerlegung: Σ = L · Lᵀ
3. Für jedes Szenario s = 1, ..., S:
   - Ziehe Z ~ N(0, I_N) (unabhängige Standardnormale)
   - Korreliere: Z_corr = L · Z
   - Transformiere: U_i = Φ(Z_corr_i) (CDF der Standardnormalen)
   - Trigger: Event i tritt ein ⟺ U_i < p_i

Ergebnis: Die marginale Triggerwahrscheinlichkeit jedes Events entspricht exakt p_i, während die Joint-Verteilung die gewünschte Korrelationsstruktur aufweist.

### 6.2 Severity-Scoring

Der Severity-Score eines Szenarios ist ein gewichteter Komposit:

```
severity = min(1, mean(normalized_params) + cascade_bonus)
```

Wobei die Normalisierung einheitenspezifisch ist: Multiplikatoren (×) als `(|val|-1)/2`, Tage (d) als `val/30`, Prozent (%) als `|val|/50`. Der Cascade-Bonus addiert `min(n_types × 0.05, 0.3)` für Multi-Typ-Szenarien.

### 6.3 Resample-Trigger-Logik

Die Pipeline entscheidet autonom über Resample-Zeitpunkte:

```
resample ← (erster Lauf)
         ∨ (kritischer Alert in letzten 2 Ticks)
         ∨ (max_drift ≥ 5% seit letztem Sample)
         ∨ (Ticks seit letztem Resample ≥ 8)
```

Der Drift wird als Maximum der absoluten Wahrscheinlichkeitsänderungen über alle Märkte berechnet.

---

## 7. Polymarket API-Integration

### 7.1 Drei API-Schichten

| API | Funktion | Authentifizierung | Nutzung in PROVIDER |
|-----|----------|-------------------|---------------------|
| Gamma Markets API | Markt-Discovery, Tags, Events, Suche | Keine | Vollständig |
| CLOB API | Live-Preise, Preishistorie, Orderbook | Keine (Lesen) | Preise + Historie |
| Data API (Polygon) | Wallet-bezogene Daten, Positionen | API-Key + Signatur | Nicht genutzt |

### 7.2 Relevante Endpunkte

- `GET /tags` — Alle verfügbaren Kategorien (inkl. tag_id)
- `GET /events?tag_id=X` — Events einer Kategorie mit Märkten
- `GET /markets?_q=X` — Freitext-Suche über alle Märkte
- `GET /prices-history?market=X&interval=Y` — Historische Preiszeitreihen
- `GET /book?token_id=X` — Aktuelles Orderbook (Bid/Ask/Spread)

### 7.3 Geo-Restriktion

Polymarket blockiert Trading aus Deutschland/EU. Die API ist jedoch read-only zugänglich — keine Wallet oder Handelsfunktion wird benötigt. Der Client nutzt ausschließlich öffentliche GET-Endpunkte.

---

## 8. Projektstruktur & Dateien

### 8.1 Python Backend

```
terminal/
├── __init__.py                 14 LOC    Package-Init, Version 0.1.0
├── polymarket_client.py       373 LOC    API-Client (Gamma + CLOB)
├── event_mapper.py            521 LOC    Klassifikation, 55 Regex-Regeln
├── scenario_sampler.py        427 LOC    Monte Carlo, Gaussian Copula
├── main.py                    392 LOC    CLI, Batch-Pipeline, Demo-Modus
├── monitor.py                 529 LOC    Echtzeit-Monitor, Alert Engine
├── pipeline.py                422 LOC    Integrierte Pipeline, Bridge
└── requirements.txt                      requests, numpy, scipy, pandas
                              ─────
                        Total: 2.678 LOC
```

### 8.2 Browser Frontend

```
├── provider_dashboard.html              916 KB   Statisches Szenario-Dashboard
├── provider_early_warning.html           47 KB   Live Early Warning
├── provider_early_warning_compact.html   36 KB   Kompakte Variante
├── provider_terminal_db.html             43 KB   Deutsche Bank Terminal
├── provider_terminal_interactive.html    64 KB   Interaktives Risk Terminal
                                        ─────
                                  Total: ~1.1 MB, 5.038 LOC
```

### 8.3 Generierte Outputs

```
scenarios_output/
├── events_20260209_162527.json          18 KB   12 Input-Events
├── scenarios_20260209_162527.json     1.084 KB   1.000 Szenarien, 430 Kombinationen
├── scenarios_20260209_162527.csv       162 KB   Flache Tabelle (1.000 × 36)
└── stress_scenarios_20260209_162527.json 106 KB   50 Stress-Szenarien (Sev ≥ 0.74)
                                       ─────
                                 Total: ~1.4 MB
```

---

## 9. Technische Highlights

**Gaussian Copula im Browser:** Die vollständige Cholesky-Zerlegung und korrelierte Normalverteilungs-Sampling läuft in reinem JavaScript — inkl. Box-Muller-Transform (`randn()`) und Error-Function-Approximation (`erf()`). 500 Szenarien werden pro Resample in <50ms generiert.

**Dual-Discovery-Strategie:** Kombiniert kategorie-basierte Suche (4 Polymarket-Tags mit bekannten IDs) und 33 freie Suchterme mit automatischer Deduplizierung. Maximiert die Abdeckung relevanter Märkte.

**Bilinguales System:** Alle Disruptions-Typen, Alert-Messages und UI-Elemente sind in Deutsch und Englisch verfügbar. Die Terminal-UI nutzt deutsche Labels, die API-Kommunikation englische.

**Self-Contained HTML:** Jedes Dashboard ist eine einzelne HTML-Datei ohne Build-Prozess. Externe Abhängigkeit einzig auf Chart.js via CDN. Sofort im Browser öffenbar.

**Command-Line-Interface:** 13 Befehle mit Tab-Autocomplete, Command-History (↑↓) und Markt-Aliase. Wandelt das Dashboard von einem Beobachtungs-Tool zu einem interaktiven Analyse-Werkzeug.

**Adaptives Resampling:** Die Pipeline resampelt nicht periodisch mit festem Intervall, sondern reagiert auf tatsächliche Marktbewegungen (Drift-basiert) und kritische Ereignisse (Alert-basiert). Jeder Resample wird mit Reason-String protokolliert.

---

## 10. Einschränkungen & nächste Schritte

### 10.1 Aktuelle Einschränkungen

- **Demo-Modus:** Der Live-API-Zugriff auf Polymarket wurde noch nicht gegen die produktive API getestet (Proxy-Einschränkungen in der Entwicklungsumgebung)
- **Statische Korrelationen:** Die paarweisen Korrelationen sind manuell definiert, nicht aus historischen Daten geschätzt
- **Keine Persistenz:** Die Browser-Terminals halten keinen State über Seitenreloads hinweg
- **Einzelnutzer:** Kein Multi-User-Support, keine Zugriffskontrolle

### 10.2 Mögliche Erweiterungen

- **Portfolio/Exposure-View:** Nutzer definiert seine spezifische Lieferkette und sieht personalisierte Risikobewertung
- **Supply-Chain-Graph:** Netzwerk-Visualisierung der Schock-Propagation durch die Kette
- **Timeline/Playback:** Zurückspulen und Krisen-Aufbau als Animation abspielen
- **Resilienz-Score:** Aggregierter Score mit konkreten Diversifikations-Empfehlungen
- **Historische Korrelationsschätzung:** Paarweise Korrelationen aus tatsächlichen Polymarket-Preishistorien ableiten
- **Integration:** Anbindung an agentenbasierte Supply-Chain-Simulationsmodelle als Szenario-Input

---

## 11. Abhängigkeiten & Deployment

### Python

```
requests>=2.31       HTTP-Client
numpy>=1.24          Numerik, Cholesky
scipy>=1.11          Statistik, Normalverteilung
pandas>=2.0          DataFrame-Export (CSV)
```

### JavaScript (CDN)

```
Chart.js 4.5.1       Diagramme, Gauge-Charts
JetBrains Mono       Terminal-Schrift (Google Fonts)
```

### Deployment

```bash
# Backend starten (Demo-Modus)
cd 04_Apps
python -m terminal.main --demo --scenarios 1000 --stress 50

# Pipeline starten (Live-Monitoring)
python pipeline.py --demo --interval 3 --scenarios 500

# Frontend öffnen
open provider_terminal_interactive.html
```

Kein Build-Prozess, kein Server, keine Datenbank erforderlich.

---

## 12. Zusammenfassung

| Kennzahl | Wert |
|----------|------|
| Python Backend | 2.678 LOC, 6 Module |
| Browser Frontend | 5.038 LOC, 5 Dashboard-Varianten |
| Disruptions-Typen | 10 |
| Klassifikations-Regeln | 55 Regex-Patterns |
| Korrelations-Paare | 20 |
| Demo-Events | 12 (basierend auf echten Polymarket-Märkten) |
| Monte-Carlo-Samples | 500–1.000 pro Lauf |
| Alert-Typen | 5 |
| Terminal-Befehle | 13 |
| Externe Abhängigkeiten | 4 Python-Packages, 1 JS-Library |
| Build-Prozess | Keiner |

Der PROVIDER Scenario Generator demonstriert, dass Prediction-Market-Daten als quantitative Grundlage für Supply-Chain-Risikobewertung genutzt werden können — in Echtzeit, korreliert und interaktiv explorierbar.
