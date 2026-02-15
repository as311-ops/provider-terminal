# PROVIDER -- Evaluationsstudie: Prediction Markets als Signalquelle

**Bewertung des Mehrwerts von Prediction Markets fuer die Supply-Chain-Fruehwarnung**

> Umsetzungsplan fuer Claude Code.
> Kontext: PROVIDER Scenario Generator (V3.0), Foerderantrag GVB 2025-10-29
> Bezug: AP1 (Szenarien), AP4 (LLM-Parametrisierung), AP8 (Evaluation)

---

## 1. Zielsetzung

Bevor Prediction Markets in die PROVIDER-Architektur integriert werden, muss belastbar geklaert werden:

1. **Abdeckung:** Wie viel Prozent der fuer PROVIDER relevanten Disruptions-Typen bilden Prediction Markets ab?
2. **Signalqualitaet:** Sind marktbasierte Wahrscheinlichkeiten besser kalibriert als NLP-basierte Schaetzungen?
3. **Kopplung:** Welchen Mehrwert bringt die Kombination beider Quellen gegenueber Einzelquellen?

Die Studie liefert ein publizierbare Ergebnis unabhaengig vom Ausgang und fuegt sich in die Evaluationsstruktur von AP8 ein.

---

## 2. Bestehendes Codebase

Der Scenario Generator (V3.0) liefert die technische Grundlage:

```
terminal/
├── polymarket_client.py       373 LOC    API-Client (Gamma + CLOB) -- WIEDERVERWENDBAR
├── event_mapper.py            521 LOC    55 Regex-Regeln, 10 Typen  -- ERWEITERBAR
├── scenario_sampler.py        427 LOC    Gaussian Copula MC         -- WIEDERVERWENDBAR
├── monitor.py                 529 LOC    Alert Engine               -- SPAETER
├── pipeline.py                422 LOC    Bridge, Resample-Logik     -- SPAETER
└── main.py                    392 LOC    CLI, Demo-Modus            -- ERWEITERBAR
```

Neue Module werden im Unterverzeichnis `evaluation/` entwickelt.

---

## 3. Phase 1 -- Abdeckungsanalyse

### 3.1 Ziel

Coverage-Matrix: Fuer welche PROVIDER-relevanten Risikotypen existieren Polymarket-Maerkte, wie liquide sind sie, und wie gut passt die Marktfrage zum Risikotyp?

### 3.2 Zielstruktur

```
terminal/
└── evaluation/
    ├── __init__.py
    ├── reference_catalog.py          Referenzkatalog relevanter Disruptions
    ├── coverage_scanner.py           Automatischer Abgleich Katalog vs. Polymarket
    ├── coverage_report.py            Auswertung und Report-Generierung
    ├── data/
    │   ├── reference_catalog.json    50-100 Risikotypen mit Metadaten
    │   └── scans/                    Zeitgestempelte Scan-Ergebnisse
    └── reports/
        └── coverage_matrix.csv       Ergebnis-Matrix
```

### 3.3 Teilschritte

#### Schritt 1: Referenzkatalog erstellen (`reference_catalog.py`)

Datenstruktur fuer einen einzelnen Risikotyp:

```python
@dataclass
class RiskType:
    id: str                          # z.B. "GEO-001"
    name_de: str                     # "Seltene-Erden-Exportbeschraenkungen China"
    name_en: str                     # "China rare earth export restrictions"
    disruption_type: str             # Einer der 10 PROVIDER-Typen aus event_mapper.py
    sector: str                      # KRITIS-Sektor oder Branche
    region: str                      # Betroffene Region
    eu_relevance: float              # 0.0-1.0
    search_terms: list[str]          # Suchbegriffe fuer Polymarket-Abgleich
    source: str                      # Herkunft (AP1-Szenario, Ontologie, Experte)
```

Quellen fuer den Katalog:
- Die 10 Disruptions-Typen aus `event_mapper.py` als Grundgeruest
- AP1-Szenarien (konjunkturell, steuerberatend, nachrichtenbasiert)
- PROVIDER-Ontologie (Disruptions-Taxonomie, sobald verfuegbar)
- CoyPu-KG Sektoren (Vorarbeit zu kritischen Lieferketten)
- KRITIS-Sektoren gemaess NIS2/KRITIS-Dachgesetz

Zielgroesse: 50-100 Risikotypen, verteilt ueber alle 10 Disruptions-Kategorien.

**Hinweis fuer Claude Code:** Der Katalog wird initial manuell kuratiert (JSON-Datei). Das Skript `reference_catalog.py` laedt, validiert und stellt Query-Methoden bereit (z.B. Filter nach Sektor, Disruptions-Typ, EU-Relevanz).

#### Schritt 2: Automatischer Polymarket-Scan (`coverage_scanner.py`)

Fuer jeden Eintrag im Referenzkatalog:

1. Nutze `polymarket_client.search_markets(query)` mit den `search_terms` des Risikotyps
2. Nutze `polymarket_client.get_events(tag_id)` fuer die 4 PROVIDER-Kategorien
3. Dedupliziere Ergebnisse (wie in `event_mapper.py` bereits implementiert)
4. Bewerte jeden gefundenen Markt:

```python
@dataclass
class MarketMatch:
    risk_type_id: str                # Referenz auf RiskType
    market_id: str                   # Polymarket Condition-ID
    market_question: str             # z.B. "Will the EU impose 30% tariffs..."
    match_quality: str               # "direct" | "proxy" | "weak"
    current_price: float             # 0.0-1.0 (Wahrscheinlichkeit)
    volume_usd: float                # Handelsvolumen in USD
    liquidity_score: float           # Abgeleitet aus Spread und Volume
    last_active: str                 # ISO-Timestamp letzter Trade
    notes: str                       # Manuelle Anmerkung (optional)
```

Klassifikation der Match-Qualitaet:
- **direct**: Marktfrage deckt den Risikotyp direkt ab (z.B. "EU tariff >30%" fuer Risiko "EU-Handelsrestriktionen")
- **proxy**: Verwandter Markt, aus dem sich Rueckschluesse ziehen lassen (z.B. "US-China tensions" als Proxy fuer "Seltene-Erden-Risiko")
- **weak**: Nur thematisch verwandt, kein belastbarer Rueckschluss

**Liquiditaets-Score:** Berechne aus Bid-Ask-Spread (via CLOB `GET /book`) und Handelsvolumen. Maerkte mit Spread >10% oder Volumen <$50k gelten als illiquide.

**Hinweis fuer Claude Code:** Rate-Limiting beachten (0.5s/Request, bereits in `polymarket_client._throttle()` implementiert). Ein vollstaendiger Scan ueber 100 Risikotypen × ~5 Suchterme × API-Calls braucht ca. 15-20 Minuten. Ergebnisse als zeitgestempeltes JSON in `data/scans/` speichern.

#### Schritt 3: Coverage-Report (`coverage_report.py`)

Aggregiere die Scan-Ergebnisse zur Coverage-Matrix:

```
Ausgabe: coverage_matrix.csv

Spalten:
- risk_type_id, name_de, disruption_type, sector, eu_relevance
- coverage_status: "direct" | "proxy" | "none"
- best_market_id, best_market_question, best_market_price
- liquidity_score
- scan_date
```

Zusammenfassende Metriken:
- Coverage-Rate gesamt (% der Risikotypen mit direct/proxy Match)
- Coverage-Rate nach Disruptions-Typ (Heatmap-Daten)
- Coverage-Rate nach Sektor
- Durchschnittliche Liquiditaet der gefundenen Maerkte
- Anzahl Risikotypen ohne jeglichen Match

**Wiederholung:** Der Scanner soll woechentlich laufen (Cron oder manuell), um die Coverage ueber Zeit zu tracken. Dafuer: Scan-Ergebnisse nicht ueberschreiben, sondern mit Timestamp versionieren.

### 3.4 Erfolgskriterien Phase 1

| Kriterium | Schwelle fuer Fortfuehrung |
|-----------|---------------------------|
| Coverage direct + proxy | >= 25% der Risikotypen |
| Davon mit Liquiditaets-Score > 0.5 | >= 50% der gematchten Maerkte |
| Abdeckung ueber mind. 6 der 10 Disruptions-Typen | Ja |

Wenn diese Schwellen nicht erreicht werden: Studie dokumentieren, publizieren als Negativergebnis, keine Integration.

### 3.5 Geschaetzter Aufwand

- Referenzkatalog (manuell + Skript): 3-4 Tage
- Coverage-Scanner: 2-3 Tage
- Report-Generierung + erste Analyse: 1-2 Tage
- **Gesamt Phase 1: ca. 1.5-2 Wochen**

---

## 4. Phase 2 -- Signalqualitaet

### 4.1 Ziel

Retrospektiver Vergleich: Wie gut kalibriert sind Polymarket-Preise vs. NLP-basierte Schaetzungen fuer Events, deren Ausgang bekannt ist?

### 4.2 Voraussetzungen

- Phase 1 abgeschlossen mit >= 25% Coverage
- Zugang zu Nachrichtenarchiven (AP4-Quellen oder oeffentliche Archive)
- Mindestens 20 historische Events mit bekanntem Ausgang und Polymarket-Preishistorie

### 4.3 Zielstruktur

```
terminal/
└── evaluation/
    ├── history_collector.py          Polymarket-Preishistorien sammeln
    ├── news_baseline.py              NLP-basierte Wahrscheinlichkeitsschaetzung
    ├── calibration_analysis.py       Brier Score, Log-Loss, Vorlaufzeit
    ├── data/
    │   ├── historical_events.json    Events mit bekanntem Ausgang
    │   ├── price_histories/          Polymarket-Preiszeitreihen pro Event
    │   └── news_signals/             NLP-Sentiment-Zeitreihen pro Event
    └── reports/
        └── signal_quality_report.json
```

### 4.4 Teilschritte

#### Schritt 1: Historische Events sammeln (`history_collector.py`)

Identifiziere abgeschlossene Polymarket-Maerkte mit bekanntem Ausgang:

```python
@dataclass
class HistoricalEvent:
    market_id: str
    question: str
    outcome: bool                    # Eingetreten oder nicht
    resolution_date: str             # Wann wurde der Markt aufgeloest
    disruption_type: str             # PROVIDER-Klassifikation
    price_history: list[tuple]       # [(timestamp, price), ...]
    volume_history: list[tuple]      # [(timestamp, volume), ...]
```

Nutze `polymarket_client.get_price_history(token_id, interval="1d")` fuer Tagespreise. Filtere auf Events, die:
- In den letzten 12 Monaten aufgeloest wurden
- Einem der 10 PROVIDER-Disruptions-Typen zuordenbar sind
- Mindestens 30 Tage Preishistorie haben

Erwartung: 20-40 nutzbare Events.

#### Schritt 2: News-Baseline erstellen (`news_baseline.py`)

Fuer jedes historische Event: Simuliere, was eine nachrichtenbasierte Analyse zum selben Zeitpunkt ergeben haette.

Ansatz (pragmatisch, kein vollstaendiges AP4-System noetig):

```python
class NewsBaseline:
    """Vereinfachte NLP-Pipeline als Vergleichsbasis."""

    def estimate_probability(self, event: HistoricalEvent, date: str) -> float:
        """
        Schaetzt P(Event) basierend auf Nachrichtenlage zum Stichtag.

        Methode:
        1. Suche Nachrichtenartikel zu event.question in Archiv (Zeitfenster: 7 Tage vor date)
        2. Sentiment-Analyse der gefundenen Artikel (positiv = Event wahrscheinlicher)
        3. Frequenz-Signal: Anzahl Artikel als Proxy fuer Aufmerksamkeit
        4. Kombination zu P-Schaetzung via logistische Regression oder einfaches Scoring
        """
```

**Optionen fuer Nachrichtenquellen:**
- GDELT Project (frei, global, maschinenlesbar, API verfuegbar)
- NewsAPI.org (limitierter Free Tier, 100 Requests/Tag)
- Common Crawl / Wayback Machine (fuer spezifische Artikel)
- LLM-basierte Einschaetzung: Fuer jeden Stichtag dem LLM die verfuegbaren Nachrichtenschlagzeilen geben und eine Wahrscheinlichkeitsschaetzung abfragen (als Proxy fuer eine vollwertige NLP-Pipeline)

**Hinweis fuer Claude Code:** Der LLM-basierte Ansatz ist der pragmatischste. Schicke dem Modell ueber die Anthropic API die Top-10-Schlagzeilen zum Thema fuer einen Stichtag und bitte um eine kalibrierte Wahrscheinlichkeitsschaetzung (0.0-1.0). Das ist methodisch nicht perfekt, aber als Baseline ausreichend und in 1-2 Tagen umsetzbar. Dokumentiere die Limitationen klar.

#### Schritt 3: Kalibrierungsanalyse (`calibration_analysis.py`)

Vergleiche fuer jedes Event die Zeitreihen: Polymarket-Preis vs. News-Baseline vs. tatsaechlicher Ausgang.

```python
@dataclass
class CalibrationResult:
    event_id: str
    disruption_type: str

    # Kalibrierungsmetriken (niedriger = besser)
    brier_score_market: float        # (p_market - outcome)^2
    brier_score_news: float          # (p_news - outcome)^2
    log_loss_market: float           # -[y*log(p) + (1-y)*log(1-p)]
    log_loss_news: float

    # Timing-Metriken
    lead_time_market_days: int       # Tage vor Resolution bei denen P > 0.6 (oder < 0.4)
    lead_time_news_days: int
    first_signal_market: str         # Datum erster deutlicher Ausschlag
    first_signal_news: str

    # Divergenz-Metriken
    max_divergence: float            # Max |p_market - p_news| im Zeitverlauf
    divergence_at_resolution: float  # Divergenz zum Zeitpunkt der Aufloesung
```

Aggregierte Auswertung:
- Brier Score und Log-Loss gemittelt ueber alle Events, aufgeschluesselt nach Disruptions-Typ
- Vorlaufzeit-Vergleich: In wie vielen Faellen hat welche Quelle frueher signalisiert?
- Kalibrierungskurve: Gruppiere Events nach prognostizierter P und vergleiche mit tatsaechlicher Eintrittsrate
- Identifikation von Mustern: Fuer welche Disruptions-Typen ist welche Quelle besser?

### 4.5 Erfolgskriterien Phase 2

| Kriterium | Schwelle fuer Fortfuehrung |
|-----------|---------------------------|
| Brier Score Markets < Brier Score News | Fuer mind. 60% der Disruptions-Typen |
| Oder: Vorlaufzeit News > Vorlaufzeit Markets | Um mind. 7 Tage im Median |
| Ausreichende Datenbasis | >= 20 Events mit vollstaendigen Zeitreihen |

Beide Schwellen zusammen wuerden die kaskadierende Architektur (News frueher, Markets praeziser) stuetzen. Wird keine der Schwellen erreicht, ist der Mehrwert von Prediction Markets nicht belegt.

### 4.6 Geschaetzter Aufwand

- History Collector: 2 Tage
- News-Baseline (LLM-basiert): 3-4 Tage
- Kalibrierungsanalyse: 3-4 Tage
- Auswertung und Dokumentation: 2-3 Tage
- **Gesamt Phase 2: ca. 2-3 Wochen**

---

## 5. Phase 3 -- Kopplungsexperiment

### 5.1 Ziel

Kontrollierter Parallel-Run: Drei Varianten des Scenario Generators gegen historische Daten, quantitativer Vergleich der Prognosequalitaet.

### 5.2 Voraussetzungen

- Phase 1 und 2 positiv abgeschlossen
- AP4 News-Pipeline als Prototyp verfuegbar (fruehestens M12-M15)
- AP8 Evaluationsszenarien definiert

### 5.3 Zielstruktur

```
terminal/
└── evaluation/
    ├── parallel_runner.py            Orchestriert die drei Varianten
    ├── variant_news.py               Variante A: Nur News/Social-Media
    ├── variant_markets.py            Variante B: Nur Prediction Markets
    ├── variant_combined.py           Variante C: Kombiniert
    ├── backtest_engine.py            Vergleich gegen historischen Verlauf
    ├── fallback_logic.py             Entscheidet Market vs. NLP-Signal
    ├── empirical_correlations.py     Korrelationsschaetzung aus Preishistorien
    └── reports/
        └── parallel_run_report.json
```

### 5.4 Teilschritte

#### Schritt 1: Fallback-Logik implementieren (`fallback_logic.py`)

Kern der kombinierten Variante C -- entscheidet pro Risiko-Event, welche Signalquelle verwendet wird:

```python
class SignalSelector:
    """Entscheidet fuer jedes Event, ob Polymarket-Preis oder NLP-Schaetzung verwendet wird."""

    def select_signal(self, risk_event: RiskEvent) -> SignalSource:
        """
        Entscheidungslogik:

        1. Existiert ein Polymarket-Markt mit match_quality "direct"
           UND liquidity_score > 0.5?
           -> Verwende Marktpreis als P(Event)

        2. Existiert ein Polymarket-Markt mit match_quality "proxy"
           UND liquidity_score > 0.7?  (hoehere Schwelle wegen Indirektheit)
           -> Verwende adjustierten Marktpreis (mit Unsicherheits-Aufschlag)

        3. Sonst:
           -> Verwende NLP-basierte Schaetzung

        Zusaetzlich: Divergenz-Alert wenn |p_market - p_news| > 0.2
        """
```

Die Schwellenwerte fuer Liquiditaet und Match-Qualitaet koennen aus den Ergebnissen von Phase 2 kalibriert werden.

#### Schritt 2: Empirische Korrelationen (`empirical_correlations.py`)

Ersetze die 20 manuellen Korrelationspaare (oder ergaenze sie) durch empirische Schaetzungen:

```python
class EmpiricalCorrelationEstimator:
    """Schaetzt paarweise Korrelationen aus gleichzeitigen Polymarket-Preisbewegungen."""

    def estimate(self, price_histories: dict[str, list]) -> np.ndarray:
        """
        1. Lade Tages-Preishistorien fuer alle relevanten Maerkte
        2. Berechne taegliche Preis-Aenderungen (Returns)
        3. Schaetze Pearson-Korrelation der Returns
        4. Shrinkage zum Prior (die manuellen Korrelationen) bei wenig Daten:
           rho_final = alpha * rho_empirisch + (1-alpha) * rho_manuell
           wobei alpha = min(1, n_observations / 90)
        """
```

**Hinweis fuer Claude Code:** Shrinkage ist kritisch. Fuer Marktpaare mit <30 gemeinsamen Tagesbeobachtungen ist die empirische Korrelation zu instabil -- hier den manuellen Prior beibehalten. Nutze `scenario_sampler.py` als Basis und erweitere die `_build_correlation_matrix()`-Methode.

#### Schritt 3: Drei Varianten aufsetzen

**Variante A (`variant_news.py`):** Verwendet ausschliesslich NLP-Signale aus AP4-Pipeline. Wahrscheinlichkeiten kommen aus Nachrichtenanalyse, Korrelationen aus den manuellen 20 Paaren. Entspricht dem "reinen PROVIDER-Ansatz" ohne Prediction Markets.

**Variante B (`variant_markets.py`):** Verwendet ausschliesslich Polymarket-Preise. Entspricht dem aktuellen Scenario Generator. Fuer Events ohne Markt: Ausschluss aus der Simulation (reduziertes Event-Set).

**Variante C (`variant_combined.py`):** Verwendet `fallback_logic.py` fuer die Signalauswahl und `empirical_correlations.py` fuer die Korrelationsmatrix. Vollstaendiges Event-Set.

#### Schritt 4: Backtest (`backtest_engine.py`)

Nehme historische Evaluationsszenarien aus AP8 und simuliere den Szenario-Generator zu historischen Stichtagen:

```python
@dataclass
class BacktestResult:
    scenario_id: str
    backtest_date: str               # Simulierter "heute"-Zeitpunkt
    actual_outcome: dict             # Was tatsaechlich passiert ist

    # Pro Variante:
    variant_a_prediction: dict       # Generierte Szenarien (News only)
    variant_b_prediction: dict       # Generierte Szenarien (Markets only)
    variant_c_prediction: dict       # Generierte Szenarien (Combined)

    # Vergleichsmetriken:
    variant_a_accuracy: float        # Wie nah am tatsaechlichen Verlauf
    variant_b_accuracy: float
    variant_c_accuracy: float

    variant_a_lead_time: int         # Tage Vorlauf
    variant_b_lead_time: int
    variant_c_lead_time: int
```

### 5.5 Erfolgskriterien Phase 3

| Kriterium | Interpretation |
|-----------|---------------|
| Variante C > A und C > B | Kombination bringt Mehrwert -- Integration empfohlen |
| Variante B > A, aber C ≈ B | Markets dominieren -- NLP-Kopplung bringt wenig |
| Variante A > B, aber C > A | News dominieren, Markets verbessern marginal |
| Variante A ≈ B ≈ C | Kein klarer Mehrwert -- keine Integration noetig |

### 5.6 Geschaetzter Aufwand

- Fallback-Logik + Empirische Korrelationen: 3-4 Tage
- Drei Varianten aufsetzen: 3-5 Tage
- Backtest-Engine: 4-5 Tage
- Auswertung und Dokumentation: 3-4 Tage
- **Gesamt Phase 3: ca. 3-4 Wochen**

---

## 6. Zusammenfassung

### Zeitplan

| Phase | Fruehester Start | Dauer | Abhaengigkeit |
|-------|-----------------|-------|---------------|
| Phase 1: Abdeckungsanalyse | Sofort | 1.5-2 Wochen | Keine |
| Phase 2: Signalqualitaet | Nach Phase 1 | 2-3 Wochen | Phase 1 positiv |
| Phase 3: Kopplungsexperiment | Ab M12-M15 | 3-4 Wochen | Phase 2 positiv + AP4-Prototyp |

### Entscheidungspunkte

```
Phase 1: Coverage >= 25%?
    │
    ├── Nein -> Dokumentieren, publizieren, keine Integration
    │
    └── Ja -> Phase 2: Signalqualitaet belegt?
                 │
                 ├── Nein -> Dokumentieren, publizieren, keine Integration
                 │
                 └── Ja -> Phase 3: Kombination bringt Mehrwert?
                              │
                              ├── Nein -> Markets als optionale Ergaenzung, kein Kernsystem
                              │
                              └── Ja -> Integration in PROVIDER-Architektur
                                        (Fallback-Logik + Empirische Korrelationen)
```

### Verbindung zu bestehenden APs

| Modul | Verbindung |
|-------|------------|
| `reference_catalog.py` | Input aus AP1 (Evaluationsszenarien) und PROVIDER-Ontologie (AP2) |
| `coverage_scanner.py` | Erweitert `polymarket_client.py` und `event_mapper.py` |
| `news_baseline.py` | Vereinfachte Vorform der AP4-Pipeline (LEU) |
| `calibration_analysis.py` | Methodik wiederverwendbar fuer AP8 |
| `fallback_logic.py` | Potenzielle Erweiterung von `pipeline.py` |
| `empirical_correlations.py` | Erweiterung von `scenario_sampler.py` |
| `backtest_engine.py` | Direkt integrierbar in AP8-Evaluationsbloecke |

### Technische Abhaengigkeiten

Neue Packages (zusaetzlich zu bestehender `requirements.txt`):

```
scikit-learn>=1.3      # Kalibrierungskurven, logistische Regression
matplotlib>=3.8        # Visualisierung Kalibrierungskurven, Coverage-Heatmaps
```

Optional:
```
anthropic>=0.30        # Fuer LLM-basierte News-Baseline (Phase 2)
gdeltdoc>=1.0          # GDELT Document API Client (Phase 2, Alternative)
```
