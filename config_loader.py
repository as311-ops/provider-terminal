"""
PROVIDER Config Loader
=======================
Loads configuration from config.yaml with fallback to hardcoded defaults.
Gracefully handles missing PyYAML.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent


@dataclass
class APIConfig:
    rate_limit_delay: float = 0.5
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    timeout: int = 30


@dataclass
class FilteringConfig:
    min_volume: float = 10_000
    min_eu_relevance: float = 0.5


@dataclass
class MonitoringConfig:
    threshold_watch: float = 0.25
    threshold_warn: float = 0.40
    threshold_critical: float = 0.60
    rapid_shift_pct: float = 0.10
    cascade_threshold: int = 3
    cache_ttl: float = 300.0
    full_discovery_interval: int = 12


@dataclass
class SamplingConfig:
    n_scenarios: int = 1000
    n_stress: int = 100
    seed: int = 42
    min_stress_severity: float = 0.4


@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    categories: dict[int, tuple[str, str]] = field(default_factory=dict)
    search_queries: list[str] = field(default_factory=list)
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)


# Default categories (same as PROVIDER_CATEGORIES in event_mapper.py)
_DEFAULT_CATEGORIES = {
    # Kern-Kategorien
    100265: ("Geopolitics", "Kriege, Sanktionen, Konflikte"),
    120:    ("Finance", "Rezession, Zinsen, Inflation, Rohstoffe"),
    2:      ("Politics", "Wahlen, Handelspolitik, Regulierung"),
    1401:   ("Tech", "Cyberrisiken, KI-Disruption, Seltene Erden"),
    # Energie
    248:    ("Energy Industry", "Energiemaerkte, Versorgungssicherheit"),
    250:    ("Crude Oil", "Rohoel-Preis und -Versorgung"),
    102092: ("Electricity", "Stromversorgung, Netzstabilitaet"),
    103637: ("Crude Oil Futures", "Rohoel-Futures, Preiserwartungen"),
    103624: ("Uranium", "Kernenergie-Rohstoff"),
    # Handel & Zoelle
    311:    ("Trade", "Internationaler Handel, Handelsabkommen"),
    101737: ("Tariff", "Zollpolitik, Handelshemmnisse"),
    101758: ("Tariffs", "Zoelle und Handelsrestriktionen"),
    # Transport & Logistik
    100597: ("Ports", "Haefen, Hafeninfrastruktur"),
    100600: ("ILA", "Hafenarbeitergewerkschaft, Hafenstreiks"),
    1439:   ("Strikes", "Streiks, Arbeitskampf"),
    102107: ("Houthi", "Houthi-Angriffe, Rotes Meer"),
    # Agrar & Lebensmittel
    101100: ("Food", "Lebensmittelversorgung, Agrarmaerkte"),
    # Pandemie & Gesundheit
    100182: ("H5N1", "Vogelgrippe, Tierseuchen"),
    570:    ("Pandemics", "Pandemien, globale Gesundheitskrisen"),
    624:    ("Diseases", "Krankheitsausbrueche, Epidemien"),
    # Klima & Naturereignisse
    832:    ("Global Temp", "Globale Temperatur, Klimaentwicklung"),
    507:    ("Volcanic Eruption", "Vulkanausbrueche, Aschebelastung"),
    # Rohstoffe & Infrastruktur
    102139: ("Mining", "Bergbau, Rohstofffoerderung"),
    100305: ("CrowdStrike", "Cyber-Vorfaelle, IT-Ausfaelle"),
    1563:   ("Outage", "Systemausfaelle, Infrastruktur-Stoerungen"),
    # Makro & Zentralbanken
    370:    ("GDP", "Wirtschaftswachstum, BIP-Entwicklung"),
    225:    ("Economics", "Wirtschaftsindikatoren"),
    100328: ("Economy", "Konjunktur, Wirtschaftslage"),
    100486: ("Rates", "Leitzinsen, Zinspolitik"),
    101615: ("European Central Bank", "EZB-Entscheidungen, Euro-Zinsen"),
    100478: ("FOMC", "Fed-Zinsentscheidungen"),
    103638: ("Gold Futures", "Gold-Preisentwicklung"),
    103636: ("Foreign Exchange", "Devisenmaerkte, Wechselkurse"),
    # Schluesselregionen
    103025: ("European Union", "EU-Politik, Regulierung"),
    100244: ("Germany", "Deutsche Politik, Wirtschaft"),
    100194: ("Panama", "Panamakanal, Transitrouten"),
    101522: ("Japan", "Halbleiter, Automobil, Lieferketten"),
    102850: ("Sudan", "Geopolitik, Rotes Meer, Migration"),
}

# Default search queries (same as PROVIDER_SEARCH_QUERIES in event_mapper.py)
_DEFAULT_SEARCH_QUERIES = [
    # Handel & Zoelle
    "tariff EU", "tariff Europe", "trade war", "trade deal", "export ban",
    "sanctions", "supply chain",
    # Energie
    "oil price", "natural gas", "OPEC", "pipeline", "energy crisis", "nuclear energy",
    # Transport & Logistik
    "Panama Canal", "Suez Canal", "Red Sea", "shipping", "port strike", "freight",
    "Black Sea", "strait Hormuz",
    # Agrar & Lebensmittel
    "grain export", "soybean", "wheat price", "corn price", "rice export",
    "coffee price", "cocoa price", "palm oil", "sugar price", "fertilizer",
    "crop failure", "food crisis", "famine",
    # Pandemie & Gesundheit
    "pandemic", "bird flu", "H5N1", "WHO emergency", "disease outbreak",
    "mpox", "epidemic", "lockdown", "quarantine",
    # Klima & Naturereignisse
    "drought", "hurricane", "flood", "wildfire", "heat wave",
    "El Nino", "La Nina", "climate disaster", "earthquake",
    # Geopolitik
    "ceasefire", "NATO", "China Taiwan", "Russia Ukraine",
    "India Pakistan", "Iran Israel", "North Korea", "coup",
    # Finanzen & Makro
    "recession Europe", "inflation", "interest rate Fed", "interest rate ECB",
    # Technologie & Rohstoffe
    "rare earth", "semiconductor", "chip shortage", "lithium", "cobalt",
    "cyber attack", "data center", "subsea cable", "undersea cable",
    # Schluessellaender
    "Brazil", "Argentina", "Indonesia", "India export",
    "South Africa", "Turkey", "Vietnam",
]


def _parse_categories(raw: dict) -> dict[int, tuple[str, str]]:
    """Parse categories from YAML format."""
    result = {}
    for tag_id, value in raw.items():
        if isinstance(value, list) and len(value) >= 2:
            result[int(tag_id)] = (str(value[0]), str(value[1]))
    return result


def _parse_correlations(raw: dict) -> dict[tuple[str, str], float]:
    """Parse correlations from YAML 'type_a-type_b: value' format."""
    result = {}
    for key, value in raw.items():
        parts = str(key).split("-", 1)
        if len(parts) == 2:
            result[(parts[0], parts[1])] = float(value)
    return result


def load_config(path: Optional[str | Path] = None) -> Config:
    """
    Load configuration from YAML file with fallback to defaults.

    Args:
        path: Path to config.yaml. Defaults to terminal/config.yaml.
    """
    if path is None:
        path = _CONFIG_DIR / "config.yaml"
    else:
        path = Path(path)

    config = Config(
        categories=dict(_DEFAULT_CATEGORIES),
        search_queries=list(_DEFAULT_SEARCH_QUERIES),
    )

    if not path.exists():
        logger.debug(f"Keine Config-Datei gefunden ({path}), nutze Defaults")
        return config

    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML nicht installiert, nutze Defaults")
        return config

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Config-Datei konnte nicht geladen werden: {e}")
        return config

    # API config
    if "api" in raw:
        api_raw = raw["api"]
        config.api = APIConfig(
            rate_limit_delay=api_raw.get("rate_limit_delay", config.api.rate_limit_delay),
            max_retries=api_raw.get("max_retries", config.api.max_retries),
            initial_retry_delay=api_raw.get("initial_retry_delay", config.api.initial_retry_delay),
            retry_backoff_factor=api_raw.get("retry_backoff_factor", config.api.retry_backoff_factor),
            timeout=api_raw.get("timeout", config.api.timeout),
        )

    # Filtering config
    if "filtering" in raw:
        f_raw = raw["filtering"]
        config.filtering = FilteringConfig(
            min_volume=f_raw.get("min_volume", config.filtering.min_volume),
            min_eu_relevance=f_raw.get("min_eu_relevance", config.filtering.min_eu_relevance),
        )

    # Monitoring config
    if "monitoring" in raw:
        m_raw = raw["monitoring"]
        config.monitoring = MonitoringConfig(
            threshold_watch=m_raw.get("threshold_watch", config.monitoring.threshold_watch),
            threshold_warn=m_raw.get("threshold_warn", config.monitoring.threshold_warn),
            threshold_critical=m_raw.get("threshold_critical", config.monitoring.threshold_critical),
            rapid_shift_pct=m_raw.get("rapid_shift_pct", config.monitoring.rapid_shift_pct),
            cascade_threshold=m_raw.get("cascade_threshold", config.monitoring.cascade_threshold),
            cache_ttl=m_raw.get("cache_ttl", config.monitoring.cache_ttl),
            full_discovery_interval=m_raw.get("full_discovery_interval", config.monitoring.full_discovery_interval),
        )

    # Sampling config
    if "sampling" in raw:
        s_raw = raw["sampling"]
        config.sampling = SamplingConfig(
            n_scenarios=s_raw.get("n_scenarios", config.sampling.n_scenarios),
            n_stress=s_raw.get("n_stress", config.sampling.n_stress),
            seed=s_raw.get("seed", config.sampling.seed),
            min_stress_severity=s_raw.get("min_stress_severity", config.sampling.min_stress_severity),
        )

    # Categories
    if "categories" in raw:
        config.categories = _parse_categories(raw["categories"])

    # Search queries
    if "search_queries" in raw:
        config.search_queries = raw["search_queries"]

    # Correlations
    if "correlations" in raw:
        config.correlations = _parse_correlations(raw["correlations"])

    logger.info(f"Config geladen: {path} ({len(config.search_queries)} Suchbegriffe)")
    return config


# Singleton
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config singleton. Loads from default path on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
