"""
PROVIDER Event Mapper
=====================
Maps Polymarket prediction events to supply chain disruption parameters
relevant for European/German supply chains.

Architecture:
    1. PROVIDER_CATEGORIES: Which Polymarket tags to scan
    2. KEYWORD_RULES: Pattern-based mapping of market questions to disruption types
    3. DISRUPTION_PROFILES: How each disruption type translates to simulation parameters
    4. EventMapper: Orchestrates discovery, classification, and parameter generation
"""

import re
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .polymarket_client import PolymarketClient, Market, Event

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. PROVIDER-RELEVANT POLYMARKET CATEGORIES
# ═══════════════════════════════════════════════════════════════

PROVIDER_CATEGORIES = {
    # tag_id: (label, relevance for EU supply chains)
    # --- Kern-Kategorien (bestehend) ---
    100265: ("Geopolitics", "Kriege, Sanktionen, Konflikte"),
    120:    ("Finance", "Rezession, Zinsen, Inflation, Rohstoffe"),
    2:      ("Politics", "Wahlen, Handelspolitik, Regulierung"),
    1401:   ("Tech", "Cyberrisiken, KI-Disruption, Seltene Erden"),

    # --- Energie ---
    248:    ("Energy Industry", "Energiemaerkte, Versorgungssicherheit"),
    250:    ("Crude Oil", "Rohoel-Preis und -Versorgung"),
    102092: ("Electricity", "Stromversorgung, Netzstabilitaet"),
    103637: ("Crude Oil Futures", "Rohoel-Futures, Preiserwartungen"),
    103624: ("Uranium", "Kernenergie-Rohstoff"),

    # --- Handel & Zoelle ---
    311:    ("Trade", "Internationaler Handel, Handelsabkommen"),
    101737: ("Tariff", "Zollpolitik, Handelshemmnisse"),
    101758: ("Tariffs", "Zoelle und Handelsrestriktionen"),

    # --- Transport & Logistik ---
    100597: ("Ports", "Haefen, Hafeninfrastruktur"),
    100600: ("ILA", "Hafenarbeitergewerkschaft, Hafenstreiks"),
    1439:   ("Strikes", "Streiks, Arbeitskampf"),
    102107: ("Houthi", "Houthi-Angriffe, Rotes Meer"),

    # --- Agrar & Lebensmittel ---
    101100: ("Food", "Lebensmittelversorgung, Agrarmaerkte"),

    # --- Pandemie & Gesundheit ---
    100182: ("H5N1", "Vogelgrippe, Tierseuchen"),
    570:    ("Pandemics", "Pandemien, globale Gesundheitskrisen"),
    624:    ("Diseases", "Krankheitsausbrueche, Epidemien"),

    # --- Klima & Naturereignisse ---
    832:    ("Global Temp", "Globale Temperatur, Klimaentwicklung"),
    507:    ("Volcanic Eruption", "Vulkanausbrueche, Aschebelastung"),

    # --- Rohstoffe & Infrastruktur ---
    102139: ("Mining", "Bergbau, Rohstofffoerderung"),
    100305: ("CrowdStrike", "Cyber-Vorfaelle, IT-Ausfaelle"),
    1563:   ("Outage", "Systemausfaelle, Infrastruktur-Stoerungen"),

    # --- Makro & Zentralbanken ---
    370:    ("GDP", "Wirtschaftswachstum, BIP-Entwicklung"),
    225:    ("Economics", "Wirtschaftsindikatoren"),
    100328: ("Economy", "Konjunktur, Wirtschaftslage"),
    100486: ("Rates", "Leitzinsen, Zinspolitik"),
    101615: ("European Central Bank", "EZB-Entscheidungen, Euro-Zinsen"),
    100478: ("FOMC", "Fed-Zinsentscheidungen"),
    103638: ("Gold Futures", "Gold-Preisentwicklung"),
    103636: ("Foreign Exchange", "Devisenmaerkte, Wechselkurse"),

    # --- Schluesselregionen ---
    103025: ("European Union", "EU-Politik, Regulierung"),
    100244: ("Germany", "Deutsche Politik, Wirtschaft"),
    100194: ("Panama", "Panamakanal, Transitrouten"),
    101522: ("Japan", "Halbleiter, Automobil, Lieferketten"),
    102850: ("Sudan", "Geopolitik, Rotes Meer, Migration"),
}

# Additional free-text searches for markets not cleanly tagged
PROVIDER_SEARCH_QUERIES = [
    # --- Handel & Zoelle ---
    "tariff EU",
    "tariff Europe",
    "trade war",
    "trade deal",
    "export ban",
    "sanctions",
    "supply chain",

    # --- Energie ---
    "oil price",
    "natural gas",
    "OPEC",
    "pipeline",
    "energy crisis",
    "nuclear energy",

    # --- Transport & Logistik ---
    "Panama Canal",
    "Suez Canal",
    "Red Sea",
    "shipping",
    "port strike",
    "freight",
    "Black Sea",
    "strait Hormuz",

    # --- Agrar & Lebensmittel ---
    "grain export",
    "soybean",
    "wheat price",
    "corn price",
    "rice export",
    "coffee price",
    "cocoa price",
    "palm oil",
    "sugar price",
    "fertilizer",
    "crop failure",
    "food crisis",
    "famine",

    # --- Pandemie & Gesundheit ---
    "pandemic",
    "bird flu",
    "H5N1",
    "WHO emergency",
    "disease outbreak",
    "mpox",
    "epidemic",
    "lockdown",
    "quarantine",

    # --- Klima & Naturereignisse ---
    "drought",
    "hurricane",
    "flood",
    "wildfire",
    "heat wave",
    "El Nino",
    "La Nina",
    "climate disaster",
    "earthquake",

    # --- Geopolitik ---
    "ceasefire",
    "NATO",
    "China Taiwan",
    "Russia Ukraine",
    "India Pakistan",
    "Iran Israel",
    "North Korea",
    "coup",

    # --- Finanzen & Makro ---
    "recession Europe",
    "inflation",
    "interest rate Fed",
    "interest rate ECB",

    # --- Technologie & Rohstoffe ---
    "rare earth",
    "semiconductor",
    "chip shortage",
    "lithium",
    "cobalt",
    "cyber attack",
    "data center",
    "subsea cable",
    "undersea cable",

    # --- Schluessellaender ---
    "Brazil",
    "Argentina",
    "Indonesia",
    "India export",
    "South Africa",
    "Turkey",
    "Vietnam",
]


# ═══════════════════════════════════════════════════════════════
# 2. DISRUPTION TYPES & KEYWORD CLASSIFICATION RULES
# ═══════════════════════════════════════════════════════════════

@dataclass
class DisruptionType:
    """A category of supply chain disruption."""
    id: str
    label_de: str
    label_en: str
    description: str


DISRUPTION_TYPES = {
    "energy_supply": DisruptionType(
        "energy_supply", "Energieversorgung", "Energy Supply",
        "Disruptions to oil, gas, electricity supply affecting production costs"
    ),
    "trade_restriction": DisruptionType(
        "trade_restriction", "Handelsrestriktionen", "Trade Restrictions",
        "Tariffs, sanctions, export bans reducing trade flows"
    ),
    "transport_disruption": DisruptionType(
        "transport_disruption", "Transportunterbrechung", "Transport Disruption",
        "Shipping route blockages, port closures, logistics failures"
    ),
    "agricultural_shock": DisruptionType(
        "agricultural_shock", "Agrarschock", "Agricultural Shock",
        "Crop failures, export bans on agricultural products"
    ),
    "financial_crisis": DisruptionType(
        "financial_crisis", "Finanzkrise", "Financial Crisis",
        "Recession, credit crunch, currency instability"
    ),
    "geopolitical_conflict": DisruptionType(
        "geopolitical_conflict", "Geopolitischer Konflikt", "Geopolitical Conflict",
        "Wars, military conflicts disrupting trade and supply"
    ),
    "pandemic": DisruptionType(
        "pandemic", "Pandemie", "Pandemic",
        "Health crises causing lockdowns, workforce reduction, border closures"
    ),
    "climate_event": DisruptionType(
        "climate_event", "Klimaereignis", "Climate Event",
        "Extreme weather, droughts, floods affecting production and transport"
    ),
    "tech_disruption": DisruptionType(
        "tech_disruption", "Technologie-Disruption", "Tech Disruption",
        "Cyber attacks, semiconductor shortages, critical mineral scarcity"
    ),
    "political_instability": DisruptionType(
        "political_instability", "Politische Instabilitaet", "Political Instability",
        "Elections, coups, policy shifts in key supplier countries"
    ),
}

# Keyword patterns for classifying markets into disruption types
# Format: (compiled_regex, disruption_type_id, eu_relevance_boost)
KEYWORD_RULES: list[tuple[re.Pattern, str, float]] = [
    # Energy
    (re.compile(r"oil|crude|brent|wti|opec", re.I), "energy_supply", 0.9),
    (re.compile(r"natural gas|lng|gas price|henry hub|ttf", re.I), "energy_supply", 1.0),
    (re.compile(r"electricity|power grid|nuclear|energy", re.I), "energy_supply", 0.7),
    (re.compile(r"pipeline|nord stream|gas pipeline", re.I), "energy_supply", 0.9),
    (re.compile(r"uranium|nuclear fuel", re.I), "energy_supply", 0.7),
    (re.compile(r"battery supply|energy storage", re.I), "energy_supply", 0.7),

    # Trade
    (re.compile(r"tariff|trade war|trade deal|trade agreement", re.I), "trade_restriction", 1.0),
    (re.compile(r"sanction|embargo|export ban|import restriction", re.I), "trade_restriction", 0.9),
    (re.compile(r"EU.*tariff|tariff.*EU|tariff.*Europe", re.I), "trade_restriction", 1.0),
    (re.compile(r"trade deal.*\b(india|indonesia|korea|japan|brazil|argentina|vietnam|turkey)\b", re.I), "trade_restriction", 0.9),
    (re.compile(r"customs|import dut", re.I), "trade_restriction", 0.8),

    # Transport
    (re.compile(r"panama canal|suez canal|shipping|freight", re.I), "transport_disruption", 0.9),
    (re.compile(r"port.*strike|port.*closure|logistics", re.I), "transport_disruption", 0.8),
    (re.compile(r"red sea|houthi|strait of hormuz", re.I), "transport_disruption", 0.9),
    (re.compile(r"black sea|grain corridor", re.I), "transport_disruption", 1.0),
    (re.compile(r"rhine|danube|inland waterway|river.*transport", re.I), "transport_disruption", 0.9),
    (re.compile(r"dock.*worker|longshor|ILA.*strike|port.*worker", re.I), "transport_disruption", 0.9),
    (re.compile(r"rail.*strike|train.*strike|railway.*disrupt", re.I), "transport_disruption", 0.8),
    (re.compile(r"container.*short|container.*crisis", re.I), "transport_disruption", 0.8),

    # Agriculture (word boundaries to prevent false positives like "Price"→"rice", "Cornyn"→"corn")
    (re.compile(r"\bsoy\b|soybean|\bwheat\b|\bcorn\b|\bgrain\b|\bcrop\b|\bharvest\b|\bfamine\b", re.I), "agricultural_shock", 0.9),
    (re.compile(r"\bfertilizer\b|\bfood price\b|\bagricultural\b|\blivestock\b", re.I), "agricultural_shock", 0.8),
    (re.compile(r"\bdrought\b.*\bcrop\b|\bflood\b.*\bharvest\b", re.I), "agricultural_shock", 0.9),
    (re.compile(r"\bcoffee\b|\bcocoa\b|\bpalm oil\b|\bsugar\b.*\bprice\b|\brice\b.*\bexport\b|\brice\b.*\bprice\b", re.I), "agricultural_shock", 0.8),
    (re.compile(r"\bfood crisis\b|\bfood shortage\b|\bhunger\b|\bmalnutrition\b", re.I), "agricultural_shock", 0.9),
    (re.compile(r"\bcattle\b|\bpork\b|\bpoultry\b|\bdairy\b|\bmeat\b.*\bprice\b", re.I), "agricultural_shock", 0.7),
    (re.compile(r"\bpotash\b|\bammonia\b|\burea\b|\bnitrogen\b.*\bfertil", re.I), "agricultural_shock", 0.8),
    (re.compile(r"\blocust\b|\bpest\b.*\boutbreak\b|\bcrop disease\b|\bblight\b", re.I), "agricultural_shock", 0.8),

    # Financial
    (re.compile(r"recession|gdp.*negative|depression", re.I), "financial_crisis", 0.9),
    (re.compile(r"interest rate|fed.*rate|ecb.*rate|rate.*cut|rate.*hike", re.I), "financial_crisis", 0.8),
    (re.compile(r"inflation|cpi|deflation", re.I), "financial_crisis", 0.8),
    (re.compile(r"stock.*crash|market.*crash|financial.*crisis", re.I), "financial_crisis", 0.9),
    (re.compile(r"euro|EUR\/USD|dollar.*euro", re.I), "financial_crisis", 0.7),
    (re.compile(r"bank of japan|BoJ|yen.*crisis", re.I), "financial_crisis", 0.7),
    (re.compile(r"sovereign debt|bond.*crisis|credit.*crisis", re.I), "financial_crisis", 0.8),

    # Geopolitical Conflict
    (re.compile(r"russia.*ukraine|ukraine.*war|ceasefire.*ukraine", re.I), "geopolitical_conflict", 1.0),
    (re.compile(r"china.*taiwan|taiwan.*invasion|strait.*taiwan", re.I), "geopolitical_conflict", 0.9),
    (re.compile(r"NATO|nuclear.*war|world war|military.*conflict", re.I), "geopolitical_conflict", 0.8),
    (re.compile(r"middle east.*war|iran.*israel|iran.*attack", re.I), "geopolitical_conflict", 0.7),
    (re.compile(r"india.*pakistan|pakistan.*india|kashmir.*conflict", re.I), "geopolitical_conflict", 0.7),
    (re.compile(r"north korea|DPRK|korean.*peninsula", re.I), "geopolitical_conflict", 0.7),
    (re.compile(r"south china sea|spratly|senkaku", re.I), "geopolitical_conflict", 0.8),

    # Pandemic
    (re.compile(r"pandemic|covid|bird flu|h5n1|monkeypox|mpox|WHO.*emergency", re.I), "pandemic", 0.9),
    (re.compile(r"lockdown|quarantine|health.*crisis|epidemic", re.I), "pandemic", 0.8),
    (re.compile(r"disease.*outbreak|virus.*outbreak|new.*virus|new.*pathogen", re.I), "pandemic", 0.9),
    (re.compile(r"avian.*influenza|swine.*flu|zoonotic", re.I), "pandemic", 0.8),
    (re.compile(r"drug.*shortage|pharmaceutical.*crisis|vaccine.*shortage", re.I), "pandemic", 0.7),
    (re.compile(r"WHO.*declar|public.*health.*emergency|PHEIC", re.I), "pandemic", 0.9),

    # Climate
    (re.compile(r"hurricane|cyclone|typhoon|flood|drought|wildfire", re.I), "climate_event", 0.7),
    (re.compile(r"climate.*change|global.*warming|sea.*level", re.I), "climate_event", 0.6),
    (re.compile(r"natural.*disaster|earthquake|volcano|eruption", re.I), "climate_event", 0.6),
    (re.compile(r"heat.*wave|heat.*record|extreme.*heat|record.*temperature", re.I), "climate_event", 0.7),
    (re.compile(r"el ni[nñ]o|la ni[nñ]a", re.I), "climate_event", 0.7),
    (re.compile(r"arctic.*ice|glacier|sea.*ice|permafrost", re.I), "climate_event", 0.5),
    (re.compile(r"water.*crisis|water.*shortage|water.*scarcity", re.I), "climate_event", 0.8),
    (re.compile(r"landslide|mudslide|tsunami|storm.*surge", re.I), "climate_event", 0.6),
    (re.compile(r"global.*temp|hottest.*year|temperature.*record", re.I), "climate_event", 0.7),

    # Tech
    (re.compile(r"semiconductor|chip.*shortage|TSMC", re.I), "tech_disruption", 0.9),
    (re.compile(r"rare earth|lithium|cobalt|critical mineral", re.I), "tech_disruption", 0.8),
    (re.compile(r"cyber.*attack|ransomware|hack.*infrastructure", re.I), "tech_disruption", 0.7),
    (re.compile(r"data center|cloud.*outage|AWS.*outage|server.*outage", re.I), "tech_disruption", 0.7),
    (re.compile(r"subsea.*cable|undersea.*cable|internet.*cable", re.I), "tech_disruption", 0.8),
    (re.compile(r"solar.*panel|photovoltaic|EV.*battery", re.I), "tech_disruption", 0.7),
    (re.compile(r"AI.*downturn|AI.*bubble|tech.*crash", re.I), "tech_disruption", 0.7),

    # Political Instability
    (re.compile(r"coup|regime.*change|civil.*war|revolution", re.I), "political_instability", 0.7),
    (re.compile(r"brazil.*election|brazil.*crisis|brazil.*political", re.I), "political_instability", 0.8),
    (re.compile(r"argentina.*election|argentina.*crisis", re.I), "political_instability", 0.7),
    (re.compile(r"china.*political|china.*xi|china.*policy", re.I), "political_instability", 0.7),
    (re.compile(r"turkey.*political|erdogan|turkey.*crisis", re.I), "political_instability", 0.7),
    (re.compile(r"south africa.*political|south africa.*crisis", re.I), "political_instability", 0.7),
    (re.compile(r"iran.*regime|iranian.*regime|iran.*fall", re.I), "political_instability", 0.8),
    (re.compile(r"military.*junta|martial.*law|state.*emergency", re.I), "political_instability", 0.7),
]


# ═══════════════════════════════════════════════════════════════
# 3. DISRUPTION PROFILES: How events map to simulation parameters
# ═══════════════════════════════════════════════════════════════

@dataclass
class SupplyChainParameter:
    """A single parameter affected by a disruption."""
    name: str           # e.g. "soy_availability"
    unit: str           # e.g. "percent_change"
    base_impact: float  # impact at p=1.0 (certain event)
    description: str
    affected_sectors: list[str] = field(default_factory=list)


# Maps disruption_type_id -> list of affected supply chain parameters
DISRUPTION_PROFILES: dict[str, list[SupplyChainParameter]] = {
    "energy_supply": [
        SupplyChainParameter("energy_cost_factor", "multiplier", 1.8,
            "Energy cost multiplier for production",
            ["Chemie", "Stahl", "Glas", "Lebensmittel", "Papier"]),
        SupplyChainParameter("gas_availability", "percent_change", -40.0,
            "Natural gas supply reduction",
            ["Chemie", "Futtermittel", "Duengemittel"]),
        SupplyChainParameter("transport_fuel_cost", "multiplier", 1.5,
            "Fuel cost increase for logistics",
            ["Logistik", "Einzelhandel"]),
    ],
    "trade_restriction": [
        SupplyChainParameter("import_cost_factor", "multiplier", 1.3,
            "Import cost increase due to tariffs",
            ["Automobil", "Maschinenbau", "Elektronik"]),
        SupplyChainParameter("export_volume", "percent_change", -20.0,
            "Export volume reduction",
            ["Automobil", "Maschinenbau", "Chemie"]),
        SupplyChainParameter("supply_diversification_delay", "days", 90.0,
            "Time to find alternative suppliers",
            ["Alle Sektoren"]),
    ],
    "transport_disruption": [
        SupplyChainParameter("shipping_lead_time", "multiplier", 2.5,
            "Lead time increase for sea freight",
            ["Alle importabhaengigen Sektoren"]),
        SupplyChainParameter("freight_cost", "multiplier", 3.0,
            "Freight cost increase",
            ["Einzelhandel", "Automobil", "Elektronik"]),
        SupplyChainParameter("route_availability", "percent_change", -30.0,
            "Reduction in available shipping routes",
            ["Logistik"]),
    ],
    "agricultural_shock": [
        SupplyChainParameter("soy_availability", "percent_change", -40.0,
            "Soy supply reduction (Brazil/Argentina)",
            ["Futtermittel", "Tierhaltung", "Lebensmittel"]),
        SupplyChainParameter("grain_price", "multiplier", 1.7,
            "Grain price increase",
            ["Futtermittel", "Backwaren", "Lebensmittel"]),
        SupplyChainParameter("feed_cost", "multiplier", 1.6,
            "Animal feed cost increase",
            ["Schweinezucht", "Gefluegel", "Milchwirtschaft"]),
        SupplyChainParameter("food_price_index", "multiplier", 1.25,
            "Consumer food price increase",
            ["Einzelhandel", "Gastronomie"]),
    ],
    "financial_crisis": [
        SupplyChainParameter("credit_availability", "percent_change", -30.0,
            "Reduction in trade financing",
            ["KMU", "Baugewerbe", "Einzelhandel"]),
        SupplyChainParameter("demand_shock", "percent_change", -15.0,
            "Consumer/industrial demand reduction",
            ["Automobil", "Maschinenbau", "Luxusgueter"]),
        SupplyChainParameter("currency_volatility", "multiplier", 1.5,
            "EUR/USD exchange rate volatility increase",
            ["Import", "Export"]),
    ],
    "geopolitical_conflict": [
        SupplyChainParameter("trade_corridor_disruption", "percent_change", -50.0,
            "Disruption of major trade corridors",
            ["Alle importabhaengigen Sektoren"]),
        SupplyChainParameter("sanctions_impact", "percent_change", -25.0,
            "Supplier loss due to sanctions",
            ["Energie", "Rohstoffe", "Chemie"]),
        SupplyChainParameter("uncertainty_index", "multiplier", 2.0,
            "Economic uncertainty multiplier",
            ["Investitionsgueter", "Bauwirtschaft"]),
        SupplyChainParameter("energy_supply_risk", "percent_change", -30.0,
            "Energy supply disruption from conflict zones",
            ["Energie", "Chemie", "Stahl"]),
    ],
    "pandemic": [
        SupplyChainParameter("workforce_availability", "percent_change", -20.0,
            "Workforce reduction due to illness/lockdowns",
            ["Alle Sektoren"]),
        SupplyChainParameter("border_closure_impact", "percent_change", -40.0,
            "Cross-border trade reduction",
            ["Automobil", "Elektronik", "Lebensmittel"]),
        SupplyChainParameter("demand_volatility", "multiplier", 2.0,
            "Demand pattern disruption (bullwhip)",
            ["Einzelhandel", "Pharma", "Lebensmittel"]),
    ],
    "climate_event": [
        SupplyChainParameter("agricultural_yield", "percent_change", -25.0,
            "Crop yield reduction",
            ["Landwirtschaft", "Lebensmittel"]),
        SupplyChainParameter("infrastructure_damage", "percent_change", -15.0,
            "Transport infrastructure damage",
            ["Logistik", "Binnenschifffahrt"]),
        SupplyChainParameter("water_availability", "percent_change", -30.0,
            "Industrial water supply reduction",
            ["Chemie", "Halbleiter", "Lebensmittel"]),
    ],
    "tech_disruption": [
        SupplyChainParameter("semiconductor_supply", "percent_change", -35.0,
            "Semiconductor availability reduction",
            ["Automobil", "Elektronik", "Maschinenbau"]),
        SupplyChainParameter("critical_mineral_cost", "multiplier", 2.0,
            "Rare earth / lithium price increase",
            ["Batterie", "Elektronik", "Erneuerbare Energien"]),
        SupplyChainParameter("it_system_downtime", "days", 14.0,
            "IT infrastructure outage duration",
            ["Alle digitalisierten Sektoren"]),
    ],
    "political_instability": [
        SupplyChainParameter("supplier_country_risk", "multiplier", 1.5,
            "Risk premium for sourcing from unstable countries",
            ["Rohstoffe", "Agrar", "Energie"]),
        SupplyChainParameter("export_policy_uncertainty", "percent_change", -15.0,
            "Supply reduction from uncertain export policies",
            ["Agrar", "Rohstoffe"]),
        SupplyChainParameter("investment_freeze", "percent_change", -20.0,
            "Reduction in cross-border investment",
            ["Maschinenbau", "Automobil"]),
    ],
}


# ═══════════════════════════════════════════════════════════════
# 4. EVENT MAPPER CLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class MappedEvent:
    """A Polymarket event mapped to PROVIDER disruption parameters."""
    market_id: str
    question: str
    probability: float
    volume: float
    token_id: Optional[str]
    disruption_type: str
    eu_relevance: float  # 0.0 - 1.0
    parameters: list[dict]  # scaled parameters based on probability

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "probability": round(self.probability, 4),
            "volume": self.volume,
            "disruption_type": self.disruption_type,
            "eu_relevance": round(self.eu_relevance, 2),
            "token_id": self.token_id,
            "parameters": self.parameters,
        }


class EventMapper:
    """
    Discovers Polymarket events and maps them to PROVIDER supply chain
    disruption parameters.

    Usage:
        client = PolymarketClient()
        mapper = EventMapper(client)
        mapped = mapper.discover_and_map()
        for event in mapped:
            print(f"{event.question}: p={event.probability:.0%}")
            for p in event.parameters:
                print(f"  {p['name']}: {p['scaled_impact']:.2f} {p['unit']}")
    """

    def __init__(
        self,
        client: PolymarketClient,
        min_volume: float | None = None,
        min_eu_relevance: float | None = None,
        cache_ttl: float | None = None,
    ):
        self.client = client

        # Load defaults from config if not explicitly provided
        try:
            from .config_loader import get_config
            cfg = get_config()
            self.min_volume = min_volume if min_volume is not None else cfg.filtering.min_volume
            self.min_eu_relevance = min_eu_relevance if min_eu_relevance is not None else cfg.filtering.min_eu_relevance
            self.cache_ttl = cache_ttl if cache_ttl is not None else cfg.monitoring.cache_ttl
        except Exception:
            self.min_volume = min_volume if min_volume is not None else 10_000
            self.min_eu_relevance = min_eu_relevance if min_eu_relevance is not None else 0.5
            self.cache_ttl = cache_ttl if cache_ttl is not None else 300.0

        self._event_cache: list[MappedEvent] = []
        self._cache_timestamp: float = 0.0

    def discover_and_map(
        self,
        use_categories: bool = True,
        use_search: bool = True,
        max_markets_per_category: int = 50,
    ) -> list[MappedEvent]:
        """
        Full discovery pipeline:
        1. Scan all PROVIDER_CATEGORIES
        2. Run PROVIDER_SEARCH_QUERIES
        3. Classify and deduplicate
        4. Map to disruption parameters

        Returns sorted list of MappedEvents (highest EU relevance first).
        """
        raw_markets: dict[str, Market] = {}  # deduplicate by market ID

        # Phase 1: Category-based discovery
        if use_categories:
            for tag_id, (label, _) in PROVIDER_CATEGORIES.items():
                logger.info(f"Scanning category: {label} (tag_id={tag_id})")
                try:
                    events = self.client.get_events(
                        tag_id=tag_id,
                        closed=False,
                        limit=max_markets_per_category,
                    )
                    for event in events:
                        for market in event.markets:
                            if market.active and not market.closed:
                                raw_markets[market.id] = market
                except Exception as e:
                    logger.warning(f"Failed to scan category {label}: {e}")

        # Phase 2: Search-based discovery
        if use_search:
            for query in PROVIDER_SEARCH_QUERIES:
                logger.info(f"Searching: '{query}'")
                try:
                    results = self.client.search_markets(query, limit=10)
                    for market in results:
                        if market.active and not market.closed:
                            raw_markets[market.id] = market
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")

        logger.info(f"Discovered {len(raw_markets)} unique active markets")

        # Phase 3: Classify and map
        mapped_events = []
        for market in raw_markets.values():
            classification = self._classify_market(market)
            if classification is None:
                continue

            disruption_type, eu_relevance = classification

            if eu_relevance < self.min_eu_relevance:
                continue
            if market.volume < self.min_volume:
                continue

            parameters = self._compute_parameters(
                disruption_type, market.probability
            )

            token_id = market.token_ids[0] if market.token_ids else None

            mapped_events.append(MappedEvent(
                market_id=market.id,
                question=market.question,
                probability=market.probability,
                volume=market.volume,
                token_id=token_id,
                disruption_type=disruption_type,
                eu_relevance=eu_relevance,
                parameters=parameters,
            ))

        # Sort by EU relevance * probability (most impactful first)
        mapped_events.sort(
            key=lambda e: e.eu_relevance * e.probability,
            reverse=True,
        )

        logger.info(
            f"Mapped {len(mapped_events)} events to PROVIDER parameters "
            f"(filtered from {len(raw_markets)} raw markets)"
        )

        # Update cache
        self._event_cache = mapped_events
        self._cache_timestamp = time.time()

        return mapped_events

    @property
    def cache_valid(self) -> bool:
        """Check if the event cache is still within TTL."""
        return bool(self._event_cache) and (time.time() - self._cache_timestamp) < self.cache_ttl

    def get_cached_events(self) -> list[MappedEvent]:
        """Return cached events (empty list if cache is invalid)."""
        if self.cache_valid:
            return self._event_cache
        return []

    def refresh_prices(self) -> list[MappedEvent]:
        """
        Refresh only the prices for cached events via CLOB API.
        Much faster than full discovery (1 API call per event vs. 30+).
        Falls back to full discovery if cache is empty.
        """
        if not self._event_cache:
            logger.info("Kein Cache vorhanden, starte Full Discovery...")
            return self.discover_and_map()

        updated = []
        for event in self._event_cache:
            if event.token_id:
                new_price = self.client.get_price(event.token_id)
                if new_price is not None:
                    event.probability = new_price
                    event.parameters = self._compute_parameters(
                        event.disruption_type, new_price
                    )
            updated.append(event)

        self._event_cache = updated
        self._cache_timestamp = time.time()
        logger.info(f"Price Refresh: {len(updated)} Events aktualisiert")
        return updated

    def _classify_market(self, market: Market) -> Optional[tuple[str, float]]:
        """
        Classify a market into a disruption type using keyword rules.
        Returns (disruption_type_id, eu_relevance) or None if no match.
        """
        text = f"{market.question} {market.description}"
        best_match: Optional[tuple[str, float]] = None
        best_score = 0.0

        for pattern, dtype, relevance in KEYWORD_RULES:
            if pattern.search(text):
                # Bonus for markets with higher volume (more liquid = more reliable)
                volume_bonus = min(market.volume / 1_000_000, 0.1)  # up to +0.1
                score = relevance + volume_bonus
                if score > best_score:
                    best_score = score
                    best_match = (dtype, relevance)

        return best_match

    def _compute_parameters(
        self,
        disruption_type: str,
        probability: float,
    ) -> list[dict]:
        """
        Scale disruption profile parameters by market probability.

        The base_impact represents the full effect at p=1.0 (certain event).
        At actual probability p, the expected impact is scaled proportionally.
        """
        profile = DISRUPTION_PROFILES.get(disruption_type, [])
        scaled = []
        for param in profile:
            if param.unit == "multiplier":
                # Interpolate between 1.0 (no effect) and base_impact
                scaled_impact = 1.0 + (param.base_impact - 1.0) * probability
            elif param.unit == "percent_change":
                scaled_impact = param.base_impact * probability
            elif param.unit == "days":
                scaled_impact = param.base_impact * probability
            else:
                scaled_impact = param.base_impact * probability

            scaled.append({
                "name": param.name,
                "unit": param.unit,
                "base_impact": param.base_impact,
                "scaled_impact": round(scaled_impact, 4),
                "probability": round(probability, 4),
                "description": param.description,
                "affected_sectors": param.affected_sectors,
            })
        return scaled
