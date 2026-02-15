"""
PROVIDER Scenario Generator
============================
Generates supply chain crisis scenarios for ORCA simulations
based on Polymarket prediction market data.

Modules:
    polymarket_client  - API client for Gamma & CLOB endpoints
    event_mapper       - Maps prediction markets to supply chain parameters
    scenario_sampler   - Monte Carlo scenario generation with correlations
    main               - CLI entry point
"""

__version__ = "0.1.0"
