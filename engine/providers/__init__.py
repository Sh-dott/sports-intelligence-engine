"""
Data Provider Registry.
Import and register all available providers here.
"""

from engine.providers.base import DataProvider

# Registry: provider name -> class
_PROVIDERS: dict[str, type[DataProvider]] = {}


def register_provider(name: str, provider_class: type[DataProvider]):
    """Register a data provider."""
    _PROVIDERS[name] = provider_class


def get_provider(name: str, **kwargs) -> DataProvider:
    """Get an instance of a registered provider."""
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys()) or "(none)"
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return _PROVIDERS[name](**kwargs)


def list_providers() -> list[str]:
    """List registered provider names."""
    return list(_PROVIDERS.keys())


# Auto-register providers on import
try:
    from engine.providers.statsbomb import StatsBombProvider
    register_provider("statsbomb", StatsBombProvider)
except ImportError:
    pass

try:
    from engine.providers.nba import NBAProvider
    register_provider("nba", NBAProvider)
except ImportError:
    pass

try:
    from engine.providers.footballdata import FootballDataProvider
    register_provider("football-data", FootballDataProvider)
except ImportError:
    pass
