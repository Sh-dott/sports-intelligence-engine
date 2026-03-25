"""
Data Provider Registry.
Provides a unified "football" provider (merging StatsBomb deep data + Football-Data broad coverage)
and an "nba" provider for basketball.
"""

from engine.providers.base import DataProvider

_PROVIDERS: dict[str, type[DataProvider]] = {}


def register_provider(name: str, provider_class: type[DataProvider]):
    _PROVIDERS[name] = provider_class


def get_provider(name: str, **kwargs) -> DataProvider:
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys()) or "(none)"
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return _PROVIDERS[name](**kwargs)


def list_providers() -> list[str]:
    return list(_PROVIDERS.keys())


# Register providers
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

try:
    from engine.providers.unified import UnifiedFootballProvider
    register_provider("football", UnifiedFootballProvider)
except ImportError:
    pass
