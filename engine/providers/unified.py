"""
Unified Football Provider
Merges StatsBomb (deep analytics) and Football-Data.org (broad coverage)
into a single "Football" data source with MongoDB caching.
"""

import pandas as pd

from engine.providers.base import DataProvider
from engine.providers import mongo_cache


class UnifiedFootballProvider(DataProvider):
    """
    Single 'Football' provider that combines:
    - StatsBomb: deep event data (xG, pass coordinates, etc.) for available matches
    - Football-Data.org: broad coverage (12 leagues, all seasons) for everything else
    Uses MongoDB Atlas for caching.
    """

    def __init__(self, **kwargs):
        self._sb = None
        self._fd = None

    def _get_statsbomb(self):
        if self._sb is None:
            from engine.providers.statsbomb import StatsBombProvider
            self._sb = StatsBombProvider()
        return self._sb

    def _get_footballdata(self):
        if self._fd is None:
            from engine.providers.footballdata import FootballDataProvider
            self._fd = FootballDataProvider()
        return self._fd

    def get_sport(self) -> str:
        return "football"

    def list_competitions(self) -> pd.DataFrame:
        """Merge competitions from both sources, deduplicated."""
        # Check MongoDB cache first
        cached = mongo_cache.get_cached_competitions("football")
        if cached:
            return pd.DataFrame(cached)

        all_comps = []

        # StatsBomb competitions (tagged with source)
        try:
            sb = self._get_statsbomb().list_competitions()
            sb["source"] = "statsbomb"
            sb["has_deep_data"] = True
            all_comps.append(sb)
        except Exception:
            pass

        # Football-Data competitions
        try:
            fd = self._get_footballdata().list_competitions()
            fd["source"] = "football-data"
            fd["has_deep_data"] = False
            all_comps.append(fd)
        except Exception:
            pass

        if not all_comps:
            return pd.DataFrame(columns=["competition_id", "competition_name", "season_id", "season_name", "country", "source", "has_deep_data"])

        combined = pd.concat(all_comps, ignore_index=True)

        # Deduplicate: prefer StatsBomb entries (they have deeper data)
        # For same competition+season, keep StatsBomb if available
        combined["sort_key"] = combined["has_deep_data"].apply(lambda x: 0 if x else 1)
        combined = combined.sort_values(["competition_name", "season_id", "sort_key"])
        combined = combined.drop_duplicates(subset=["competition_name", "season_id"], keep="first")
        combined = combined.drop(columns=["sort_key"])

        combined = combined.sort_values(["competition_name", "season_id"]).reset_index(drop=True)

        # Cache to MongoDB
        mongo_cache.cache_competitions("football", combined.to_dict(orient="records"))

        return combined

    def list_matches(self, competition_id, season_id=None) -> pd.DataFrame:
        """List matches, choosing the right backend based on competition source."""
        # Check MongoDB cache
        cached = mongo_cache.get_cached_matches("football", competition_id, season_id)
        if cached:
            return pd.DataFrame(cached)

        # Determine which backend to use
        comps = self.list_competitions()
        match_row = comps[
            (comps["competition_id"].astype(str) == str(competition_id)) &
            (comps["season_id"].astype(str) == str(season_id))
        ]

        if not match_row.empty and match_row.iloc[0].get("source") == "statsbomb":
            matches = self._get_statsbomb().list_matches(competition_id, season_id)
            matches["source"] = "statsbomb"
        else:
            matches = self._get_footballdata().list_matches(competition_id, season_id)
            matches["source"] = "football-data"

        # Cache
        mongo_cache.cache_matches("football", competition_id, season_id, matches.to_dict(orient="records"))

        return matches

    def get_match_events(self, match_id, source: str = None) -> pd.DataFrame:
        """
        Get match events. Tries StatsBomb first (deep data), falls back to Football-Data.
        Results are cached in MongoDB for instant subsequent loads.
        """
        # Check MongoDB cache first (instant!)
        cached = mongo_cache.get_cached_events("football", match_id)
        if cached is not None:
            return cached

        df = None

        # Try StatsBomb first if source suggests it or if match_id looks like StatsBomb
        if source == "statsbomb" or (isinstance(match_id, (int, str)) and str(match_id).isdigit() and len(str(match_id)) >= 7):
            try:
                df = self._get_statsbomb().get_match_events(match_id)
            except Exception:
                pass

        # Fall back to Football-Data
        if df is None:
            try:
                df = self._get_footballdata().get_match_events(match_id)
            except Exception as e:
                if df is None:
                    raise ValueError(f"Could not fetch match {match_id} from any source: {e}")

        # Cache in MongoDB permanently
        mongo_cache.cache_events("football", match_id, df, "football")

        return df
