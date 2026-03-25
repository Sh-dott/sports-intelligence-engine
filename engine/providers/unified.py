"""
Unified Football Provider
Merges StatsBomb (deep analytics) and Football-Data.org (broad coverage)
into a single "Football" data source with MongoDB caching.
Handles name normalization and deduplication across sources.
"""

import pandas as pd

from engine.providers.base import DataProvider
from engine.providers import mongo_cache

# Normalize competition names across providers
NAME_NORMALIZATION = {
    # StatsBomb -> canonical
    "1. Bundesliga": "Bundesliga",
    "Champions League": "UEFA Champions League",
    "UEFA Euro": "European Championship",
    "Indian Super league": "Indian Super League",
    "FA Women's Super League": "FA WSL",
    "Liga Profesional": "Liga Profesional Argentina",
    # Football-Data -> canonical
    "Primera Division": "La Liga",
    "Primeira Liga": "Liga Portugal",
}


class UnifiedFootballProvider(DataProvider):
    """
    Single 'Football' provider combining StatsBomb + Football-Data.org.
    StatsBomb provides deep event data (xG, coordinates) for available matches.
    Football-Data provides broad coverage (12+ leagues, decades of history).
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

    def _normalize_name(self, name: str) -> str:
        return NAME_NORMALIZATION.get(name, name)

    def list_competitions(self) -> pd.DataFrame:
        """Merge competitions from both sources, properly deduplicated."""
        cached = mongo_cache.get_cached_competitions("football")
        if cached:
            return pd.DataFrame(cached)

        sb_comps = pd.DataFrame()
        fd_comps = pd.DataFrame()

        # StatsBomb competitions
        try:
            sb_comps = self._get_statsbomb().list_competitions().copy()
            sb_comps["source"] = "statsbomb"
            sb_comps["has_deep_data"] = True
            # Store original IDs for later lookup
            sb_comps["original_competition_id"] = sb_comps["competition_id"]
            sb_comps["original_season_id"] = sb_comps["season_id"]
            sb_comps["competition_name"] = sb_comps["competition_name"].apply(self._normalize_name)
        except Exception:
            pass

        # Football-Data competitions
        try:
            fd_comps = self._get_footballdata().list_competitions().copy()
            fd_comps["source"] = "football-data"
            fd_comps["has_deep_data"] = False
            fd_comps["original_competition_id"] = fd_comps["competition_id"]
            fd_comps["original_season_id"] = fd_comps["season_id"]
            fd_comps["competition_name"] = fd_comps["competition_name"].apply(self._normalize_name)
        except Exception:
            pass

        frames = [f for f in [sb_comps, fd_comps] if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["competition_id", "competition_name", "season_id",
                                         "season_name", "country", "source", "has_deep_data"])

        combined = pd.concat(frames, ignore_index=True)

        # Build unified competition IDs: use competition_name as the grouping key
        # For season display, use the season_name (which is human-readable)
        # Deduplicate: same competition_name + same season_name -> keep StatsBomb (deeper data)
        combined["dedup_key"] = combined["competition_name"] + "|" + combined["season_name"]
        combined["priority"] = combined["has_deep_data"].apply(lambda x: 0 if x else 1)
        combined = combined.sort_values(["dedup_key", "priority"])
        combined = combined.drop_duplicates(subset=["dedup_key"], keep="first")
        combined = combined.drop(columns=["dedup_key", "priority"])

        # Use competition_name as competition_id for the unified view
        # Keep original IDs in separate columns for backend lookups
        combined["competition_id"] = combined["competition_name"]
        combined = combined.sort_values(["competition_name", "season_name"]).reset_index(drop=True)

        mongo_cache.cache_competitions("football", combined.to_dict(orient="records"))
        return combined

    def list_matches(self, competition_id, season_id=None) -> pd.DataFrame:
        """List matches, routing to the correct backend."""
        cached = mongo_cache.get_cached_matches("football", competition_id, season_id)
        if cached:
            return pd.DataFrame(cached)

        comps = self.list_competitions()

        # Find the matching entry to determine source and original IDs
        mask = comps["competition_name"] == str(competition_id)
        if season_id:
            mask = mask & (comps["season_id"].astype(str) == str(season_id))

        match_rows = comps[mask]
        if match_rows.empty:
            # Try original_competition_id
            mask = comps["original_competition_id"].astype(str) == str(competition_id)
            if season_id:
                mask = mask & (comps["original_season_id"].astype(str) == str(season_id))
            match_rows = comps[mask]

        if match_rows.empty:
            return pd.DataFrame(columns=["match_id", "home_team", "away_team", "match_date",
                                         "score_home", "score_away", "score", "source"])

        row = match_rows.iloc[0]
        source = row.get("source", "football-data")
        orig_comp = row.get("original_competition_id", competition_id)
        orig_season = row.get("original_season_id", season_id)

        if source == "statsbomb":
            matches = self._get_statsbomb().list_matches(orig_comp, orig_season)
            matches["source"] = "statsbomb"
        else:
            matches = self._get_footballdata().list_matches(orig_comp, orig_season)
            matches["source"] = "football-data"

        mongo_cache.cache_matches("football", competition_id, season_id, matches.to_dict(orient="records"))
        return matches

    def get_match_events(self, match_id, source: str = None) -> pd.DataFrame:
        """Get match events with MongoDB caching for instant repeat loads."""
        cached = mongo_cache.get_cached_events("football", match_id)
        if cached is not None:
            return cached

        df = None

        # Try StatsBomb first for deep data (their match IDs are 7-digit numbers)
        if source == "statsbomb" or (str(match_id).isdigit() and len(str(match_id)) >= 7):
            try:
                df = self._get_statsbomb().get_match_events(match_id)
            except Exception:
                pass

        # Fall back to Football-Data
        if df is None:
            try:
                df = self._get_footballdata().get_match_events(match_id)
            except Exception as e:
                raise ValueError(f"Could not fetch match {match_id}: {e}")

        mongo_cache.cache_events("football", match_id, df, "football")
        return df
