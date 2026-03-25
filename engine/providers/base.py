"""
Abstract base class for data providers.
Each provider fetches real match data from an API and normalizes it
to the engine's standard DataFrame schema.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from engine.ingestion import REQUIRED_COLUMNS


class DataProvider(ABC):
    """Base class for all sports data providers."""

    @abstractmethod
    def get_sport(self) -> str:
        """Return 'football' or 'basketball'."""
        ...

    @abstractmethod
    def list_competitions(self) -> pd.DataFrame:
        """
        List available competitions/leagues.
        Returns DataFrame with columns: competition_id, competition_name, season_id, season_name, country
        """
        ...

    @abstractmethod
    def list_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """
        List matches for a given competition and season.
        Returns DataFrame with columns: match_id, home_team, away_team, match_date, score_home, score_away
        """
        ...

    @abstractmethod
    def get_match_events(self, match_id) -> pd.DataFrame:
        """
        Fetch and normalize match events into the engine's standard schema.
        Must return DataFrame with at least: timestamp, team, player, event_type
        Optional columns: location_x, location_y, detail, period, match_id,
                          xg, end_x, end_y, pass_outcome, pass_recipient
        """
        ...

    def validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that the provider output matches the engine schema."""
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Provider output missing required columns: {missing}")

        if df.empty:
            raise ValueError("Provider returned empty DataFrame")

        if not pd.api.types.is_numeric_dtype(df["timestamp"]):
            raise ValueError("timestamp column must be numeric (minutes)")

        return df
