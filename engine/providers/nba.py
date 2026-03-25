"""
NBA Data Provider
Fetches real basketball play-by-play data from NBA.com via nba_api.
"""

import time
import re

import pandas as pd
import numpy as np

from engine.providers.base import DataProvider
from engine.providers.cache import MatchCache


# NBA PlayByPlayV3 actionType -> engine event type mapping
ACTION_TYPE_MAP = {
    "Made Shot": "field_goal",       # Refined based on shotValue
    "Missed Shot": "field_goal_miss",  # Refined based on description
    "Free Throw": "free_throw",      # Refined based on shotResult
    "Rebound": "rebound",
    "Turnover": "turnover",
    "Foul": "personal_foul",
    "Substitution": "substitution",
    "Timeout": "timeout",
    "Jump Ball": "tip_off",
    "Violation": None,               # Skip
    "period": None,                   # Skip (start/end)
    "Instant Replay": None,          # Skip
    "": None,                         # Block/assist annotations
}


class NBAProvider(DataProvider):
    """Provider for NBA play-by-play data via nba_api."""

    def __init__(self, **kwargs):
        self.cache = MatchCache("nba")
        self._last_request = 0

    def _throttle(self):
        """Rate limit: wait 0.6s between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < 0.6:
            time.sleep(0.6 - elapsed)
        self._last_request = time.time()

    def get_sport(self) -> str:
        return "basketball"

    def list_competitions(self) -> pd.DataFrame:
        """List NBA seasons as 'competitions'."""
        current_year = 2025
        seasons = []
        for year in range(current_year, current_year - 6, -1):
            season_str = f"{year-1}-{str(year)[-2:]}"
            seasons.append({
                "competition_id": "nba",
                "competition_name": "NBA",
                "season_id": season_str,
                "season_name": f"{year-1}-{year} NBA Season",
                "country": "USA",
            })
        return pd.DataFrame(seasons)

    def list_matches(self, competition_id=None, season_id: str = "2024-25") -> pd.DataFrame:
        """List NBA games for a season."""
        from nba_api.stats.endpoints import leaguegamefinder

        self._throttle()
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=str(season_id),
            league_id_nullable="00",
        )
        games = finder.get_data_frames()[0]

        if games.empty:
            return pd.DataFrame(columns=["match_id", "home_team", "away_team", "match_date", "score_home", "score_away"])

        # Each game appears twice (once per team). Deduplicate by GAME_ID.
        # Home team has "vs." in MATCHUP, away has "@"
        home = games[games["MATCHUP"].str.contains("vs.", na=False)].copy()
        away = games[games["MATCHUP"].str.contains("@", na=False)].copy()

        merged = home[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS"]].rename(
            columns={"TEAM_ABBREVIATION": "home_team", "PTS": "score_home"}
        )

        away_slim = away[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
            columns={"TEAM_ABBREVIATION": "away_team", "PTS": "score_away"}
        )

        result = merged.merge(away_slim, on="GAME_ID", how="inner")
        result = result.rename(columns={"GAME_ID": "match_id", "GAME_DATE": "match_date"})
        result["score"] = result["score_home"].astype(str) + "-" + result["score_away"].astype(str)
        return result.sort_values("match_date", ascending=False).reset_index(drop=True)

    def get_match_events(self, match_id) -> pd.DataFrame:
        match_id = str(match_id)

        # Check cache
        cached = self.cache.get(match_id)
        if cached is not None:
            cached.attrs["sport"] = "basketball"
            return cached

        # Fetch from API
        from nba_api.stats.endpoints import playbyplayv3

        self._throttle()
        pbp = playbyplayv3.PlayByPlayV3(game_id=match_id)
        raw = pbp.get_data_frames()[0]

        if raw.empty:
            raise ValueError(f"No play-by-play data for game {match_id}")

        # Also get team info for mapping teamId -> name
        team_map = self._build_team_map(raw)

        df = self._normalize(raw, match_id, team_map)
        df = self.validate_output(df)

        self.cache.put(match_id, df)
        df.attrs["sport"] = "basketball"
        return df

    def _build_team_map(self, raw: pd.DataFrame) -> dict:
        """Map teamTricode to full team name."""
        teams = raw[raw["teamTricode"].notna() & (raw["teamTricode"] != "")]
        tricodes = teams["teamTricode"].unique()
        # Use tricode as team name (e.g., "OKC", "IND")
        return {tc: tc for tc in tricodes}

    def _normalize(self, raw: pd.DataFrame, match_id: str, team_map: dict) -> pd.DataFrame:
        """Transform NBA PlayByPlayV3 to engine schema."""
        rows = []

        for _, event in raw.iterrows():
            action_type = event.get("actionType", "")
            engine_type = ACTION_TYPE_MAP.get(action_type)

            if engine_type is None:
                # Check for blocks/steals in the description (they appear as actionType="")
                desc = str(event.get("description", "")).upper()
                if "BLOCK" in desc and action_type == "":
                    engine_type = "block"
                elif "STEAL" in desc and action_type == "":
                    engine_type = "steal"
                elif "AST" in desc and action_type == "":
                    continue  # Assist annotations — skip (captured in made shot)
                else:
                    continue

            # Refine made/missed shots
            shot_value = event.get("shotValue", 0)
            desc = str(event.get("description", ""))

            if action_type == "Made Shot":
                if shot_value == 3 or "3PT" in desc:
                    engine_type = "three_pointer"
                else:
                    engine_type = "field_goal"
            elif action_type == "Missed Shot":
                if "3PT" in desc:
                    engine_type = "three_pointer_miss"
                else:
                    engine_type = "field_goal_miss"
            elif action_type == "Free Throw":
                shot_result = event.get("shotResult", "")
                if shot_result == "Missed" or "MISS" in desc:
                    engine_type = "free_throw_miss"
                else:
                    engine_type = "free_throw"
            elif action_type == "Foul":
                sub_type = str(event.get("subType", "")).lower()
                if "technical" in sub_type or "flagrant" in sub_type:
                    engine_type = "technical_foul"
                else:
                    engine_type = "personal_foul"

            # Parse clock (format: PT11M44.00S)
            timestamp = self._parse_clock(event.get("clock", ""), event.get("period", 1))

            # Team
            team = event.get("teamTricode", "")
            if not team or pd.isna(team):
                team = "Unknown"

            # Player
            player = event.get("playerName", "Unknown")
            if not player or pd.isna(player):
                player = "Unknown"

            # Location (NBA xLegacy/yLegacy coordinates)
            loc_x_raw = event.get("xLegacy", 0)
            loc_y_raw = event.get("yLegacy", 0)

            # NBA court is ~94x50 feet, xLegacy ranges roughly -250 to 250, yLegacy -50 to 400
            # Normalize to 0-100
            loc_x = np.nan
            loc_y = np.nan
            if loc_x_raw != 0 or loc_y_raw != 0:
                loc_x = ((loc_x_raw + 250) / 500) * 100  # -250..250 -> 0..100
                loc_y = ((loc_y_raw + 50) / 450) * 100    # -50..400 -> 0..100

            row = {
                "timestamp": timestamp,
                "team": str(team),
                "player": str(player),
                "event_type": engine_type,
                "location_x": loc_x,
                "location_y": loc_y,
                "detail": desc.strip(),
                "period": int(event.get("period", 1)),
                "match_id": match_id,
                "score_home": event.get("scoreHome", ""),
                "score_away": event.get("scoreAway", ""),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        # Filter out events with no team (team rebounds, neutral events)
        df = df[df["team"] != "Unknown"].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _parse_clock(self, clock_str: str, period: int) -> float:
        """
        Convert NBA clock string to match minutes.
        Clock format: 'PT12M00.00S' (ISO 8601 duration)
        NBA quarters are 12 minutes. Clock counts DOWN.
        """
        if not clock_str or pd.isna(clock_str):
            return (period - 1) * 12.0

        match = re.match(r"PT(\d+)M([\d.]+)S", str(clock_str))
        if not match:
            return (period - 1) * 12.0

        minutes_remaining = int(match.group(1))
        seconds_remaining = float(match.group(2))

        # Elapsed time in this period
        elapsed = 12.0 - minutes_remaining - seconds_remaining / 60.0

        # Add completed periods (12 min each, OT = 5 min)
        if period <= 4:
            total = (period - 1) * 12.0 + elapsed
        else:
            # Overtime periods are 5 minutes
            total = 48.0 + (period - 5) * 5.0 + min(elapsed, 5.0)

        return round(total, 3)
