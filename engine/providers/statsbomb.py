"""
StatsBomb Data Provider
Fetches real football match data from StatsBomb's free open data via statsbombpy.
3000+ matches across World Cup, Euro, Premier League, La Liga, Champions League, etc.
"""

import warnings

import pandas as pd
import numpy as np

from engine.providers.base import DataProvider
from engine.providers.cache import MatchCache

# Suppress the NoAuthWarning from statsbombpy
warnings.filterwarnings("ignore", message="credentials were not supplied")

# StatsBomb event type -> engine event type mapping
EVENT_TYPE_MAP = {
    "Pass": "pass",
    "Shot": "shot",              # Refined below based on shot_outcome
    "Dribble": "dribble",
    "Carry": "carry",
    "Ball Receipt*": None,       # Skip — redundant with pass
    "Pressure": "pressure_event",
    "Ball Recovery": "interception",
    "Duel": "tackle",
    "Block": "block",
    "Clearance": "clearance",
    "Interception": "interception",
    "Foul Committed": "foul",
    "Foul Won": "free_kick",
    "Goal Keeper": "save",
    "Miscontrol": "turnover",
    "Dispossessed": "turnover",
    "Dribbled Past": None,       # Skip — opponent perspective of dribble
    "Substitution": "substitution",
    "Offside": "offside",
    "Bad Behaviour": "foul",
    "50/50": "tackle",
    "Shield": None,              # Skip
    "Injury Stoppage": None,     # Skip
    "Tactical Shift": None,      # Skip
    "Starting XI": None,         # Skip
    "Half Start": None,          # Skip
    "Half End": None,            # Skip
    "Player On": "substitution",
    "Player Off": "substitution",
    "Error": "turnover",
    "Own Goal Against": "goal",
    "Own Goal For": "goal",
}

# StatsBomb pitch is 120x80 yards — normalize to 0-100
SB_PITCH_X = 120.0
SB_PITCH_Y = 80.0


class StatsBombProvider(DataProvider):
    """Provider for StatsBomb free open data."""

    def __init__(self, **kwargs):
        self.cache = MatchCache("statsbomb")

    def get_sport(self) -> str:
        return "football"

    def list_competitions(self) -> pd.DataFrame:
        from statsbombpy import sb

        comps = sb.competitions()
        return comps[[
            "competition_id", "competition_name", "season_id",
            "season_name", "country_name"
        ]].rename(columns={"country_name": "country"}).sort_values(
            ["competition_name", "season_name"]
        ).reset_index(drop=True)

    def list_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        from statsbombpy import sb

        matches = sb.matches(competition_id=int(competition_id), season_id=int(season_id))
        result = matches[[
            "match_id", "home_team", "away_team", "match_date",
            "home_score", "away_score"
        ]].copy()
        result["score"] = result["home_score"].astype(str) + "-" + result["away_score"].astype(str)
        return result.sort_values("match_date").reset_index(drop=True)

    def get_match_events(self, match_id) -> pd.DataFrame:
        match_id = int(match_id)

        # Check cache first
        cached = self.cache.get(match_id)
        if cached is not None:
            cached.attrs["sport"] = "football"
            return cached

        # Fetch from API
        from statsbombpy import sb
        raw = sb.events(match_id=match_id)

        if raw.empty:
            raise ValueError(f"No events found for match {match_id}")

        df = self._normalize(raw, match_id)
        df = self.validate_output(df)

        # Cache for future use
        self.cache.put(match_id, df)

        df.attrs["sport"] = "football"
        return df

    def _normalize(self, raw: pd.DataFrame, match_id: int) -> pd.DataFrame:
        """Transform StatsBomb event DataFrame to engine schema."""
        rows = []

        for _, event in raw.iterrows():
            sb_type = event.get("type")
            engine_type = EVENT_TYPE_MAP.get(sb_type)

            if engine_type is None:
                continue

            # Refine shot events based on outcome
            if sb_type == "Shot":
                outcome = event.get("shot_outcome", "")
                if outcome == "Goal":
                    engine_type = "goal"
                elif outcome == "Saved":
                    engine_type = "shot_on_target"
                else:
                    engine_type = "shot"

            # Refine foul with card
            card = event.get("foul_committed_card") or event.get("bad_behaviour_card")
            if card == "Yellow Card":
                engine_type = "yellow_card"
            elif card in ("Red Card", "Second Yellow"):
                engine_type = "red_card"

            # Check for penalty (goals from penalties are still goals)
            if sb_type == "Shot" and event.get("shot_type") == "Penalty":
                if event.get("shot_outcome") != "Goal":
                    engine_type = "penalty_miss"
                # If it's a goal, keep engine_type = "goal" (set above)

            # Extract location (StatsBomb: [x, y] on 120x80 pitch)
            loc = event.get("location")
            loc_x = np.nan
            loc_y = np.nan
            if isinstance(loc, list) and len(loc) >= 2:
                loc_x = (loc[0] / SB_PITCH_X) * 100
                loc_y = (loc[1] / SB_PITCH_Y) * 100

            # Extract pass end location
            end_loc = event.get("pass_end_location")
            end_x = np.nan
            end_y = np.nan
            if isinstance(end_loc, list) and len(end_loc) >= 2:
                end_x = (end_loc[0] / SB_PITCH_X) * 100
                end_y = (end_loc[1] / SB_PITCH_Y) * 100

            # Also handle carry end locations
            carry_end = event.get("carry_end_location")
            if isinstance(carry_end, list) and len(carry_end) >= 2 and pd.isna(end_x):
                end_x = (carry_end[0] / SB_PITCH_X) * 100
                end_y = (carry_end[1] / SB_PITCH_Y) * 100

            # Build detail string
            detail_parts = []
            if event.get("play_pattern"):
                detail_parts.append(str(event["play_pattern"]))
            if event.get("pass_type") and pd.notna(event.get("pass_type")):
                detail_parts.append(str(event["pass_type"]))
            if event.get("shot_technique") and pd.notna(event.get("shot_technique")):
                detail_parts.append(str(event["shot_technique"]))
            if event.get("shot_body_part") and pd.notna(event.get("shot_body_part")):
                detail_parts.append(str(event["shot_body_part"]))

            # Pass outcome: NaN = successful, otherwise = failed
            pass_outcome_raw = event.get("pass_outcome")
            pass_outcome = "complete" if pd.isna(pass_outcome_raw) and sb_type == "Pass" else str(pass_outcome_raw) if pd.notna(pass_outcome_raw) else ""

            row = {
                "timestamp": event.get("minute", 0) + event.get("second", 0) / 60.0,
                "team": str(event.get("team", "Unknown")),
                "player": str(event.get("player", "Unknown")) if pd.notna(event.get("player")) else "Unknown",
                "event_type": engine_type,
                "location_x": loc_x,
                "location_y": loc_y,
                "end_x": end_x,
                "end_y": end_y,
                "detail": "; ".join(detail_parts) if detail_parts else "",
                "period": int(event.get("period", 1)),
                "match_id": str(match_id),
                "xg": event.get("shot_statsbomb_xg", np.nan),
                "pass_outcome": pass_outcome,
                "pass_recipient": str(event.get("pass_recipient", "")) if pd.notna(event.get("pass_recipient")) else "",
                "under_pressure": bool(event.get("under_pressure", False)),
            }

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
