"""
Data Ingestion Module
Handles loading, validating, and normalizing match event data from CSV/JSON sources.
"""

import json
import csv
import io
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = {"timestamp", "team", "player", "event_type"}
OPTIONAL_COLUMNS = {"location_x", "location_y", "detail", "period", "match_id",
                     "xg", "end_x", "end_y", "pass_outcome", "pass_recipient",
                     "under_pressure", "score_home", "score_away"}

VALID_EVENT_TYPES = {
    # Football events
    "pass", "shot", "shot_on_target", "goal", "save", "tackle", "foul",
    "free_kick", "corner", "throw_in", "offside", "yellow_card", "red_card",
    "substitution", "cross", "header", "dribble", "interception", "clearance",
    "goal_kick", "penalty", "penalty_miss",
    # Basketball events
    "field_goal", "field_goal_miss", "three_pointer", "three_pointer_miss",
    "free_throw", "free_throw_miss", "rebound", "assist", "steal", "block",
    "turnover", "personal_foul", "technical_foul", "timeout",
    # StatsBomb additional
    "carry", "pressure_event",
    # Common
    "possession_change", "kickoff", "halftime", "fulltime", "tip_off",
}

EVENT_TYPE_ALIASES = {
    "sg": "shot_on_target", "sot": "shot_on_target",
    "yc": "yellow_card", "rc": "red_card",
    "fg": "field_goal", "fgm": "field_goal_miss",
    "3pt": "three_pointer", "3pm": "three_pointer_miss",
    "ft": "free_throw", "ftm": "free_throw_miss",
    "reb": "rebound", "ast": "assist", "stl": "steal",
    "blk": "block", "to": "turnover", "pf": "personal_foul",
    "sub": "substitution", "int": "interception",
}


class IngestionError(Exception):
    """Raised when data ingestion fails validation."""
    pass


def load_match_data(source: str | Path | dict | list, sport: Optional[str] = None) -> pd.DataFrame:
    """
    Load match event data from CSV file, JSON file, JSON string, or dict/list.

    Args:
        source: File path (CSV/JSON), JSON string, dict, or list of event dicts.
        sport: Optional sport type hint ('football' or 'basketball').

    Returns:
        Cleaned and validated DataFrame of match events.
    """
    df = _parse_source(source)
    df = _normalize_columns(df)
    _validate_schema(df)
    df = _normalize_event_types(df)
    df = _parse_timestamps(df)
    df = _fill_defaults(df)

    if sport:
        df.attrs["sport"] = sport
    else:
        df.attrs["sport"] = _detect_sport(df)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _parse_source(source) -> pd.DataFrame:
    """Parse data from various input formats."""
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, (list, dict)):
        data = source if isinstance(source, list) else [source]
        return pd.DataFrame(data)

    source_str = str(source)

    # Try as file path
    path = Path(source_str)
    if path.exists():
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "events" in data:
                data = data["events"]
            return pd.DataFrame(data)
        else:
            raise IngestionError(f"Unsupported file format: {path.suffix}")

    # Try as JSON string
    try:
        data = json.loads(source_str)
        if isinstance(data, dict) and "events" in data:
            data = data["events"]
        if isinstance(data, list):
            return pd.DataFrame(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try as CSV string
    try:
        return pd.read_csv(io.StringIO(source_str))
    except Exception:
        pass

    raise IngestionError(f"Could not parse source: {type(source)}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_").str.replace("-", "_")

    rename_map = {
        "time": "timestamp", "minute": "timestamp", "event": "event_type",
        "type": "event_type", "action": "event_type", "name": "player",
        "player_name": "player", "team_name": "team", "loc_x": "location_x",
        "loc_y": "location_y", "x": "location_x", "y": "location_y",
        "game_id": "match_id", "half": "period", "quarter": "period",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns})
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Ensure required columns exist."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise IngestionError(f"Missing required columns: {missing}")

    if df.empty:
        raise IngestionError("Dataset is empty")


def _normalize_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize event type names."""
    df["event_type"] = (
        df["event_type"]
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .replace(EVENT_TYPE_ALIASES)
    )

    unknown = set(df["event_type"].unique()) - VALID_EVENT_TYPES
    if unknown:
        print(f"[ingestion] Warning: Unknown event types will be kept: {unknown}")

    return df


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps to numeric minutes."""
    ts = df["timestamp"]

    # Already numeric
    if pd.api.types.is_numeric_dtype(ts):
        df["timestamp"] = ts.astype(float)
        return df

    # Try MM:SS format
    try:
        parts = ts.str.split(":")
        df["timestamp"] = parts.str[0].astype(float) + parts.str[1].astype(float) / 60
        return df
    except Exception:
        pass

    # Try datetime parsing
    try:
        dt = pd.to_datetime(ts)
        start = dt.min()
        df["timestamp"] = (dt - start).dt.total_seconds() / 60
        return df
    except Exception:
        pass

    raise IngestionError("Could not parse timestamp column")


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill optional columns with defaults."""
    if "location_x" not in df.columns:
        df["location_x"] = np.nan
    if "location_y" not in df.columns:
        df["location_y"] = np.nan
    if "detail" not in df.columns:
        df["detail"] = ""
    if "period" not in df.columns:
        df["period"] = 1
    if "match_id" not in df.columns:
        df["match_id"] = "match_001"

    df["team"] = df["team"].fillna("Unknown")
    df["player"] = df["player"].fillna("Unknown")
    df["detail"] = df["detail"].fillna("")

    return df


def _detect_sport(df: pd.DataFrame) -> str:
    """Auto-detect sport based on event types present."""
    events = set(df["event_type"].unique())
    basketball_events = {"field_goal", "three_pointer", "free_throw", "rebound",
                         "steal", "block", "field_goal_miss", "three_pointer_miss",
                         "free_throw_miss", "tip_off"}
    football_events = {"goal", "corner", "throw_in", "offside", "goal_kick",
                       "penalty", "kickoff", "cross", "header", "clearance"}

    b_count = len(events & basketball_events)
    f_count = len(events & football_events)

    if b_count > f_count:
        return "basketball"
    return "football"
