"""
Data Processing Module
Time-windowed aggregation, possession tracking, and derived metrics.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class MatchContext:
    """Processed match data with aggregated metrics and derived features."""
    events: pd.DataFrame
    sport: str
    teams: list[str]
    match_id: str
    duration_minutes: float
    total_events: int

    # Aggregated DataFrames
    team_stats: Optional[pd.DataFrame] = None
    player_stats: Optional[pd.DataFrame] = None
    time_segments: Optional[pd.DataFrame] = None
    rolling_metrics: Optional[pd.DataFrame] = None
    possession_timeline: Optional[pd.DataFrame] = None
    scoring_timeline: Optional[pd.DataFrame] = None

    # Advanced metrics (populated when enriched data available)
    xg_analysis: Optional[dict] = None
    progressive_passes: Optional[pd.DataFrame] = None
    ppda: Optional[dict] = None
    pass_network: Optional[dict] = None
    possession_chains: Optional[list] = None
    pass_completion: Optional[dict] = None


SCORING_EVENTS = {
    "football": {"goal"},
    "basketball": {"field_goal", "three_pointer", "free_throw"},
}

TURNOVER_EVENTS = {
    "football": {"interception", "clearance", "tackle"},
    "basketball": {"turnover", "steal"},
}

PRESSURE_EVENTS = {
    "football": {"shot", "shot_on_target", "goal", "corner", "free_kick", "penalty"},
    "basketball": {"field_goal", "three_pointer", "steal", "block", "free_throw"},
}

MISTAKE_EVENTS = {
    "football": {"foul", "yellow_card", "red_card", "offside", "penalty"},
    "basketball": {"turnover", "personal_foul", "technical_foul", "free_throw_miss"},
}


def process_match(df: pd.DataFrame) -> MatchContext:
    """
    Full processing pipeline: clean data, compute aggregates, build context.
    """
    sport = df.attrs.get("sport", "football")
    teams = sorted(df["team"].unique().tolist())
    match_id = df["match_id"].iloc[0] if "match_id" in df.columns else "match_001"

    ctx = MatchContext(
        events=df,
        sport=sport,
        teams=teams,
        match_id=match_id,
        duration_minutes=df["timestamp"].max() - df["timestamp"].min(),
        total_events=len(df),
    )

    ctx.events = _add_derived_columns(ctx.events, sport)
    ctx.team_stats = _compute_team_stats(ctx)
    ctx.player_stats = _compute_player_stats(ctx)
    ctx.time_segments = _compute_time_segments(ctx)
    ctx.rolling_metrics = _compute_rolling_metrics(ctx)
    ctx.possession_timeline = _compute_possession_timeline(ctx)
    ctx.scoring_timeline = _compute_scoring_timeline(ctx)

    # Advanced metrics (when enriched data is available)
    _compute_advanced_metrics(ctx)

    return ctx


def _add_derived_columns(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Add boolean flags and categorizations."""
    scoring = SCORING_EVENTS.get(sport, set())
    turnovers = TURNOVER_EVENTS.get(sport, set())
    pressure = PRESSURE_EVENTS.get(sport, set())
    mistakes = MISTAKE_EVENTS.get(sport, set())

    df["is_scoring"] = df["event_type"].isin(scoring)
    df["is_turnover"] = df["event_type"].isin(turnovers)
    df["is_pressure"] = df["event_type"].isin(pressure)
    df["is_mistake"] = df["event_type"].isin(mistakes)

    # Zone classification based on location_x (0-100 scale)
    df["zone"] = pd.cut(
        df["location_x"],
        bins=[-np.inf, 33, 66, np.inf],
        labels=["defensive", "midfield", "attacking"],
    )

    # Time bucket (5-minute segments)
    df["time_bucket"] = (df["timestamp"] // 5).astype(int) * 5

    return df


def _compute_team_stats(ctx: MatchContext) -> pd.DataFrame:
    """Per-team aggregate statistics."""
    df = ctx.events
    stats = []

    for team in ctx.teams:
        t = df[df["team"] == team]
        row = {
            "team": team,
            "total_events": len(t),
            "scoring_events": t["is_scoring"].sum(),
            "turnovers": t["is_turnover"].sum(),
            "pressure_events": t["is_pressure"].sum(),
            "mistakes": t["is_mistake"].sum(),
            "unique_players": t["player"].nunique(),
        }

        # Event type breakdown
        for evt in t["event_type"].unique():
            row[f"count_{evt}"] = (t["event_type"] == evt).sum()

        stats.append(row)

    return pd.DataFrame(stats).fillna(0)


def _compute_player_stats(ctx: MatchContext) -> pd.DataFrame:
    """Per-player aggregate statistics."""
    df = ctx.events
    stats = []

    for (team, player), group in df.groupby(["team", "player"]):
        row = {
            "team": team,
            "player": player,
            "total_events": len(group),
            "scoring_events": group["is_scoring"].sum(),
            "turnovers": group["is_turnover"].sum(),
            "pressure_events": group["is_pressure"].sum(),
            "mistakes": group["is_mistake"].sum(),
            "active_minutes": group["timestamp"].max() - group["timestamp"].min(),
        }
        stats.append(row)

    result = pd.DataFrame(stats)
    if not result.empty:
        result["events_per_minute"] = np.where(
            result["active_minutes"] > 0,
            result["total_events"] / result["active_minutes"],
            0,
        )
        result["mistake_rate"] = np.where(
            result["total_events"] > 0,
            result["mistakes"] / result["total_events"],
            0,
        )
    return result


def _compute_time_segments(ctx: MatchContext) -> pd.DataFrame:
    """5-minute time segment analysis per team."""
    df = ctx.events
    segments = []

    for bucket, group in df.groupby("time_bucket"):
        for team in ctx.teams:
            t = group[group["team"] == team]
            segments.append({
                "time_bucket": bucket,
                "team": team,
                "events": len(t),
                "scoring": t["is_scoring"].sum(),
                "turnovers": t["is_turnover"].sum(),
                "pressure": t["is_pressure"].sum(),
                "mistakes": t["is_mistake"].sum(),
            })

    return pd.DataFrame(segments)


def _compute_rolling_metrics(ctx: MatchContext, windows: list[int] = [1, 3, 5]) -> pd.DataFrame:
    """Rolling window metrics for each team (1, 3, 5 minute windows)."""
    df = ctx.events
    results = []
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()

    for team in ctx.teams:
        t = df[df["team"] == team].copy()
        t = t.sort_values("timestamp")

        for col in ["is_scoring", "is_pressure", "is_mistake", "is_turnover"]:
            t[col] = t[col].astype(int)

        for window in windows:
            # Bin events into fixed-width time buckets
            bins = np.arange(min_time, max_time + window, window)
            t["_bin"] = pd.cut(t["timestamp"], bins=bins, labels=bins[:-1], include_lowest=True)

            agg = t.groupby("_bin", observed=False).agg({
                "is_scoring": "sum",
                "is_pressure": "sum",
                "is_mistake": "sum",
                "is_turnover": "sum",
                "event_type": "count",
            }).rename(columns={"event_type": "event_count"})

            agg.index = agg.index.astype(float)
            agg.index.name = "timestamp"
            agg["team"] = team
            agg["window_minutes"] = window
            results.append(agg.reset_index())

            t.drop(columns="_bin", inplace=True)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def _compute_possession_timeline(ctx: MatchContext) -> pd.DataFrame:
    """Track which team has possession over time."""
    df = ctx.events
    possession_indicators = {"pass", "shot", "shot_on_target", "goal", "dribble", "cross",
                             "field_goal", "three_pointer", "free_throw", "assist"}

    possession_events = df[df["event_type"].isin(possession_indicators)].copy()
    if possession_events.empty:
        return pd.DataFrame(columns=["timestamp", "team", "duration"])

    possession_events["next_timestamp"] = possession_events["timestamp"].shift(-1)
    possession_events["duration"] = (
        possession_events["next_timestamp"] - possession_events["timestamp"]
    ).clip(upper=2.0)  # Cap at 2 minutes to handle breaks

    return possession_events[["timestamp", "team", "player", "event_type", "duration"]].copy()


def _compute_scoring_timeline(ctx: MatchContext) -> pd.DataFrame:
    """Running score over time."""
    df = ctx.events
    scoring = df[df["is_scoring"]].copy()

    if scoring.empty:
        return pd.DataFrame(columns=["timestamp", "team", "player", "event_type", "running_score"])

    sport = ctx.sport
    if sport == "basketball":
        point_map = {"field_goal": 2, "three_pointer": 3, "free_throw": 1}
        scoring["points"] = scoring["event_type"].map(point_map).fillna(1)
    else:
        scoring["points"] = 1  # Each goal = 1 point

    scoring["running_score"] = scoring.groupby("team")["points"].cumsum()
    return scoring[["timestamp", "team", "player", "event_type", "points", "running_score"]].copy()


def _compute_advanced_metrics(ctx: MatchContext) -> None:
    """Compute advanced metrics if enriched data columns are available."""
    df = ctx.events
    has_xg = "xg" in df.columns and df["xg"].notna().any()
    has_end_loc = "end_x" in df.columns and df["end_x"].notna().any()
    has_pass_outcome = "pass_outcome" in df.columns and (df["pass_outcome"] != "").any()

    if not (has_xg or has_end_loc or has_pass_outcome):
        return

    from engine.advanced_metrics import (
        compute_xg_analysis, compute_progressive_passes, compute_ppda,
        compute_pass_network, compute_possession_chains, compute_pass_completion,
    )

    if has_xg:
        ctx.xg_analysis = compute_xg_analysis(df)

    if has_end_loc:
        ctx.progressive_passes = compute_progressive_passes(df)

    ctx.ppda = compute_ppda(df)
    ctx.possession_chains = compute_possession_chains(df)

    if has_pass_outcome:
        ctx.pass_completion = compute_pass_completion(df)

    if "pass_recipient" in df.columns:
        ctx.pass_network = compute_pass_network(df)
