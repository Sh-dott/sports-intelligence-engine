"""
Advanced Metrics Module
Computes xG analysis, progressive passes, PPDA, pass networks, possession chains,
and pass completion rates from enriched match data (StatsBomb).
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_xg_analysis(df: pd.DataFrame) -> Optional[dict]:
    """
    Analyze Expected Goals (xG) vs actual goals.
    Requires 'xg' column from StatsBomb data.
    """
    if "xg" not in df.columns or df["xg"].isna().all():
        return None

    shots = df[df["event_type"].isin(["shot", "shot_on_target", "goal"])].copy()
    if shots.empty:
        return None

    result = {"teams": {}, "top_chances": [], "total_xg": 0, "total_goals": 0}

    for team in df["team"].unique():
        team_shots = shots[shots["team"] == team]
        if team_shots.empty:
            continue

        team_xg = team_shots["xg"].sum()
        team_goals = (team_shots["event_type"] == "goal").sum()
        diff = team_goals - team_xg

        result["teams"][team] = {
            "xg": round(team_xg, 2),
            "goals": int(team_goals),
            "difference": round(diff, 2),
            "shots": len(team_shots),
            "shots_on_target": int((team_shots["event_type"].isin(["shot_on_target", "goal"])).sum()),
            "xg_per_shot": round(team_xg / max(len(team_shots), 1), 3),
            "assessment": (
                "Clinical finishing" if diff > 1 else
                "Slightly overperforming" if diff > 0.3 else
                "Performing at expected level" if abs(diff) <= 0.3 else
                "Slightly wasteful" if diff > -1 else
                "Very wasteful"
            ),
        }
        result["total_xg"] += team_xg
        result["total_goals"] += team_goals

    # Top missed chances (highest xG non-goals)
    missed = shots[(shots["event_type"] != "goal") & (shots["xg"].notna())].nlargest(5, "xg")
    for _, row in missed.iterrows():
        result["top_chances"].append({
            "minute": round(row["timestamp"], 1),
            "player": row["player"],
            "team": row["team"],
            "xg": round(row["xg"], 3),
            "outcome": row["event_type"],
        })

    result["total_xg"] = round(result["total_xg"], 2)
    return result


def compute_progressive_passes(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Identify progressive passes: passes that advance the ball significantly
    toward the opponent's goal.
    A pass is progressive if end_x is at least 10 units closer to goal (x=100)
    than start location_x, and is in opponent half (location_x > 40).
    """
    if "end_x" not in df.columns or df["end_x"].isna().all():
        return None

    passes = df[(df["event_type"] == "pass") & df["end_x"].notna() & df["location_x"].notna()].copy()
    if passes.empty:
        return None

    # Progressive: moves ball significantly forward in attacking areas
    passes["distance_gained"] = passes["end_x"] - passes["location_x"]
    progressive = passes[
        (passes["distance_gained"] >= 8.33) &  # ~10 yards on 120-yard pitch normalized to 100
        (passes["location_x"] >= 33)  # At least in midfield
    ].copy()

    if progressive.empty:
        return None

    # Aggregate per player
    player_stats = progressive.groupby(["team", "player"]).agg(
        progressive_passes=("distance_gained", "count"),
        avg_distance=("distance_gained", "mean"),
        total_distance=("distance_gained", "sum"),
    ).reset_index().sort_values("progressive_passes", ascending=False)

    return player_stats


def compute_ppda(df: pd.DataFrame, window_minutes: int = 15) -> Optional[dict]:
    """
    Compute Passes Per Defensive Action (PPDA).
    PPDA = opponent passes / team's defensive actions (tackles, interceptions, fouls)
    Lower PPDA = higher pressing intensity.
    Computed per time window and overall.
    """
    teams = sorted(df["team"].unique().tolist())
    if len(teams) < 2:
        return None

    defensive_actions = {"tackle", "interception", "foul", "pressure_event"}

    result = {"teams": {}, "windows": []}

    for team in teams:
        opponents = [t for t in teams if t != team]
        if not opponents:
            continue
        opponent = opponents[0]

        # Team's defensive actions
        team_defense = df[(df["team"] == team) & (df["event_type"].isin(defensive_actions))]
        # Opponent's passes
        opp_passes = df[(df["team"] == opponent) & (df["event_type"] == "pass")]

        total_def = len(team_defense)
        total_opp_passes = len(opp_passes)
        ppda = total_opp_passes / max(total_def, 1)

        result["teams"][team] = {
            "ppda": round(ppda, 2),
            "defensive_actions": total_def,
            "opponent_passes": total_opp_passes,
            "pressing_intensity": (
                "Very high (gegenpressing)" if ppda < 6 else
                "High" if ppda < 10 else
                "Medium" if ppda < 15 else
                "Low (deep block)"
            ),
        }

    # Windowed PPDA
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    bins = np.arange(min_time, max_time + window_minutes, window_minutes)

    for i in range(len(bins) - 1):
        window_start = bins[i]
        window_end = bins[i + 1]
        window_df = df[(df["timestamp"] >= window_start) & (df["timestamp"] < window_end)]

        window_data = {"start": round(window_start, 1), "end": round(window_end, 1)}
        for team in teams:
            opponents = [t for t in teams if t != team]
            if not opponents:
                continue
            opponent = opponents[0]
            team_def = len(window_df[(window_df["team"] == team) & (window_df["event_type"].isin(defensive_actions))])
            opp_pass = len(window_df[(window_df["team"] == opponent) & (window_df["event_type"] == "pass")])
            window_data[f"{team}_ppda"] = round(opp_pass / max(team_def, 1), 2)

        result["windows"].append(window_data)

    return result


def compute_pass_network(df: pd.DataFrame) -> Optional[dict]:
    """
    Build passing network: who passes to whom, with average positions.
    Requires pass_recipient column.
    """
    if "pass_recipient" not in df.columns:
        return None

    passes = df[
        (df["event_type"] == "pass") &
        (df["pass_recipient"].notna()) &
        (df["pass_recipient"] != "") &
        (df["pass_outcome"].isin(["complete", ""]))  # Only successful passes
    ].copy()

    if passes.empty:
        return None

    result = {}

    for team in df["team"].unique():
        team_passes = passes[passes["team"] == team]
        if len(team_passes) < 5:
            continue

        # Edges: passer -> recipient with count
        edges = team_passes.groupby(["player", "pass_recipient"]).size().reset_index(name="count")
        edges = edges.sort_values("count", ascending=False)

        # Nodes: average position per player
        all_events = df[df["team"] == team]
        nodes = all_events.groupby("player").agg(
            avg_x=("location_x", "mean"),
            avg_y=("location_y", "mean"),
            total_events=("event_type", "count"),
            total_passes=("event_type", lambda x: (x == "pass").sum()),
        ).reset_index()

        result[team] = {
            "nodes": nodes.to_dict(orient="records"),
            "edges": edges.head(20).to_dict(orient="records"),  # Top 20 connections
            "total_passes": len(team_passes),
        }

    return result if result else None


def compute_possession_chains(df: pd.DataFrame) -> Optional[list]:
    """
    Identify possession chains: consecutive events by the same team.
    Track how each chain ends (goal, shot, turnover, etc.).
    """
    possession_events = {"pass", "carry", "dribble", "shot", "shot_on_target", "goal",
                         "cross", "header", "free_kick", "corner"}
    chain_breakers = {"interception", "tackle", "clearance", "turnover",
                      "goal", "foul", "offside"}

    chains = []
    current_chain = []
    current_team = None

    for _, event in df.iterrows():
        team = event["team"]
        evt_type = event["event_type"]

        if team != current_team or evt_type in chain_breakers:
            # Save completed chain
            if len(current_chain) >= 3:
                chain_end = current_chain[-1]["event_type"]
                chains.append({
                    "team": current_team,
                    "start_minute": round(current_chain[0]["timestamp"], 1),
                    "end_minute": round(current_chain[-1]["timestamp"], 1),
                    "length": len(current_chain),
                    "end_result": chain_end,
                    "led_to_goal": chain_end == "goal",
                    "led_to_shot": chain_end in ("shot", "shot_on_target", "goal"),
                })

            current_chain = []
            current_team = team

        if evt_type in possession_events or evt_type in chain_breakers:
            current_chain.append({
                "timestamp": event["timestamp"],
                "event_type": evt_type,
                "player": event["player"],
            })

    # Final chain
    if len(current_chain) >= 3:
        chain_end = current_chain[-1]["event_type"]
        chains.append({
            "team": current_team,
            "start_minute": round(current_chain[0]["timestamp"], 1),
            "end_minute": round(current_chain[-1]["timestamp"], 1),
            "length": len(current_chain),
            "end_result": chain_end,
            "led_to_goal": chain_end == "goal",
            "led_to_shot": chain_end in ("shot", "shot_on_target", "goal"),
        })

    return chains if chains else None


def compute_pass_completion(df: pd.DataFrame) -> Optional[dict]:
    """
    Compute pass completion rates per player and per team.
    Requires pass_outcome column.
    """
    if "pass_outcome" not in df.columns:
        return None

    passes = df[df["event_type"] == "pass"].copy()
    if passes.empty:
        return None

    passes["is_complete"] = passes["pass_outcome"].isin(["complete", ""])

    result = {"teams": {}, "players": []}

    # Team level
    for team in passes["team"].unique():
        team_passes = passes[passes["team"] == team]
        total = len(team_passes)
        complete = team_passes["is_complete"].sum()
        result["teams"][team] = {
            "total": int(total),
            "complete": int(complete),
            "rate": round(complete / max(total, 1), 3),
        }

    # Player level
    player_stats = passes.groupby(["team", "player"]).agg(
        total=("is_complete", "count"),
        complete=("is_complete", "sum"),
    ).reset_index()
    player_stats["rate"] = (player_stats["complete"] / player_stats["total"]).round(3)
    player_stats = player_stats[player_stats["total"] >= 5]  # Minimum 5 passes
    player_stats = player_stats.sort_values("rate", ascending=False)

    result["players"] = player_stats.to_dict(orient="records")
    return result
