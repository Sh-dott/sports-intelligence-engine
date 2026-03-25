"""
Multi-Match Analysis Module
Season-level analysis: team form, player tracking, league benchmarks, head-to-head.
"""

from dataclasses import dataclass, field
from typing import Optional
import json

import pandas as pd
import numpy as np

from engine.processing import process_match, MatchContext
from engine.detection import DetectionEngine
from engine.statistics import StatisticalAnalyzer
from engine.insights import InsightGenerator, report_to_json


@dataclass
class SeasonContext:
    """Aggregated data across multiple matches."""
    provider: str
    competition: str
    season: str
    match_count: int
    match_summaries: list[dict] = field(default_factory=list)
    team_form: dict = field(default_factory=dict)
    team_aggregates: dict = field(default_factory=dict)
    player_aggregates: dict = field(default_factory=dict)
    league_averages: dict = field(default_factory=dict)


def analyze_season(provider_name: str, competition_id, season_id,
                   max_matches: int = 20, progress_callback=None) -> SeasonContext:
    """
    Analyze multiple matches from a competition/season.

    Args:
        provider_name: 'statsbomb' or 'nba'
        competition_id: Competition identifier
        season_id: Season identifier
        max_matches: Maximum number of matches to analyze (for performance)
        progress_callback: Optional function called with (current, total) for progress updates

    Returns:
        SeasonContext with aggregated analysis
    """
    from engine.providers import get_provider
    from engine.ingestion import load_match_data

    provider = get_provider(provider_name)
    matches = provider.list_matches(competition_id=competition_id, season_id=season_id)

    if matches.empty:
        raise ValueError("No matches found for this competition/season")

    # Limit matches
    if len(matches) > max_matches:
        matches = matches.head(max_matches)

    ctx = SeasonContext(
        provider=provider_name,
        competition=str(competition_id),
        season=str(season_id),
        match_count=len(matches),
    )

    # Analyze each match
    all_team_stats = []
    all_player_stats = []
    match_results = {}  # team -> [(match_id, goals_for, goals_against)]

    for idx, (_, match) in enumerate(matches.iterrows()):
        match_id = match["match_id"]
        if progress_callback:
            progress_callback(idx + 1, len(matches))

        try:
            df = provider.get_match_events(match_id)
            sport = provider.get_sport()
            processed = load_match_data(df, sport=sport)
            match_ctx = process_match(processed)

            # Collect match summary
            scoring = match_ctx.scoring_timeline
            final_scores = {}
            if scoring is not None and not scoring.empty:
                for team in match_ctx.teams:
                    team_scoring = scoring[scoring["team"] == team]
                    final_scores[team] = int(team_scoring["running_score"].max()) if not team_scoring.empty else 0

            summary = {
                "match_id": str(match_id),
                "home_team": match.get("home_team", ""),
                "away_team": match.get("away_team", ""),
                "match_date": str(match.get("match_date", "")),
                "score": final_scores,
                "duration": round(match_ctx.duration_minutes, 1),
                "total_events": match_ctx.total_events,
            }

            # Add xG if available
            if match_ctx.xg_analysis and "teams" in match_ctx.xg_analysis:
                summary["xg"] = {
                    team: data["xg"] for team, data in match_ctx.xg_analysis["teams"].items()
                }

            ctx.match_summaries.append(summary)

            # Collect team stats
            if match_ctx.team_stats is not None:
                ts = match_ctx.team_stats.copy()
                ts["match_id"] = str(match_id)
                all_team_stats.append(ts)

            # Collect player stats
            if match_ctx.player_stats is not None:
                ps = match_ctx.player_stats.copy()
                ps["match_id"] = str(match_id)
                all_player_stats.append(ps)

            # Track results for form calculation
            for team in match_ctx.teams:
                if team not in match_results:
                    match_results[team] = []
                goals_for = final_scores.get(team, 0)
                goals_against = sum(v for k, v in final_scores.items() if k != team)
                match_results[team].append({
                    "match_id": str(match_id),
                    "date": str(match.get("match_date", "")),
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "result": "W" if goals_for > goals_against else ("D" if goals_for == goals_against else "L"),
                })

        except Exception as e:
            print(f"  [!] Error analyzing match {match_id}: {e}")
            continue

    # Aggregate team stats
    if all_team_stats:
        combined_ts = pd.concat(all_team_stats, ignore_index=True)
        ctx.team_aggregates = _aggregate_team_stats(combined_ts)

    # Aggregate player stats
    if all_player_stats:
        combined_ps = pd.concat(all_player_stats, ignore_index=True)
        ctx.player_aggregates = _aggregate_player_stats(combined_ps)

    # Compute form
    ctx.team_form = _compute_form(match_results)

    # Compute league averages
    ctx.league_averages = _compute_league_averages(ctx)

    return ctx


def _aggregate_team_stats(df: pd.DataFrame) -> dict:
    """Aggregate team stats across multiple matches."""
    result = {}
    numeric_cols = ["total_events", "scoring_events", "turnovers", "pressure_events", "mistakes"]
    available = [c for c in numeric_cols if c in df.columns]

    for team, group in df.groupby("team"):
        matches_played = len(group)
        stats = {
            "matches_played": matches_played,
        }
        for col in available:
            stats[f"total_{col}"] = int(group[col].sum())
            stats[f"avg_{col}"] = round(float(group[col].mean()), 1)

        # Win/Draw/Loss not directly available, computed from form
        result[team] = stats

    return result


def _aggregate_player_stats(df: pd.DataFrame) -> dict:
    """Aggregate player stats across matches."""
    result = {}
    for (team, player), group in df.groupby(["team", "player"]):
        if player == "Unknown":
            continue
        result[f"{player} ({team})"] = {
            "team": team,
            "player": player,
            "matches": len(group),
            "total_events": int(group["total_events"].sum()),
            "total_scoring": int(group["scoring_events"].sum()),
            "total_mistakes": int(group["mistakes"].sum()),
            "avg_events_per_match": round(float(group["total_events"].mean()), 1),
            "avg_mistake_rate": round(float(group["mistake_rate"].mean()), 3) if "mistake_rate" in group.columns else 0,
        }

    # Sort by total events
    result = dict(sorted(result.items(), key=lambda x: x[1]["total_events"], reverse=True))
    return result


def _compute_form(match_results: dict) -> dict:
    """Compute team form: points, win rate, streaks."""
    form = {}
    for team, results in match_results.items():
        wins = sum(1 for r in results if r["result"] == "W")
        draws = sum(1 for r in results if r["result"] == "D")
        losses = sum(1 for r in results if r["result"] == "L")
        points = wins * 3 + draws
        goals_for = sum(r["goals_for"] for r in results)
        goals_against = sum(r["goals_against"] for r in results)
        matches = len(results)

        # Current streak
        if results:
            current_result = results[-1]["result"]
            streak = 0
            for r in reversed(results):
                if r["result"] == current_result:
                    streak += 1
                else:
                    break
            streak_str = f"{current_result}{streak}"
        else:
            streak_str = ""

        # Form string (last 5)
        form_str = "".join(r["result"] for r in results[-5:])

        form[team] = {
            "matches": matches,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "points": points,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_difference": goals_for - goals_against,
            "win_rate": round(wins / max(matches, 1), 2),
            "points_per_match": round(points / max(matches, 1), 2),
            "form": form_str,
            "current_streak": streak_str,
        }

    # Sort by points, then goal difference
    form = dict(sorted(form.items(), key=lambda x: (x[1]["points"], x[1]["goal_difference"]), reverse=True))
    return form


def _compute_league_averages(ctx: SeasonContext) -> dict:
    """Compute league-wide averages from team aggregates."""
    if not ctx.team_aggregates:
        return {}

    averages = {}
    all_stats = list(ctx.team_aggregates.values())

    for key in ["avg_scoring_events", "avg_turnovers", "avg_pressure_events", "avg_mistakes"]:
        values = [s.get(key, 0) for s in all_stats if key in s]
        if values:
            averages[key] = round(np.mean(values), 2)
            averages[f"{key}_std"] = round(np.std(values), 2)

    return averages


def season_to_json(ctx: SeasonContext) -> str:
    """Serialize SeasonContext to JSON."""
    return json.dumps({
        "provider": ctx.provider,
        "competition": ctx.competition,
        "season": ctx.season,
        "match_count": ctx.match_count,
        "matches_analyzed": len(ctx.match_summaries),
        "match_summaries": ctx.match_summaries,
        "team_form": ctx.team_form,
        "team_aggregates": ctx.team_aggregates,
        "player_aggregates": dict(list(ctx.player_aggregates.items())[:50]),  # Top 50
        "league_averages": ctx.league_averages,
    }, indent=2, default=str)
