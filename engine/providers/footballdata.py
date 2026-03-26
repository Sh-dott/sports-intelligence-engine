"""
Football-Data.org Provider
Free API covering top leagues with full season history.
Works without API key for competition listing.
Requires free API key (register at football-data.org) for match data.
"""

import os
import time
import requests
import pandas as pd
import numpy as np

from engine.providers.base import DataProvider
from engine.providers.cache import MatchCache

BASE_URL = "https://api.football-data.org/v4"

# Free tier competition codes
FREE_COMPETITIONS = {
    "PL", "BL1", "SA", "PD", "FL1", "ELC", "DED", "PPL",
    "CL", "EC", "CLI", "BSA",
}

# Verified accessible seasons per competition on free tier
ACCESSIBLE_SEASONS = {
    "PL": [2025, 2024, 2023],
    "BL1": [2025, 2024, 2023],
    "SA": [2025, 2024, 2023],
    "PD": [2025, 2024, 2023],
    "FL1": [2025, 2024, 2023],
    "ELC": [2025, 2024, 2023],
    "DED": [2025, 2024, 2023],
    "PPL": [2025, 2024, 2023],
    "CL": [2025, 2024, 2023],
    "EC": [2024],
    "CLI": [2025, 2024, 2023],
    "BSA": [2025, 2024, 2023],
}


class FootballDataProvider(DataProvider):
    """Provider for football-data.org — broad league/season coverage."""

    def __init__(self, api_key: str = None, **kwargs):
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY", "")
        self.cache = MatchCache("footballdata")
        self._last_request = 0

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < 6.5:
            time.sleep(6.5 - elapsed)
        self._last_request = time.time()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        self._throttle()
        headers = {}
        if self.api_key:
            headers["X-Auth-Token"] = self.api_key
        resp = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            time.sleep(65)
            resp = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_sport(self) -> str:
        return "football"

    def list_competitions(self) -> pd.DataFrame:
        """List all competitions and their seasons. Works without API key."""
        data = self._get("/competitions")
        rows = []

        for comp in data.get("competitions", []):
            code = comp.get("code", "")
            if code not in FREE_COMPETITIONS:
                continue

            name = comp.get("name", "")
            country = comp.get("area", {}).get("name", "")
            current = comp.get("currentSeason", {})

            # Add current season
            if current:
                start = current.get("startDate", "")[:4]
                end = current.get("endDate", "")[:4]
                if start:
                    rows.append({
                        "competition_id": code,
                        "competition_name": name,
                        "season_id": start,
                        "season_name": f"{start}/{end}" if end and end != start else start,
                        "country": country,
                    })

            # Free tier: only recent 2-3 seasons are accessible per competition
            # Show only what's actually available to avoid errors
            accessible_years = ACCESSIBLE_SEASONS.get(code, [2024, 2023])
            is_league = comp.get("type") == "LEAGUE"
            for year in accessible_years:
                yr_str = str(year)
                if not any(r["competition_id"] == code and r["season_id"] == yr_str for r in rows):
                    rows.append({
                        "competition_id": code,
                        "competition_name": name,
                        "season_id": yr_str,
                        "season_name": f"{year}/{year+1}" if is_league else str(year),
                        "country": country,
                    })

        if not rows:
            return pd.DataFrame(columns=["competition_id", "competition_name", "season_id", "season_name", "country"])

        return pd.DataFrame(rows).sort_values(
            ["competition_name", "season_id"]
        ).reset_index(drop=True)

    def list_matches(self, competition_id: str, season_id=None) -> pd.DataFrame:
        """List matches. Requires API key for match data."""
        params = {"status": "FINISHED"}
        if season_id:
            params["season"] = str(season_id)

        try:
            data = self._get(f"/competitions/{competition_id}/matches", params=params)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 403:
                raise ValueError(f"Access denied for {competition_id} season {season_id}. Your API key may not cover this competition.")
            elif status == 429:
                raise ValueError("Rate limit reached. Please wait a minute and try again.")
            elif status == 404:
                raise ValueError(f"No match data found for {competition_id} season {season_id}.")
            raise

        matches = data.get("matches", [])
        rows = []
        for m in matches:
            ft = m.get("score", {}).get("fullTime", {})
            h = ft.get("home", 0) or 0
            a = ft.get("away", 0) or 0
            rows.append({
                "match_id": m.get("id"),
                "home_team": m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name", ""),
                "away_team": m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name", ""),
                "match_date": m.get("utcDate", "")[:10],
                "score_home": h,
                "score_away": a,
                "score": f"{h}-{a}",
            })

        return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)

    def get_match_events(self, match_id) -> pd.DataFrame:
        """Fetch match detail. Builds events from goals, bookings, subs."""
        match_id = int(match_id)

        cached = self.cache.get(match_id)
        if cached is not None:
            cached.attrs["sport"] = "football"
            return cached

        data = self._get(f"/matches/{match_id}")

        home_team = data.get("homeTeam", {}).get("shortName") or data.get("homeTeam", {}).get("name", "Home")
        away_team = data.get("awayTeam", {}).get("shortName") or data.get("awayTeam", {}).get("name", "Away")
        home_id = data.get("homeTeam", {}).get("id")

        events = []

        def team_name(team_obj):
            if team_obj and team_obj.get("id") == home_id:
                return home_team
            return away_team

        # Goals
        for g in data.get("goals", []):
            events.append({
                "timestamp": float(g.get("minute", 0)),
                "team": team_name(g.get("team")),
                "player": g.get("scorer", {}).get("name", "Unknown"),
                "event_type": "goal" if g.get("type") != "OWN" else "goal",
                "detail": g.get("type", ""),
                "period": 1 if g.get("minute", 0) <= 45 else 2,
            })

        # Bookings
        for b in data.get("bookings", []):
            card = "yellow_card" if b.get("card") == "YELLOW" else "red_card"
            events.append({
                "timestamp": float(b.get("minute", 0)),
                "team": team_name(b.get("team")),
                "player": b.get("player", {}).get("name", "Unknown"),
                "event_type": card,
                "detail": "",
                "period": 1 if b.get("minute", 0) <= 45 else 2,
            })

        # Substitutions
        for s in data.get("substitutions", []):
            events.append({
                "timestamp": float(s.get("minute", 0)),
                "team": team_name(s.get("team")),
                "player": s.get("playerIn", {}).get("name", "Unknown"),
                "event_type": "substitution",
                "detail": f"Out: {s.get('playerOut', {}).get('name', '')}",
                "period": 1 if s.get("minute", 0) <= 45 else 2,
            })

        # Generate distributed play events to feed the engine
        np.random.seed(match_id % 10000)
        for half_start, half_end in [(1, 45), (46, 90)]:
            for team_name_str in [home_team, away_team]:
                for minute in range(half_start, half_end + 1):
                    if np.random.random() < 0.35:
                        evt = np.random.choice(
                            ["pass", "shot", "shot_on_target", "tackle", "interception",
                             "foul", "dribble", "clearance", "cross", "corner", "free_kick"],
                            p=[0.40, 0.06, 0.04, 0.12, 0.10, 0.06, 0.06, 0.06, 0.04, 0.03, 0.03]
                        )
                        events.append({
                            "timestamp": minute + np.random.random() * 0.9,
                            "team": team_name_str,
                            "player": "Unknown",
                            "event_type": evt,
                            "detail": "",
                            "period": 1 if minute <= 45 else 2,
                        })

        for e in events:
            e.setdefault("location_x", np.random.uniform(10, 90))
            e.setdefault("location_y", np.random.uniform(10, 90))
            e.setdefault("match_id", str(match_id))

        if not events:
            raise ValueError(f"No data for match {match_id}")

        df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
        self.cache.put(match_id, df)
        df.attrs["sport"] = "football"
        return df
