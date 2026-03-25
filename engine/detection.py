"""
Logic-Based Event Detection Engine
Rule-based detection of high-impact events: momentum shifts, critical mistakes,
high-pressure sequences, and game-changing moments.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from engine.processing import MatchContext


@dataclass
class DetectedEvent:
    """A high-impact event detected by the rules engine."""
    event_type: str          # momentum_shift, critical_mistake, pressure_sequence, game_changer
    timestamp: float         # Match minute
    team: str                # Affected team
    severity: str            # low, medium, high, critical
    description: str         # Human-readable description
    evidence: dict           # Supporting data
    player: Optional[str] = None


class DetectionEngine:
    """
    Rule-based engine that scans match events for high-impact patterns.
    Each rule is a method that yields DetectedEvent instances.
    """

    def __init__(self, ctx: MatchContext):
        self.ctx = ctx
        self.events = ctx.events
        self.sport = ctx.sport

    def run_all(self) -> list[DetectedEvent]:
        """Execute all detection rules and return sorted list of detected events."""
        detected = []
        detected.extend(self._detect_momentum_shifts())
        detected.extend(self._detect_critical_mistakes())
        detected.extend(self._detect_pressure_sequences())
        detected.extend(self._detect_game_changers())
        detected.extend(self._detect_scoring_runs())
        detected.extend(self._detect_defensive_collapses())
        detected.extend(self._detect_late_game_surges())

        # Advanced metric detections (skip if data not available)
        detected.extend(self._detect_xg_overperformance())
        detected.extend(self._detect_pressing_phases())
        detected.extend(self._detect_progressive_play())

        detected.sort(key=lambda e: e.timestamp)
        return detected

    def _detect_momentum_shifts(self) -> list[DetectedEvent]:
        """
        Detect momentum shifts: when a team concedes multiple high-impact events
        within a short window, then the opposing team takes over.

        Logic: In any 3-minute window, if team A has 0-1 pressure events while
        team B has 3+, and team A had more pressure events in the previous window,
        that's a momentum shift.
        """
        detected = []
        df = self.events
        window = 3.0  # minutes

        for team in self.ctx.teams:
            other_teams = [t for t in self.ctx.teams if t != team]
            if not other_teams:
                continue
            opponent = other_teams[0]

            timestamps = sorted(df["timestamp"].unique())
            for i, t in enumerate(timestamps):
                if t < window:
                    continue

                current = df[(df["timestamp"] >= t - window) & (df["timestamp"] <= t)]
                previous = df[(df["timestamp"] >= t - 2 * window) & (df["timestamp"] < t - window)]

                team_pressure_now = current[(current["team"] == team) & (current["is_pressure"])].shape[0]
                opp_pressure_now = current[(current["team"] == opponent) & (current["is_pressure"])].shape[0]
                team_pressure_prev = previous[(previous["team"] == team) & (previous["is_pressure"])].shape[0]
                opp_pressure_prev = previous[(previous["team"] == opponent) & (previous["is_pressure"])].shape[0]

                # Team had dominance, now opponent dominates
                if (team_pressure_prev >= 3 and team_pressure_now <= 1 and
                        opp_pressure_now >= 3 and opp_pressure_prev <= 1):

                    # Avoid duplicate detections within 2 minutes
                    if any(e.event_type == "momentum_shift" and abs(e.timestamp - t) < 2
                           and e.team == opponent for e in detected):
                        continue

                    detected.append(DetectedEvent(
                        event_type="momentum_shift",
                        timestamp=round(t, 1),
                        team=opponent,
                        severity="high",
                        description=(
                            f"Momentum shifted to {opponent} around minute {t:.0f}. "
                            f"{team}'s pressure dropped from {team_pressure_prev} to {team_pressure_now} events, "
                            f"while {opponent} surged from {opp_pressure_prev} to {opp_pressure_now}."
                        ),
                        evidence={
                            "window_minutes": window,
                            "team_pressure_before": team_pressure_prev,
                            "team_pressure_after": team_pressure_now,
                            "opponent_pressure_before": opp_pressure_prev,
                            "opponent_pressure_after": opp_pressure_now,
                        },
                    ))

        return detected

    def _detect_critical_mistakes(self) -> list[DetectedEvent]:
        """
        Detect critical mistakes: possession loss in attacking/midfield zone
        followed by opponent scoring opportunity within 1 minute.
        """
        detected = []
        df = self.events
        turnovers = df[df["is_turnover"]].copy()

        for _, turnover in turnovers.iterrows():
            team = turnover["team"]
            t = turnover["timestamp"]
            zone = turnover.get("zone", None)

            # Only flag turnovers in midfield or attacking zones
            if zone not in ["midfield", "attacking"] and not pd.isna(turnover.get("location_x", np.nan)):
                if turnover.get("location_x", 0) < 33:
                    continue

            # Look for opponent scoring/pressure event within 1 minute
            opponent_events = df[
                (df["team"] != team) &
                (df["timestamp"] > t) &
                (df["timestamp"] <= t + 1.0) &
                (df["is_scoring"] | df["is_pressure"])
            ]

            if not opponent_events.empty:
                scoring = opponent_events[opponent_events["is_scoring"]]
                severity = "critical" if not scoring.empty else "high"
                result = "a goal" if not scoring.empty else "a scoring opportunity"

                detected.append(DetectedEvent(
                    event_type="critical_mistake",
                    timestamp=round(t, 1),
                    team=team,
                    player=turnover["player"],
                    severity=severity,
                    description=(
                        f"{turnover['player']} ({team}) lost possession at minute {t:.0f} "
                        f"leading to {result} for the opponent within 60 seconds."
                    ),
                    evidence={
                        "turnover_event": turnover["event_type"],
                        "zone": str(zone),
                        "opponent_events_after": len(opponent_events),
                        "led_to_goal": not scoring.empty,
                    },
                ))

        return detected

    def _detect_pressure_sequences(self) -> list[DetectedEvent]:
        """
        Detect sustained pressure: 4+ pressure events by one team in 2 minutes
        with no pressure events by the opponent.
        """
        detected = []
        df = self.events
        window = 2.0

        for team in self.ctx.teams:
            other_teams = [t for t in self.ctx.teams if t != team]
            opponent = other_teams[0] if other_teams else None

            team_pressure = df[(df["team"] == team) & (df["is_pressure"])].copy()
            if len(team_pressure) < 4:
                continue

            timestamps = team_pressure["timestamp"].values
            for i in range(len(timestamps)):
                t_start = timestamps[i]
                t_end = t_start + window
                cluster = team_pressure[
                    (team_pressure["timestamp"] >= t_start) &
                    (team_pressure["timestamp"] <= t_end)
                ]

                if len(cluster) < 4:
                    continue

                # Check opponent had no pressure in this window
                if opponent:
                    opp_pressure = df[
                        (df["team"] == opponent) &
                        (df["is_pressure"]) &
                        (df["timestamp"] >= t_start) &
                        (df["timestamp"] <= t_end)
                    ]
                    if len(opp_pressure) > 0:
                        continue

                # Deduplicate
                if any(e.event_type == "pressure_sequence" and abs(e.timestamp - t_start) < 2
                       and e.team == team for e in detected):
                    continue

                event_types = cluster["event_type"].value_counts().to_dict()
                detected.append(DetectedEvent(
                    event_type="pressure_sequence",
                    timestamp=round(t_start, 1),
                    team=team,
                    severity="medium",
                    description=(
                        f"{team} mounted sustained pressure from minute {t_start:.0f} to {t_end:.0f} "
                        f"with {len(cluster)} pressure events and no opponent response."
                    ),
                    evidence={
                        "event_count": len(cluster),
                        "window_minutes": window,
                        "event_breakdown": event_types,
                    },
                ))

        return detected

    def _detect_game_changers(self) -> list[DetectedEvent]:
        """
        Detect game-changing moments: events that fundamentally alter the match state.
        - Equalizing or go-ahead goals
        - Red cards
        - Goals/scores immediately after a momentum shift
        """
        detected = []
        df = self.events
        scoring = self.ctx.scoring_timeline

        if scoring is None or scoring.empty:
            return detected

        # Track running score per team
        score = {team: 0 for team in self.ctx.teams}

        for _, row in scoring.iterrows():
            team = row["team"]
            prev_score = dict(score)
            score[team] = row["running_score"]

            other_teams = [t for t in self.ctx.teams if t != team]
            if not other_teams:
                continue
            opponent = other_teams[0]
            opp_score = score.get(opponent, 0)

            is_equalizer = prev_score.get(team, 0) < opp_score and row["running_score"] == opp_score
            is_go_ahead = prev_score.get(team, 0) <= opp_score and row["running_score"] > opp_score

            if is_equalizer or is_go_ahead:
                label = "equalizer" if is_equalizer else "go-ahead"
                detected.append(DetectedEvent(
                    event_type="game_changer",
                    timestamp=round(row["timestamp"], 1),
                    team=team,
                    player=row.get("player"),
                    severity="critical",
                    description=(
                        f"{label.title()} by {row.get('player', 'Unknown')} ({team}) "
                        f"at minute {row['timestamp']:.0f}! "
                        f"Score now {row['running_score']:.0f}-{opp_score:.0f}."
                    ),
                    evidence={
                        "type": label,
                        "score_before": prev_score,
                        "score_after": dict(score),
                    },
                ))

        # Red cards
        red_cards = df[df["event_type"] == "red_card"]
        for _, card in red_cards.iterrows():
            detected.append(DetectedEvent(
                event_type="game_changer",
                timestamp=round(card["timestamp"], 1),
                team=card["team"],
                player=card["player"],
                severity="critical",
                description=(
                    f"Red card for {card['player']} ({card['team']}) at minute {card['timestamp']:.0f}. "
                    f"Team reduced to fewer players for the remainder."
                ),
                evidence={"event": "red_card"},
            ))

        return detected

    def _detect_scoring_runs(self) -> list[DetectedEvent]:
        """
        Detect scoring runs: one team scores 3+ times without opponent scoring.
        Especially relevant for basketball.
        """
        detected = []
        scoring = self.ctx.scoring_timeline
        if scoring is None or scoring.empty:
            return detected

        consecutive = 0
        current_team = None
        run_start = 0

        for _, row in scoring.iterrows():
            if row["team"] == current_team:
                consecutive += 1
            else:
                if consecutive >= 3 and current_team:
                    detected.append(DetectedEvent(
                        event_type="scoring_run",
                        timestamp=round(run_start, 1),
                        team=current_team,
                        severity="high" if consecutive >= 5 else "medium",
                        description=(
                            f"{current_team} went on a {consecutive}-0 scoring run "
                            f"starting at minute {run_start:.0f}."
                        ),
                        evidence={"consecutive_scores": consecutive},
                    ))
                current_team = row["team"]
                consecutive = 1
                run_start = row["timestamp"]

        # Check final run
        if consecutive >= 3 and current_team:
            detected.append(DetectedEvent(
                event_type="scoring_run",
                timestamp=round(run_start, 1),
                team=current_team,
                severity="high" if consecutive >= 5 else "medium",
                description=(
                    f"{current_team} went on a {consecutive}-0 scoring run "
                    f"starting at minute {run_start:.0f}."
                ),
                evidence={"consecutive_scores": consecutive},
            ))

        return detected

    def _detect_defensive_collapses(self) -> list[DetectedEvent]:
        """
        Detect defensive collapse: team concedes 2+ goals/scores within 5 minutes.
        """
        detected = []
        scoring = self.ctx.scoring_timeline
        if scoring is None or scoring.empty:
            return detected

        for team in self.ctx.teams:
            opponent_scoring = scoring[scoring["team"] != team].copy()
            if len(opponent_scoring) < 2:
                continue

            timestamps = opponent_scoring["timestamp"].values
            for i in range(len(timestamps) - 1):
                window_end = timestamps[i] + 5.0
                in_window = opponent_scoring[
                    (opponent_scoring["timestamp"] >= timestamps[i]) &
                    (opponent_scoring["timestamp"] <= window_end)
                ]
                if len(in_window) >= 2:
                    if any(e.event_type == "defensive_collapse" and abs(e.timestamp - timestamps[i]) < 3
                           and e.team == team for e in detected):
                        continue

                    detected.append(DetectedEvent(
                        event_type="defensive_collapse",
                        timestamp=round(timestamps[i], 1),
                        team=team,
                        severity="high",
                        description=(
                            f"{team} conceded {len(in_window)} scores between minutes "
                            f"{timestamps[i]:.0f}-{window_end:.0f}, indicating a defensive collapse."
                        ),
                        evidence={
                            "goals_conceded": len(in_window),
                            "window_minutes": 5,
                        },
                    ))

        return detected

    def _detect_late_game_surges(self) -> list[DetectedEvent]:
        """
        Detect late-game surges: significant increase in pressure/scoring
        in the final 15% of match time.
        """
        detected = []
        df = self.events
        duration = self.ctx.duration_minutes
        if duration < 10:
            return detected

        late_cutoff = duration * 0.85 + df["timestamp"].min()

        for team in self.ctx.teams:
            early = df[(df["team"] == team) & (df["timestamp"] < late_cutoff)]
            late = df[(df["team"] == team) & (df["timestamp"] >= late_cutoff)]

            if len(early) == 0 or len(late) == 0:
                continue

            early_minutes = late_cutoff - df["timestamp"].min()
            late_minutes = duration * 0.15

            early_rate = early["is_pressure"].sum() / max(early_minutes, 1)
            late_rate = late["is_pressure"].sum() / max(late_minutes, 1)

            if late_rate > early_rate * 1.5 and late["is_pressure"].sum() >= 3:
                detected.append(DetectedEvent(
                    event_type="late_game_surge",
                    timestamp=round(late_cutoff, 1),
                    team=team,
                    severity="medium",
                    description=(
                        f"{team} significantly increased pressure in the final stretch "
                        f"({late['is_pressure'].sum()} pressure events vs "
                        f"{early_rate:.1f}/min earlier). "
                        f"Late-game urgency or tactical shift."
                    ),
                    evidence={
                        "early_pressure_rate": round(early_rate, 2),
                        "late_pressure_rate": round(late_rate, 2),
                        "late_pressure_events": int(late["is_pressure"].sum()),
                    },
                ))

        return detected

    def _detect_xg_overperformance(self) -> list[DetectedEvent]:
        """
        Detect when a team's actual goals significantly differ from xG.
        Clinical finishing (goals >> xG) or wasteful play (xG >> goals).
        """
        detected = []
        xg = self.ctx.xg_analysis
        if not xg or "teams" not in xg:
            return detected

        for team, data in xg["teams"].items():
            diff = data.get("difference", 0)
            goals = data.get("goals", 0)
            team_xg = data.get("xg", 0)

            if abs(diff) > 1.0 and goals >= 1:
                if diff > 0:
                    severity = "high" if diff > 2 else "medium"
                    desc = (
                        f"{team} scored {goals} goals from just {team_xg:.1f} xG "
                        f"(+{diff:.1f} overperformance). "
                        f"Clinical finishing or fortunate outcomes."
                    )
                else:
                    severity = "high" if diff < -2 else "medium"
                    desc = (
                        f"{team} scored only {goals} goals despite {team_xg:.1f} xG "
                        f"({diff:.1f} underperformance). "
                        f"Wasteful finishing or outstanding opponent goalkeeper."
                    )

                detected.append(DetectedEvent(
                    event_type="xg_overperformance",
                    timestamp=0,
                    team=team,
                    severity=severity,
                    description=desc,
                    evidence={"xg": team_xg, "goals": goals, "difference": diff},
                ))

        return detected

    def _detect_pressing_phases(self) -> list[DetectedEvent]:
        """
        Detect periods of intense pressing using PPDA windows.
        PPDA < 6 indicates very high pressing (gegenpressing).
        """
        detected = []
        ppda = self.ctx.ppda
        if not ppda or "windows" not in ppda:
            return detected

        for team in self.ctx.teams:
            key = f"{team}_ppda"
            for window in ppda["windows"]:
                val = window.get(key)
                if val is not None and val < 6 and val > 0:
                    detected.append(DetectedEvent(
                        event_type="pressing_phase",
                        timestamp=window["start"],
                        team=team,
                        severity="medium",
                        description=(
                            f"{team} applied intense pressing between minutes "
                            f"{window['start']:.0f}-{window['end']:.0f} "
                            f"(PPDA: {val:.1f}). Opponent heavily disrupted."
                        ),
                        evidence={"ppda": val, "window": [window["start"], window["end"]]},
                    ))

        return detected

    def _detect_progressive_play(self) -> list[DetectedEvent]:
        """
        Detect teams/players with exceptionally high progressive passing.
        """
        detected = []
        prog = self.ctx.progressive_passes
        if prog is None or prog.empty:
            return detected

        for team in self.ctx.teams:
            team_prog = prog[prog["team"] == team]
            if team_prog.empty:
                continue

            total = team_prog["progressive_passes"].sum()
            top_player = team_prog.iloc[0]

            if total >= 20:
                detected.append(DetectedEvent(
                    event_type="progressive_play",
                    timestamp=0,
                    team=team,
                    player=top_player["player"],
                    severity="medium",
                    description=(
                        f"{team} made {total} progressive passes. "
                        f"Top contributor: {top_player['player']} "
                        f"({int(top_player['progressive_passes'])} progressive passes, "
                        f"avg {top_player['avg_distance']:.1f} units gained). "
                        f"Vertical, purposeful build-up play."
                    ),
                    evidence={
                        "total_progressive": int(total),
                        "top_player": top_player["player"],
                        "top_count": int(top_player["progressive_passes"]),
                    },
                ))

        return detected
