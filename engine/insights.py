"""
Insight Generation Module
Combines detection results and statistical analysis into structured,
human-readable, actionable insights.
"""

import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from engine.processing import MatchContext
from engine.detection import DetectedEvent
from engine.statistics import StatisticalInsight


@dataclass
class MatchInsight:
    """A single actionable insight with context and recommendation."""
    id: str
    category: str           # tactical, player, momentum, statistical
    title: str
    description: str
    impact: str             # low, medium, high, critical
    recommendation: str     # Actionable advice
    timestamp: Optional[float] = None
    team: Optional[str] = None
    player: Optional[str] = None
    evidence: Optional[dict] = None


@dataclass
class MatchReport:
    """Complete match analysis report."""
    match_id: str
    sport: str
    teams: list[str]
    duration_minutes: float
    generated_at: str
    summary: str
    key_moments: list[dict]
    insights: list[dict]
    team_comparison: dict
    player_highlights: list[dict]
    statistical_findings: list[dict]
    recommendations: list[str]


class InsightGenerator:
    """Generate structured insights from detection and statistical analysis."""

    def __init__(self, ctx: MatchContext, detected_events: list[DetectedEvent],
                 stat_insights: list[StatisticalInsight]):
        self.ctx = ctx
        self.detected = detected_events
        self.stats = stat_insights

    def generate_report(self) -> MatchReport:
        """Generate a complete match analysis report."""
        insights = self._generate_insights()
        key_moments = self._format_key_moments()
        team_comparison = self._build_team_comparison()
        player_highlights = self._build_player_highlights()
        stat_findings = self._format_statistical_findings()
        recommendations = self._generate_recommendations(insights)
        summary = self._generate_summary(insights, key_moments)

        return MatchReport(
            match_id=self.ctx.match_id,
            sport=self.ctx.sport,
            teams=self.ctx.teams,
            duration_minutes=round(self.ctx.duration_minutes, 1),
            generated_at=datetime.now().isoformat(),
            summary=summary,
            key_moments=key_moments,
            insights=[asdict(i) for i in insights],
            team_comparison=team_comparison,
            player_highlights=player_highlights,
            statistical_findings=stat_findings,
            recommendations=recommendations,
        )

    def _generate_insights(self) -> list[MatchInsight]:
        """Convert detected events and stats into actionable insights."""
        insights = []
        idx = 0

        # From detected events
        for event in self.detected:
            idx += 1
            insight = self._event_to_insight(event, idx)
            if insight:
                insights.append(insight)

        # From statistical analysis
        for stat in self.stats:
            idx += 1
            insight = self._stat_to_insight(stat, idx)
            if insight:
                insights.append(insight)

        # Cross-cutting insights
        insights.extend(self._generate_cross_insights(idx))

        return insights

    def _event_to_insight(self, event: DetectedEvent, idx: int) -> Optional[MatchInsight]:
        """Convert a detected event into an insight with recommendation."""
        recommendations = {
            "momentum_shift": (
                "Watch the replay of this phase. What triggered the shift? "
                "Was it a substitution, a tactical change, or a loss of concentration? "
                "Understanding the trigger helps prevent it in future matches."
            ),
            "critical_mistake": (
                "This wasn't just bad luck. Break down the sequence: "
                "Where was the player positioned? What were the passing options? "
                "Drill decision-making under pressure to reduce these errors."
            ),
            "pressure_sequence": (
                "This dominant phase shows what works. Replicate the formation, "
                "pressing triggers, and tempo that created this sustained pressure. "
                "It's a blueprint for controlling the game."
            ),
            "game_changer": (
                "This single moment reshaped the entire match. "
                "How did each team react afterward? Build mental resilience drills "
                "and rehearse game-state management for similar scenarios."
            ),
            "scoring_run": (
                "When one team scores repeatedly without reply, the opponent is broken. "
                "Identify what caused the collapse: fatigue, tactical mismatch, or panic? "
                "Practice composure drills to prevent being on the wrong side."
            ),
            "defensive_collapse": (
                "Conceding multiple goals in quick succession is a structural failure. "
                "Review the defensive shape, communication breakdowns, and whether "
                "the backline was too high or too passive. Fix the system, not just the individuals."
            ),
            "late_game_surge": (
                "The late push could be desperation or superior fitness. "
                "If it works, build it into the game plan. If it doesn't, "
                "consider whether the energy would be better spent earlier."
            ),
            "xg_overperformance": (
                "xG tells us how many goals the chances were worth. "
                "Overperforming means clinical finishing or lucky bounces. "
                "Underperforming means the quality is there but execution isn't."
            ),
            "pressing_phase": (
                "High pressing is exhausting but effective. "
                "Can the team maintain this intensity, or should it be "
                "deployed in short bursts at key moments? Monitor fitness data."
            ),
            "progressive_play": (
                "Progressive passes move the ball toward the goal with purpose. "
                "The players driving this are the creative engine. "
                "Build the attacking system around their strengths."
            ),
        }

        category_map = {
            "momentum_shift": "momentum",
            "critical_mistake": "tactical",
            "pressure_sequence": "tactical",
            "game_changer": "momentum",
            "scoring_run": "momentum",
            "defensive_collapse": "tactical",
            "late_game_surge": "tactical",
            "xg_overperformance": "statistical",
            "pressing_phase": "tactical",
            "progressive_play": "tactical",
        }

        # Cleaner titles
        title_map = {
            "momentum_shift": "Momentum Shift",
            "critical_mistake": "Costly Error",
            "pressure_sequence": "Dominant Pressure Phase",
            "game_changer": "Game-Changing Moment",
            "scoring_run": "Scoring Run",
            "defensive_collapse": "Defensive Breakdown",
            "late_game_surge": "Late Surge",
            "xg_overperformance": "Finishing vs Expected (xG)",
            "pressing_phase": "High Press Activated",
            "progressive_play": "Progressive Build-Up",
        }

        return MatchInsight(
            id=f"INS-{idx:03d}",
            category=category_map.get(event.event_type, "tactical"),
            title=title_map.get(event.event_type, event.event_type.replace("_", " ").title()),
            description=event.description,
            impact=event.severity,
            recommendation=recommendations.get(event.event_type, "Review the match footage for this phase."),
            timestamp=event.timestamp,
            team=event.team,
            player=event.player,
            evidence=event.evidence,
        )

    def _stat_to_insight(self, stat: StatisticalInsight, idx: int) -> Optional[MatchInsight]:
        """Convert a statistical insight into an actionable insight."""
        rec_map = {
            "performance_comparison": (
                "This gap between the two sides tells a clear story. "
                "The underperforming team should study what the other side did differently "
                "and adapt their approach in training."
            ),
            "trend": (
                "Performance changing over the course of the match is a signal. "
                "Improving? The game plan is working. Declining? "
                "Look at fitness, concentration, or whether substitutions came too late."
            ),
            "anomaly": (
                "This player stands out statistically from their teammates. "
                "If it's a strength, build the game plan around them. "
                "If it's a weakness, provide support or consider tactical protection."
            ),
            "correlation": (
                "This tells you which actions actually lead to goals. "
                "Train the activities that correlate with scoring, "
                "and don't waste energy on actions that don't convert."
            ),
            "efficiency": (
                "Efficiency separates good teams from great ones. "
                "If conversion is low, focus on final-third composure. "
                "If it's high, maintain that clinical edge."
            ),
        }

        impact_map = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}
        impact = "low"
        for level, threshold in sorted(impact_map.items(), key=lambda x: x[1]):
            if stat.significance >= threshold:
                impact = level

        return MatchInsight(
            id=f"INS-{idx:03d}",
            category="statistical",
            title=f"{stat.category.replace('_', ' ').title()}: {stat.metric}",
            description=stat.description,
            impact=impact,
            recommendation=rec_map.get(stat.category, "Review the underlying data."),
            evidence=stat.data,
        )

    def _generate_cross_insights(self, start_idx: int) -> list[MatchInsight]:
        """Generate insights that span multiple detected patterns."""
        insights = []
        idx = start_idx

        # Check if one team dominates across multiple categories
        team_events = {}
        for event in self.detected:
            team_events.setdefault(event.team, []).append(event)

        for team, events in team_events.items():
            event_types = [e.event_type for e in events]
            has_momentum = "momentum_shift" in event_types
            has_pressure = "pressure_sequence" in event_types
            has_scoring_run = "scoring_run" in event_types

            if has_momentum and has_pressure and has_scoring_run:
                idx += 1
                insights.append(MatchInsight(
                    id=f"INS-{idx:03d}",
                    category="tactical",
                    title=f"{team} Complete Dominance Phase",
                    description=(
                        f"{team} showed a complete dominance pattern: sustained pressure, "
                        f"momentum capture, and a scoring run. This indicates a period where "
                        f"the opponent was completely overwhelmed tactically and physically."
                    ),
                    impact="critical",
                    recommendation=(
                        "This dominance pattern is gold for match preparation. "
                        "Study the exact conditions, formation, and personnel that created it. "
                        "For the opponent: identify the breaking point and build resilience drills."
                    ),
                    team=team,
                ))

        # Check for teams with critical mistakes leading to goals
        mistakes_to_goals = [e for e in self.detected
                             if e.event_type == "critical_mistake"
                             and e.evidence.get("led_to_goal")]
        if len(mistakes_to_goals) >= 2:
            teams_affected = set(e.team for e in mistakes_to_goals)
            for team in teams_affected:
                team_mistakes = [e for e in mistakes_to_goals if e.team == team]
                if len(team_mistakes) >= 2:
                    idx += 1
                    players = list(set(e.player for e in team_mistakes if e.player))
                    insights.append(MatchInsight(
                        id=f"INS-{idx:03d}",
                        category="player",
                        title=f"{team} Recurring Error Pattern",
                        description=(
                            f"{team} made {len(team_mistakes)} critical errors that directly "
                            f"led to opponent goals. Players involved: {', '.join(players)}. "
                            f"This is not bad luck --it's a systemic issue."
                        ),
                        impact="critical",
                        recommendation=(
                            "This recurring pattern demands immediate tactical review. "
                            "Focus on decision-making under pressure and risk management "
                            "in the build-up phase."
                        ),
                        team=team,
                    ))

        return insights

    def _format_key_moments(self) -> list[dict]:
        """Format detected events as key moments timeline."""
        moments = []
        for event in sorted(self.detected, key=lambda e: e.timestamp):
            moments.append({
                "minute": round(event.timestamp, 1),
                "type": event.event_type,
                "team": event.team,
                "player": event.player,
                "severity": event.severity,
                "description": event.description,
            })
        return moments

    def _build_team_comparison(self) -> dict:
        """Build side-by-side team comparison."""
        ts = self.ctx.team_stats
        if ts is None or ts.empty:
            return {}

        comparison = {}
        for _, row in ts.iterrows():
            team = row["team"]
            comparison[team] = {
                "total_events": int(row.get("total_events", 0)),
                "scoring_events": int(row.get("scoring_events", 0)),
                "turnovers": int(row.get("turnovers", 0)),
                "pressure_events": int(row.get("pressure_events", 0)),
                "mistakes": int(row.get("mistakes", 0)),
                "unique_players": int(row.get("unique_players", 0)),
            }

        # Add advanced metrics if available
        xg = self.ctx.xg_analysis
        if xg and "teams" in xg:
            for team, data in xg["teams"].items():
                if team in comparison:
                    comparison[team]["xg"] = data.get("xg", 0)
                    comparison[team]["xg_difference"] = data.get("difference", 0)
                    comparison[team]["shots_on_target"] = data.get("shots_on_target", 0)

        ppda = self.ctx.ppda
        if ppda and "teams" in ppda:
            for team, data in ppda["teams"].items():
                if team in comparison:
                    comparison[team]["ppda"] = data.get("ppda", 0)
                    comparison[team]["pressing_intensity"] = data.get("pressing_intensity", "")

        pc = self.ctx.pass_completion
        if pc and "teams" in pc:
            for team, data in pc["teams"].items():
                if team in comparison:
                    comparison[team]["pass_completion"] = f"{data.get('rate', 0):.0%}"

        return comparison

    def _build_player_highlights(self) -> list[dict]:
        """Identify standout and underperforming players."""
        ps = self.ctx.player_stats
        if ps is None or ps.empty:
            return []

        highlights = []

        # Top scorers
        if "scoring_events" in ps.columns:
            top_scorers = ps.nlargest(3, "scoring_events")
            for _, p in top_scorers.iterrows():
                if p["scoring_events"] > 0:
                    highlights.append({
                        "player": p["player"],
                        "team": p["team"],
                        "highlight": "top_scorer",
                        "detail": f"{int(p['scoring_events'])} scoring events",
                    })

        # Highest mistake rate
        if "mistake_rate" in ps.columns:
            error_prone = ps[ps["total_events"] >= 5].nlargest(2, "mistake_rate")
            for _, p in error_prone.iterrows():
                if p["mistake_rate"] > 0.15:
                    highlights.append({
                        "player": p["player"],
                        "team": p["team"],
                        "highlight": "error_prone",
                        "detail": f"{p['mistake_rate']:.0%} mistake rate ({int(p['mistakes'])} mistakes in {int(p['total_events'])} events)",
                    })

        # Most active
        most_active = ps.nlargest(2, "total_events")
        for _, p in most_active.iterrows():
            highlights.append({
                "player": p["player"],
                "team": p["team"],
                "highlight": "most_active",
                "detail": f"{int(p['total_events'])} total events over {p['active_minutes']:.0f} minutes",
            })

        return highlights

    def _format_statistical_findings(self) -> list[dict]:
        """Format statistical insights for the report."""
        return [
            {
                "category": s.category,
                "metric": s.metric,
                "description": s.description,
                "significance": s.significance,
                "data": s.data,
            }
            for s in sorted(self.stats, key=lambda s: s.significance, reverse=True)
        ]

    def _generate_recommendations(self, insights: list[MatchInsight]) -> list[str]:
        """Generate top-level coaching takeaways from all insights."""
        recommendations = []

        # Team-specific narratives
        for team in self.ctx.teams:
            team_insights = [i for i in insights if i.team == team]
            mistakes = [i for i in team_insights if any(w in (i.title or "").lower()
                        for w in ["error", "mistake", "collapse", "breakdown", "costly"])]
            strengths = [i for i in team_insights if any(w in (i.title or "").lower()
                         for w in ["dominance", "surge", "run", "progressive", "press", "dominant"])]

            if mistakes and strengths:
                recommendations.append(
                    f"{team} showed both quality and vulnerability. "
                    f"The {len(strengths)} positive phases prove the system works, "
                    f"but {len(mistakes)} defensive errors need fixing to be consistent."
                )
            elif mistakes:
                recommendations.append(
                    f"{team} must address {len(mistakes)} error patterns. "
                    f"Focus on defensive structure and composure in possession under pressure."
                )
            elif strengths:
                recommendations.append(
                    f"{team} created {len(strengths)} dominant phases. "
                    f"Study what triggered these periods and make them repeatable."
                )

        # xG recommendation
        xg = self.ctx.xg_analysis
        if xg and "teams" in xg:
            for team, data in xg["teams"].items():
                diff = data.get("difference", 0)
                if diff < -1.5:
                    recommendations.append(
                        f"{team} created {data.get('xg', 0):.1f} xG worth of chances but only scored "
                        f"{data.get('goals', 0)}. Finishing drills and composure in the box should be the priority."
                    )

        # Pass completion difference
        pc = self.ctx.pass_completion
        if pc and "teams" in pc:
            rates = {t: d.get("rate", 0) for t, d in pc["teams"].items()}
            if len(rates) == 2:
                teams_sorted = sorted(rates.items(), key=lambda x: x[1], reverse=True)
                gap = teams_sorted[0][1] - teams_sorted[1][1]
                if gap > 0.08:
                    recommendations.append(
                        f"{teams_sorted[0][0]} controlled the ball significantly better "
                        f"({teams_sorted[0][1]:.0%} vs {teams_sorted[1][1]:.0%} pass accuracy). "
                        f"{teams_sorted[1][0]} should work on retaining possession under pressure."
                    )

        if not recommendations:
            recommendations.append(
                "An evenly matched contest with no standout tactical concerns. "
                "Focus on marginal gains from the statistical analysis."
            )

        return recommendations

    def _generate_summary(self, insights: list[MatchInsight], key_moments: list[dict]) -> str:
        """Generate a narrative executive summary of the match."""
        sport = self.ctx.sport
        teams = self.ctx.teams
        duration = self.ctx.duration_minutes

        # Final score
        scoring = self.ctx.scoring_timeline
        final_score = {}
        if scoring is not None and not scoring.empty:
            for team in teams:
                team_scoring = scoring[scoring["team"] == team]
                final_score[team] = int(team_scoring["running_score"].max()) if not team_scoring.empty else 0

        critical_count = len([i for i in insights if i.impact == "critical"])
        high_count = len([i for i in insights if i.impact == "high"])

        parts = []

        # Opening with score context
        if len(teams) == 2 and final_score:
            s1, s2 = final_score.get(teams[0], 0), final_score.get(teams[1], 0)

            if s1 > s2:
                winner, loser, ws, ls = teams[0], teams[1], s1, s2
            elif s2 > s1:
                winner, loser, ws, ls = teams[1], teams[0], s2, s1
            else:
                winner, loser, ws, ls = None, None, s1, s2

            if winner:
                margin = ws - ls
                if margin >= 3:
                    parts.append(f"A commanding {ws}-{ls} victory for {winner} over {loser}.")
                elif margin == 1:
                    parts.append(f"A tight {ws}-{ls} win for {winner} against {loser} in a closely contested match.")
                else:
                    parts.append(f"{winner} beat {loser} {ws}-{ls} in an eventful {sport} match.")
            else:
                parts.append(f"A {s1}-{s2} draw between {teams[0]} and {teams[1]}.")

            # xG context
            xg = self.ctx.xg_analysis
            if xg and "teams" in xg:
                xg_parts = []
                for team in teams:
                    data = xg["teams"].get(team, {})
                    if data:
                        diff = data.get("difference", 0)
                        if abs(diff) > 1:
                            direction = "outperformed" if diff > 0 else "underperformed"
                            xg_parts.append(f"{team} {direction} their xG by {abs(diff):.1f}")
                if xg_parts:
                    parts.append(", and ".join(xg_parts) + ".")

            # Narrative about the match flow
            dominance = [e for e in self.detected if e.event_type in
                         ("momentum_shift", "pressure_sequence", "scoring_run", "defensive_collapse")]
            game_changers = [e for e in self.detected if e.event_type == "game_changer"]

            if winner and dominance:
                winner_dom = [e for e in dominance if e.team == winner]
                loser_dom = [e for e in dominance if e.team == loser]
                if len(winner_dom) > len(loser_dom) * 2:
                    parts.append(f"{winner} dominated the key phases of the match.")
                elif len(loser_dom) > 0:
                    parts.append("Both sides had periods of control, making for an open contest.")

            if game_changers:
                parts.append(f"The analysis identified {len(game_changers)} game-changing moments that shifted the outcome.")
        else:
            parts.append(f"Analysis of {' vs '.join(teams)} ({sport.title()}, {duration:.0f} minutes).")

        # Findings summary
        if critical_count > 0:
            parts.append(f"{critical_count} critical findings demand attention.")
        elif high_count > 0:
            parts.append(f"{high_count} significant patterns were detected.")

        return " ".join(parts)


def report_to_json(report: MatchReport) -> str:
    """Serialize a MatchReport to JSON."""
    return json.dumps(asdict(report), indent=2, default=str)


def report_to_text(report: MatchReport) -> str:
    """Generate a human-readable text summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("SPORTS INTELLIGENCE ENGINE --MATCH REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Match: {report.match_id}")
    lines.append(f"Sport: {report.sport.title()}")
    lines.append(f"Teams: {' vs '.join(report.teams)}")
    lines.append(f"Duration: {report.duration_minutes} minutes")
    lines.append(f"Generated: {report.generated_at}")
    lines.append("")
    lines.append("--EXECUTIVE SUMMARY —")
    lines.append(report.summary)
    lines.append("")

    if report.key_moments:
        lines.append("--KEY MOMENTS —")
        for m in report.key_moments:
            icon = {"critical": "[!!]", "high": "[!]", "medium": "[~]", "low": "[.]"}.get(m["severity"], "[ ]")
            lines.append(f"  {icon} [{m['minute']:.0f}'] {m['description']}")
        lines.append("")

    if report.team_comparison:
        lines.append("--TEAM COMPARISON —")
        headers = list(next(iter(report.team_comparison.values())).keys())
        lines.append(f"  {'Metric':<20} " + " ".join(f"{t:>12}" for t in report.teams))
        lines.append(f"  {'-'*20} " + " ".join(f"{'—'*12}" for _ in report.teams))
        for h in headers:
            vals = [str(report.team_comparison[t].get(h, 0)) for t in report.teams]
            lines.append(f"  {h:<20} " + " ".join(f"{v:>12}" for v in vals))
        lines.append("")

    if report.player_highlights:
        lines.append("--PLAYER HIGHLIGHTS —")
        for p in report.player_highlights:
            lines.append(f"  [{p['highlight'].upper()}] {p['player']} ({p['team']}): {p['detail']}")
        lines.append("")

    if report.insights:
        lines.append("--INSIGHTS —")
        for i in report.insights:
            impact_icon = {"critical": "[!!]", "high": "[!]", "medium": "[~]", "low": "[.]"}.get(i["impact"], "[ ]")
            lines.append(f"  {impact_icon} [{i['id']}] {i['title']}")
            lines.append(f"    {i['description']}")
            lines.append(f"    -> {i['recommendation']}")
            lines.append("")

    if report.recommendations:
        lines.append("--TOP RECOMMENDATIONS —")
        for idx, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {idx}. {rec}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
