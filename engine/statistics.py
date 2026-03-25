"""
Statistical Analysis Module
Anomaly detection, comparative analysis, and advanced metrics.
"""

from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from engine.processing import MatchContext


@dataclass
class StatisticalInsight:
    """A statistically-derived observation."""
    category: str       # performance_comparison, anomaly, trend, correlation
    metric: str
    description: str
    significance: float  # 0.0 to 1.0, higher = more significant
    data: dict


class StatisticalAnalyzer:
    """Perform statistical analysis on processed match data."""

    def __init__(self, ctx: MatchContext):
        self.ctx = ctx

    def run_all(self) -> list[StatisticalInsight]:
        """Run all statistical analyses."""
        insights = []
        insights.extend(self._compare_team_performance())
        insights.extend(self._analyze_time_trends())
        insights.extend(self._detect_anomaly_players())
        insights.extend(self._analyze_momentum_correlation())
        insights.extend(self._analyze_efficiency())
        return [i for i in insights if i.significance > 0.3]

    def _compare_team_performance(self) -> list[StatisticalInsight]:
        """Compare teams across key metrics."""
        insights = []
        ts = self.ctx.time_segments
        if ts is None or ts.empty:
            return insights

        for metric in ["scoring", "pressure", "turnovers", "mistakes"]:
            team_values = {}
            for team in self.ctx.teams:
                vals = ts[ts["team"] == team][metric].values
                if len(vals) > 0:
                    team_values[team] = vals

            if len(team_values) < 2:
                continue

            teams = list(team_values.keys())
            vals_a, vals_b = team_values[teams[0]], team_values[teams[1]]

            mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
            std_a, std_b = np.std(vals_a), np.std(vals_b)

            # Mann-Whitney U test (non-parametric, works for small samples)
            if len(vals_a) >= 3 and len(vals_b) >= 3:
                try:
                    u_stat, p_value = scipy_stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                    significance = 1.0 - p_value
                except ValueError:
                    significance = 0.0
            else:
                # Simple effect size for small samples
                pooled_std = np.sqrt((std_a**2 + std_b**2) / 2) if (std_a + std_b) > 0 else 1
                effect_size = abs(mean_a - mean_b) / max(pooled_std, 0.01)
                significance = min(effect_size / 2, 1.0)

            if significance > 0.3:
                better = teams[0] if mean_a > mean_b else teams[1]
                worse = teams[1] if mean_a > mean_b else teams[0]
                better_mean = max(mean_a, mean_b)
                worse_mean = min(mean_a, mean_b)

                if metric in ["turnovers", "mistakes"]:
                    # Fewer is better for these
                    better, worse = worse, better
                    better_mean, worse_mean = worse_mean, better_mean

                insights.append(StatisticalInsight(
                    category="performance_comparison",
                    metric=metric,
                    description=(
                        f"{better} significantly outperforms {worse} in {metric} "
                        f"(avg {better_mean:.1f} vs {worse_mean:.1f} per 5-min segment)."
                    ),
                    significance=round(significance, 3),
                    data={
                        "teams": {teams[0]: {"mean": round(mean_a, 2), "std": round(std_a, 2)},
                                  teams[1]: {"mean": round(mean_b, 2), "std": round(std_b, 2)}},
                    },
                ))

        return insights

    def _analyze_time_trends(self) -> list[StatisticalInsight]:
        """Detect performance trends over time (improving/declining)."""
        insights = []
        ts = self.ctx.time_segments
        if ts is None or ts.empty or len(ts) < 4:
            return insights

        for team in self.ctx.teams:
            team_ts = ts[ts["team"] == team].sort_values("time_bucket")

            for metric in ["pressure", "scoring", "mistakes"]:
                values = team_ts[metric].values
                if len(values) < 4:
                    continue

                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, values)

                r_squared = r_value ** 2
                significance = r_squared

                if significance > 0.4 and abs(slope) > 0.1:
                    direction = "increasing" if slope > 0 else "declining"

                    # First half vs second half comparison
                    mid = len(values) // 2
                    first_half = np.mean(values[:mid])
                    second_half = np.mean(values[mid:])

                    if metric == "mistakes":
                        quality = "concerning" if slope > 0 else "positive"
                    else:
                        quality = "positive" if slope > 0 else "concerning"

                    insights.append(StatisticalInsight(
                        category="trend",
                        metric=f"{team}_{metric}",
                        description=(
                            f"{team}'s {metric} shows a {quality} {direction} trend "
                            f"(first half avg: {first_half:.1f}, second half avg: {second_half:.1f}). "
                            f"{'They finish stronger than they start.' if quality == 'positive' else 'Performance deteriorates as the match progresses.'}"
                        ),
                        significance=round(significance, 3),
                        data={
                            "slope": round(slope, 3),
                            "r_squared": round(r_squared, 3),
                            "first_half_avg": round(first_half, 2),
                            "second_half_avg": round(second_half, 2),
                        },
                    ))

        return insights

    def _detect_anomaly_players(self) -> list[StatisticalInsight]:
        """Identify players whose stats are statistical outliers."""
        insights = []
        ps = self.ctx.player_stats
        if ps is None or ps.empty or len(ps) < 4:
            return insights

        for metric in ["mistake_rate", "events_per_minute", "scoring_events"]:
            if metric not in ps.columns:
                continue

            values = ps[metric].values
            if np.std(values) < 0.001:
                continue

            z_scores = np.abs(scipy_stats.zscore(values, nan_policy="omit"))

            for idx in np.where(z_scores > 1.5)[0]:
                player = ps.iloc[idx]
                z = z_scores[idx]
                val = player[metric]
                mean = np.mean(values)

                direction = "above" if val > mean else "below"
                is_negative = (metric == "mistake_rate" and direction == "above") or \
                              (metric in ["events_per_minute", "scoring_events"] and direction == "below")

                insights.append(StatisticalInsight(
                    category="anomaly",
                    metric=f"player_{metric}",
                    description=(
                        f"{player['player']} ({player['team']}) has an unusually "
                        f"{'high' if val > mean else 'low'} {metric.replace('_', ' ')} "
                        f"({val:.2f} vs team avg {mean:.2f}). "
                        f"{'This is a liability.' if is_negative else 'This is a standout performance.'}"
                    ),
                    significance=round(min(z / 3, 1.0), 3),
                    data={
                        "player": player["player"],
                        "team": player["team"],
                        "value": round(val, 3),
                        "team_mean": round(mean, 3),
                        "z_score": round(z, 2),
                    },
                ))

        return insights

    def _analyze_momentum_correlation(self) -> list[StatisticalInsight]:
        """Analyze whether pressure leads to scoring."""
        insights = []
        ts = self.ctx.time_segments
        if ts is None or ts.empty or len(ts) < 6:
            return insights

        for team in self.ctx.teams:
            team_ts = ts[ts["team"] == team].sort_values("time_bucket")
            pressure = team_ts["pressure"].values
            scoring = team_ts["scoring"].values

            if len(pressure) < 4 or np.std(pressure) < 0.01 or np.std(scoring) < 0.01:
                continue

            corr, p_value = scipy_stats.pearsonr(pressure, scoring)
            significance = abs(corr)

            if significance > 0.4:
                if corr > 0:
                    desc = (
                        f"{team}'s pressure directly converts to scoring (r={corr:.2f}). "
                        f"When they sustain pressure, goals follow. Their attack is clinical."
                    )
                else:
                    desc = (
                        f"{team}'s pressure does NOT convert to scoring (r={corr:.2f}). "
                        f"They create chances but fail to capitalize. "
                        f"Finishing or final-third decision-making needs improvement."
                    )

                insights.append(StatisticalInsight(
                    category="correlation",
                    metric=f"{team}_pressure_scoring",
                    description=desc,
                    significance=round(significance, 3),
                    data={"correlation": round(corr, 3), "p_value": round(p_value, 4)},
                ))

        return insights

    def _analyze_efficiency(self) -> list[StatisticalInsight]:
        """Analyze scoring efficiency: goals per pressure event."""
        insights = []
        team_stats = self.ctx.team_stats
        if team_stats is None or team_stats.empty:
            return insights

        for _, row in team_stats.iterrows():
            team = row["team"]
            pressure = row.get("pressure_events", 0)
            scoring = row.get("scoring_events", 0)

            if pressure > 0:
                efficiency = scoring / pressure
                insights.append(StatisticalInsight(
                    category="efficiency",
                    metric=f"{team}_scoring_efficiency",
                    description=(
                        f"{team} converts {efficiency:.0%} of pressure events into scores "
                        f"({int(scoring)} scores from {int(pressure)} pressure events). "
                        f"{'Highly clinical.' if efficiency > 0.3 else 'Room for improvement in finishing.' if efficiency < 0.15 else 'Average conversion rate.'}"
                    ),
                    significance=round(min(abs(efficiency - 0.2) * 3, 1.0), 3),
                    data={"efficiency": round(efficiency, 3), "scoring": int(scoring), "pressure": int(pressure)},
                ))

        return insights
