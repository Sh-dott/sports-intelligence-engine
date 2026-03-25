"""
Visualization Module
Generate charts for momentum, scoring timelines, team comparisons, and more.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd

from engine.processing import MatchContext
from engine.detection import DetectedEvent


# Consistent color palette
TEAM_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
SEVERITY_COLORS = {"critical": "#D32F2F", "high": "#F57C00", "medium": "#FBC02D", "low": "#66BB6A"}


def generate_all_charts(ctx: MatchContext, detected: list[DetectedEvent],
                        output_dir: str | Path) -> list[str]:
    """Generate all visualization charts and return file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 120

    charts = []

    path = _plot_momentum_timeline(ctx, detected, output_dir)
    if path:
        charts.append(str(path))

    path = _plot_scoring_timeline(ctx, output_dir)
    if path:
        charts.append(str(path))

    path = _plot_team_comparison(ctx, output_dir)
    if path:
        charts.append(str(path))

    path = _plot_event_heatmap(ctx, output_dir)
    if path:
        charts.append(str(path))

    path = _plot_pressure_over_time(ctx, output_dir)
    if path:
        charts.append(str(path))

    return charts


def _plot_momentum_timeline(ctx: MatchContext, detected: list[DetectedEvent],
                            output_dir: Path) -> Path | None:
    """Plot momentum indicator over time with key events annotated."""
    ts = ctx.time_segments
    if ts is None or ts.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    teams = ctx.teams[:2]
    if len(teams) < 2:
        plt.close(fig)
        return None

    for i, team in enumerate(teams):
        team_data = ts[ts["team"] == team].sort_values("time_bucket")
        momentum = team_data["pressure"].values - team_data["mistakes"].values
        minutes = team_data["time_bucket"].values

        ax.plot(minutes, momentum, color=TEAM_COLORS[i], linewidth=2.5, label=team, alpha=0.9)
        ax.fill_between(minutes, 0, momentum, color=TEAM_COLORS[i], alpha=0.15)

    # Annotate key events
    for event in detected:
        if event.severity in ("critical", "high"):
            color = SEVERITY_COLORS[event.severity]
            ax.axvline(x=event.timestamp, color=color, linestyle="--", alpha=0.6, linewidth=1)
            ax.annotate(
                event.event_type.replace("_", " ").title(),
                xy=(event.timestamp, ax.get_ylim()[1] * 0.8),
                fontsize=7, rotation=45, color=color, alpha=0.8,
            )

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Match Minute")
    ax.set_ylabel("Momentum Index (pressure − mistakes)")
    ax.set_title("Momentum Timeline", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left")

    path = output_dir / "momentum_timeline.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_scoring_timeline(ctx: MatchContext, output_dir: Path) -> Path | None:
    """Plot running score over time."""
    scoring = ctx.scoring_timeline
    if scoring is None or scoring.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))

    teams = ctx.teams[:2]
    for i, team in enumerate(teams):
        team_scoring = scoring[scoring["team"] == team]
        if team_scoring.empty:
            continue

        # Add starting point
        times = [0] + team_scoring["timestamp"].tolist()
        scores = [0] + team_scoring["running_score"].tolist()
        # Extend to match end
        times.append(ctx.duration_minutes + ctx.events["timestamp"].min())
        scores.append(scores[-1])

        ax.step(times, scores, where="post", color=TEAM_COLORS[i],
                linewidth=2.5, label=team, alpha=0.9)

        # Mark each score
        for _, row in team_scoring.iterrows():
            ax.plot(row["timestamp"], row["running_score"], "o",
                    color=TEAM_COLORS[i], markersize=8, zorder=5)

    ax.set_xlabel("Match Minute")
    ax.set_ylabel("Score")
    ax.set_title("Scoring Timeline", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    path = output_dir / "scoring_timeline.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_team_comparison(ctx: MatchContext, output_dir: Path) -> Path | None:
    """Radar-style bar chart comparing teams across metrics."""
    ts = ctx.team_stats
    if ts is None or ts.empty:
        return None

    metrics = ["scoring_events", "pressure_events", "turnovers", "mistakes"]
    available = [m for m in metrics if m in ts.columns]
    if not available:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(available))
    width = 0.35
    teams = ctx.teams[:2]

    for i, team in enumerate(teams):
        team_data = ts[ts["team"] == team]
        if team_data.empty:
            continue
        values = [team_data[m].values[0] for m in available]
        offset = -width/2 + i * width
        bars = ax.bar(x + offset, values, width, label=team, color=TEAM_COLORS[i], alpha=0.85)
        ax.bar_label(bars, padding=3, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in available])
    ax.set_title("Team Comparison", fontweight="bold", fontsize=14)
    ax.legend()

    path = output_dir / "team_comparison.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_event_heatmap(ctx: MatchContext, output_dir: Path) -> Path | None:
    """Heatmap of event types per time segment per team."""
    df = ctx.events
    if df.empty:
        return None

    # Build pivot: time_bucket × event_type, count
    top_events = df["event_type"].value_counts().head(10).index.tolist()
    subset = df[df["event_type"].isin(top_events)]

    pivot = subset.pivot_table(
        index="time_bucket", columns="event_type",
        values="player", aggfunc="count", fill_value=0
    )

    if pivot.empty or pivot.shape[0] < 2:
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt="d", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Event Count"})
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Match Minute (5-min buckets)")
    ax.set_title("Event Distribution Heatmap", fontweight="bold", fontsize=14)

    path = output_dir / "event_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pressure_over_time(ctx: MatchContext, output_dir: Path) -> Path | None:
    """Stacked area chart of pressure events over time per team."""
    ts = ctx.time_segments
    if ts is None or ts.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))

    teams = ctx.teams[:2]
    for i, team in enumerate(teams):
        team_data = ts[ts["team"] == team].sort_values("time_bucket")
        ax.fill_between(
            team_data["time_bucket"], 0, team_data["pressure"],
            alpha=0.4, color=TEAM_COLORS[i], label=f"{team} pressure"
        )
        ax.plot(team_data["time_bucket"], team_data["pressure"],
                color=TEAM_COLORS[i], linewidth=2)

    ax.set_xlabel("Match Minute")
    ax.set_ylabel("Pressure Events (per 5-min segment)")
    ax.set_title("Pressure Intensity Over Time", fontweight="bold", fontsize=14)
    ax.legend()

    path = output_dir / "pressure_over_time.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
