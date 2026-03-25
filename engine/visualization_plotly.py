"""
Interactive Visualization Module (Plotly)
Generates Plotly chart JSON for the web dashboard.
"""

import json

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from engine.processing import MatchContext
from engine.detection import DetectedEvent

TEAM_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]


def generate_all_plotly(ctx: MatchContext, detected: list[DetectedEvent]) -> dict[str, str]:
    """Generate all Plotly charts and return dict of name -> JSON string."""
    charts = {}

    c = _plot_momentum_timeline(ctx, detected)
    if c:
        charts["momentum_timeline"] = c

    c = _plot_scoring_timeline(ctx)
    if c:
        charts["scoring_timeline"] = c

    c = _plot_team_radar(ctx)
    if c:
        charts["team_radar"] = c

    c = _plot_event_heatmap(ctx)
    if c:
        charts["event_heatmap"] = c

    c = _plot_xg_timeline(ctx)
    if c:
        charts["xg_timeline"] = c

    c = _plot_pressure_over_time(ctx)
    if c:
        charts["pressure_over_time"] = c

    c = _plot_pass_network(ctx)
    if c:
        charts["pass_network"] = c

    return charts


def _plot_momentum_timeline(ctx: MatchContext, detected: list[DetectedEvent]) -> str | None:
    ts = ctx.time_segments
    if ts is None or ts.empty:
        return None

    teams = ctx.teams[:2]
    if len(teams) < 2:
        return None

    fig = go.Figure()

    for i, team in enumerate(teams):
        team_data = ts[ts["team"] == team].sort_values("time_bucket")
        momentum = team_data["pressure"].values - team_data["mistakes"].values
        minutes = team_data["time_bucket"].values

        fig.add_trace(go.Scatter(
            x=minutes, y=momentum,
            name=team, mode="lines+markers",
            line=dict(color=TEAM_COLORS[i], width=3),
            fill="tozeroy", fillcolor=f"rgba({','.join(str(int(TEAM_COLORS[i][j:j+2], 16)) for j in (1,3,5))},0.1)",
        ))

    # Add key event annotations
    for event in detected:
        if event.severity in ("critical", "high") and event.event_type in ("game_changer", "momentum_shift", "defensive_collapse"):
            fig.add_vline(x=event.timestamp, line_dash="dash", line_color="#888", opacity=0.5)
            fig.add_annotation(
                x=event.timestamp, y=1, yref="paper",
                text=event.event_type.replace("_", " ").title()[:20],
                showarrow=False, font=dict(size=9, color="#666"),
                textangle=-45,
            )

    fig.update_layout(
        title="Momentum Timeline",
        xaxis_title="Match Minute",
        yaxis_title="Momentum Index (pressure - mistakes)",
        template="plotly_white",
        height=450,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)

    return fig.to_json()


def _plot_scoring_timeline(ctx: MatchContext) -> str | None:
    scoring = ctx.scoring_timeline
    if scoring is None or scoring.empty:
        return None

    fig = go.Figure()
    teams = ctx.teams[:2]

    for i, team in enumerate(teams):
        team_scoring = scoring[scoring["team"] == team]
        if team_scoring.empty:
            continue

        times = [0] + team_scoring["timestamp"].tolist()
        scores = [0] + team_scoring["running_score"].tolist()
        times.append(ctx.duration_minutes + ctx.events["timestamp"].min())
        scores.append(scores[-1])

        hover = ["Start"] + [
            f"{row['player']}<br>{row['event_type']}"
            for _, row in team_scoring.iterrows()
        ] + ["End"]

        fig.add_trace(go.Scatter(
            x=times, y=scores, name=team,
            mode="lines+markers",
            line=dict(color=TEAM_COLORS[i], width=3, shape="hv"),
            marker=dict(size=10),
            hovertext=hover, hoverinfo="text+x+y",
        ))

    fig.update_layout(
        title="Scoring Timeline",
        xaxis_title="Match Minute",
        yaxis_title="Score",
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(dtick=1),
    )

    return fig.to_json()


def _plot_team_radar(ctx: MatchContext) -> str | None:
    ts = ctx.team_stats
    if ts is None or ts.empty:
        return None

    teams = ctx.teams[:2]
    metrics = ["scoring_events", "pressure_events", "turnovers", "mistakes", "unique_players"]
    labels = ["Scoring", "Pressure", "Turnovers", "Mistakes", "Players Used"]

    # xG is in team_comparison (from insights), not team_stats

    fig = go.Figure()

    for i, team in enumerate(teams):
        team_data = ts[ts["team"] == team]
        if team_data.empty:
            continue

        values = [float(team_data[m].iloc[0]) for m in metrics if m in team_data.columns]
        avail_labels = [labels[j] for j, m in enumerate(metrics) if m in team_data.columns]

        # Normalize to 0-1 for radar
        max_vals = [max(float(ts[m].max()), 1) for m in metrics if m in ts.columns]
        norm_values = [v / mv for v, mv in zip(values, max_vals)]
        norm_values.append(norm_values[0])  # Close the polygon
        plot_labels = avail_labels + [avail_labels[0]]

        fig.add_trace(go.Scatterpolar(
            r=norm_values, theta=plot_labels,
            fill="toself", name=team,
            line=dict(color=TEAM_COLORS[i]),
            opacity=0.7,
            hovertext=[f"{l}: {v:.0f}" for l, v in zip(avail_labels, values)] + [""],
            hoverinfo="text",
        ))

    fig.update_layout(
        title="Team Comparison Radar",
        template="plotly_white",
        height=450,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )

    return fig.to_json()


def _plot_event_heatmap(ctx: MatchContext) -> str | None:
    df = ctx.events
    if df.empty:
        return None

    top_events = df["event_type"].value_counts().head(10).index.tolist()
    subset = df[df["event_type"].isin(top_events)]

    pivot = subset.pivot_table(
        index="time_bucket", columns="event_type",
        values="player", aggfunc="count", fill_value=0
    )

    if pivot.empty or pivot.shape[0] < 2:
        return None

    fig = px.imshow(
        pivot.values,
        labels=dict(x="Event Type", y="Match Minute", color="Count"),
        x=pivot.columns.tolist(),
        y=[f"{int(t)}'" for t in pivot.index],
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )

    fig.update_layout(
        title="Event Distribution Heatmap",
        template="plotly_white",
        height=500,
        margin=dict(t=50, b=40),
    )

    return fig.to_json()


def _plot_xg_timeline(ctx: MatchContext) -> str | None:
    """Cumulative xG over time vs actual goals."""
    xg_data = ctx.xg_analysis
    if not xg_data:
        return None

    df = ctx.events
    shots = df[df["event_type"].isin(["shot", "shot_on_target", "goal"]) & df.get("xg", pd.Series(dtype=float)).notna()].copy()

    if "xg" not in df.columns or shots.empty:
        return None

    shots = df[df["xg"].notna()].copy()
    if shots.empty:
        return None

    fig = go.Figure()
    teams = ctx.teams[:2]

    for i, team in enumerate(teams):
        team_shots = shots[shots["team"] == team].sort_values("timestamp")
        if team_shots.empty:
            continue

        # Cumulative xG line
        cum_xg = team_shots["xg"].cumsum()
        fig.add_trace(go.Scatter(
            x=team_shots["timestamp"], y=cum_xg,
            name=f"{team} xG", mode="lines",
            line=dict(color=TEAM_COLORS[i], width=2, dash="dash"),
        ))

        # Actual goals as steps
        goals = team_shots[team_shots["event_type"] == "goal"]
        if not goals.empty:
            goal_times = [0] + goals["timestamp"].tolist()
            goal_counts = list(range(len(goal_times)))
            goal_times.append(ctx.duration_minutes + ctx.events["timestamp"].min())
            goal_counts.append(goal_counts[-1])

            fig.add_trace(go.Scatter(
                x=goal_times, y=goal_counts,
                name=f"{team} Goals", mode="lines",
                line=dict(color=TEAM_COLORS[i], width=3, shape="hv"),
            ))

    fig.update_layout(
        title="Expected Goals (xG) vs Actual Goals",
        xaxis_title="Match Minute",
        yaxis_title="Cumulative xG / Goals",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig.to_json()


def _plot_pressure_over_time(ctx: MatchContext) -> str | None:
    ts = ctx.time_segments
    if ts is None or ts.empty:
        return None

    fig = go.Figure()
    teams = ctx.teams[:2]

    for i, team in enumerate(teams):
        team_data = ts[ts["team"] == team].sort_values("time_bucket")
        fig.add_trace(go.Scatter(
            x=team_data["time_bucket"], y=team_data["pressure"],
            name=team, mode="lines",
            line=dict(color=TEAM_COLORS[i], width=2),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(TEAM_COLORS[i][j:j+2], 16)) for j in (1,3,5))},0.2)",
        ))

    fig.update_layout(
        title="Pressure Intensity Over Time",
        xaxis_title="Match Minute",
        yaxis_title="Pressure Events (per 5-min segment)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig.to_json()


def _plot_pass_network(ctx: MatchContext) -> str | None:
    """Plot passing network for the first team."""
    pn = ctx.pass_network
    if not pn:
        return None

    team = ctx.teams[0] if ctx.teams else None
    if not team or team not in pn:
        return None

    data = pn[team]
    nodes = pd.DataFrame(data["nodes"])
    edges = pd.DataFrame(data["edges"])

    if nodes.empty or edges.empty:
        return None

    fig = go.Figure()

    # Draw edges
    for _, edge in edges.head(15).iterrows():
        passer = nodes[nodes["player"] == edge["player"]]
        receiver = nodes[nodes["player"] == edge["pass_recipient"]]
        if passer.empty or receiver.empty:
            continue

        fig.add_trace(go.Scatter(
            x=[passer["avg_x"].iloc[0], receiver["avg_x"].iloc[0]],
            y=[passer["avg_y"].iloc[0], receiver["avg_y"].iloc[0]],
            mode="lines",
            line=dict(width=max(1, edge["count"] / 3), color="rgba(100,100,100,0.3)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Draw nodes
    fig.add_trace(go.Scatter(
        x=nodes["avg_x"], y=nodes["avg_y"],
        mode="markers+text",
        marker=dict(
            size=nodes["total_passes"].clip(lower=5) / 2,
            color=TEAM_COLORS[0], opacity=0.8,
            line=dict(width=1, color="white"),
        ),
        text=nodes["player"].str.split().str[-1],  # Last name only
        textposition="top center",
        textfont=dict(size=9),
        hovertext=[
            f"{row['player']}<br>Passes: {row['total_passes']}<br>Events: {row['total_events']}"
            for _, row in nodes.iterrows()
        ],
        hoverinfo="text",
        name=team,
    ))

    fig.update_layout(
        title=f"Pass Network: {team}",
        xaxis_title="Pitch Position (goal ->)",
        yaxis_title="Width",
        template="plotly_white",
        height=450,
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 100]),
        showlegend=False,
    )

    return fig.to_json()
