"""
Machine Learning Module (Bonus)
Clustering and pattern recognition to identify play styles and similar match scenarios.
"""

from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from engine.processing import MatchContext


@dataclass
class ClusterResult:
    """Result of clustering analysis."""
    name: str
    n_clusters: int
    labels: list[int]
    descriptions: list[str]
    silhouette: float
    feature_importance: dict


class PatternAnalyzer:
    """ML-based pattern analysis using clustering."""

    def __init__(self, ctx: MatchContext):
        self.ctx = ctx

    def analyze_play_patterns(self) -> list[ClusterResult]:
        """Run all clustering analyses."""
        results = []

        ts_result = self._cluster_time_segments()
        if ts_result:
            results.append(ts_result)

        player_result = self._cluster_players()
        if player_result:
            results.append(player_result)

        return results

    def _cluster_time_segments(self) -> ClusterResult | None:
        """
        Cluster 5-minute time segments to identify distinct phases of play.
        E.g., "defensive grind", "open attacking play", "chaotic transition".
        """
        ts = self.ctx.time_segments
        if ts is None or ts.empty or len(ts) < 6:
            return None

        features = ["events", "scoring", "turnovers", "pressure", "mistakes"]
        available = [f for f in features if f in ts.columns]
        if len(available) < 3:
            return None

        X = ts[available].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Find optimal k using silhouette score
        best_k, best_score = 2, -1
        max_k = min(5, len(X_scaled) - 1)

        for k in range(2, max_k + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        # Describe each cluster
        descriptions = []
        feature_importance = {}
        for c in range(best_k):
            mask = labels == c
            cluster_data = ts[mask]
            means = cluster_data[available].mean()

            # Characterize the cluster
            desc_parts = []
            if means.get("pressure", 0) > ts["pressure"].mean() * 1.3:
                desc_parts.append("high pressure")
            elif means.get("pressure", 0) < ts["pressure"].mean() * 0.7:
                desc_parts.append("low pressure")

            if means.get("scoring", 0) > ts["scoring"].mean() * 1.3:
                desc_parts.append("high scoring")
            elif means.get("scoring", 0) < ts["scoring"].mean() * 0.5:
                desc_parts.append("low scoring")

            if means.get("turnovers", 0) > ts["turnovers"].mean() * 1.3:
                desc_parts.append("transitional/chaotic")
            if means.get("mistakes", 0) > ts["mistakes"].mean() * 1.3:
                desc_parts.append("error-prone")

            if not desc_parts:
                desc_parts.append("balanced play")

            phase_name = ", ".join(desc_parts)
            count = mask.sum()
            descriptions.append(
                f"Phase {c+1} ({phase_name}): {count} segments. "
                f"Avg events={means.get('events', 0):.1f}, "
                f"pressure={means.get('pressure', 0):.1f}, "
                f"scoring={means.get('scoring', 0):.1f}"
            )

            feature_importance[f"cluster_{c}"] = {f: round(means.get(f, 0), 2) for f in available}

        return ClusterResult(
            name="time_segment_phases",
            n_clusters=best_k,
            labels=labels.tolist(),
            descriptions=descriptions,
            silhouette=round(best_score, 3),
            feature_importance=feature_importance,
        )

    def _cluster_players(self) -> ClusterResult | None:
        """
        Cluster players by their statistical profiles to identify play style archetypes.
        E.g., "defensive anchor", "creative playmaker", "clinical finisher".
        """
        ps = self.ctx.player_stats
        if ps is None or ps.empty or len(ps) < 5:
            return None

        features = ["scoring_events", "turnovers", "pressure_events", "mistakes",
                     "events_per_minute", "mistake_rate"]
        available = [f for f in features if f in ps.columns]
        if len(available) < 3:
            return None

        X = ps[available].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Try DBSCAN first for natural clusters
        db = DBSCAN(eps=1.2, min_samples=2)
        db_labels = db.fit_predict(X_scaled)
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)

        if n_clusters_db >= 2:
            labels = db_labels
            n_clusters = n_clusters_db
            method = "DBSCAN"
        else:
            # Fall back to KMeans
            k = min(3, len(X_scaled) - 1)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            n_clusters = k
            method = "KMeans"

        try:
            valid_mask = labels >= 0
            if valid_mask.sum() >= 2 and len(set(labels[valid_mask])) >= 2:
                sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
            else:
                sil = 0.0
        except Exception:
            sil = 0.0

        # Describe player archetypes
        descriptions = []
        feature_importance = {}
        unique_labels = sorted(set(labels))

        for c in unique_labels:
            if c == -1:
                descriptions.append("Outliers: players with unique profiles")
                continue

            mask = labels == c
            cluster_players = ps[mask]
            means = cluster_players[available].mean()
            player_names = cluster_players["player"].tolist()

            # Determine archetype
            archetype_parts = []
            if means.get("scoring_events", 0) > ps["scoring_events"].mean() * 1.3:
                archetype_parts.append("scorer")
            if means.get("pressure_events", 0) > ps["pressure_events"].mean() * 1.3:
                archetype_parts.append("high-impact")
            if means.get("mistakes", 0) > ps["mistakes"].mean() * 1.3:
                archetype_parts.append("risk-taker")
            if means.get("turnovers", 0) > ps["turnovers"].mean() * 1.3:
                archetype_parts.append("defensive")
            if not archetype_parts:
                archetype_parts.append("balanced")

            archetype = " / ".join(archetype_parts)
            descriptions.append(
                f"Archetype '{archetype}': {', '.join(player_names[:5])}"
            )
            feature_importance[f"cluster_{c}"] = {f: round(means.get(f, 0), 2) for f in available}

        return ClusterResult(
            name="player_archetypes",
            n_clusters=n_clusters,
            labels=labels.tolist(),
            descriptions=descriptions,
            silhouette=round(sil, 3),
            feature_importance=feature_importance,
        )
