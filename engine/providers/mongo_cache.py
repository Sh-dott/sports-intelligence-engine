"""
MongoDB Atlas Cache Layer
Caches competitions, match lists, match events, and full analysis results
for instant subsequent loads.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGODB_URI", "")

_client: Optional[MongoClient] = None
_db = None


def _get_db():
    """Lazy-init MongoDB connection."""
    global _client, _db
    if _db is None:
        if not MONGO_URI:
            return None
        try:
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            _db = _client["sports_intel"]
            # Create indexes
            _db["competitions"].create_index("provider", background=True)
            _db["matches"].create_index([("provider", 1), ("competition_id", 1), ("season_id", 1)], background=True)
            _db["events"].create_index([("provider", 1), ("match_id", 1)], unique=True, background=True)
            _db["analyses"].create_index("analysis_id", unique=True, background=True)
            _db["analyses"].create_index("created_at", background=True)
        except Exception as e:
            print(f"[mongo] Connection failed: {e}")
            return None
    return _db


def cache_key(provider: str, *args) -> str:
    """Generate a cache key."""
    raw = f"{provider}:{'|'.join(str(a) for a in args)}"
    return hashlib.md5(raw.encode()).hexdigest()


# --- Competition Cache ---

def get_cached_competitions(provider: str) -> Optional[list]:
    """Get cached competitions list (cached for 24h)."""
    db = _get_db()
    if db is None:
        return None
    doc = db["competitions"].find_one({"provider": provider})
    if doc and doc.get("expires_at", datetime.min) > datetime.utcnow():
        return doc["data"]
    return None


def cache_competitions(provider: str, data: list):
    """Cache competitions list for 24 hours."""
    db = _get_db()
    if db is None:
        return
    db["competitions"].update_one(
        {"provider": provider},
        {"$set": {
            "provider": provider,
            "data": data,
            "expires_at": datetime.utcnow() + timedelta(hours=24),
            "updated_at": datetime.utcnow(),
        }},
        upsert=True,
    )


# --- Match List Cache ---

def get_cached_matches(provider: str, competition_id, season_id) -> Optional[list]:
    """Get cached match list (cached for 6h)."""
    db = _get_db()
    if db is None:
        return None
    doc = db["matches"].find_one({
        "provider": provider,
        "competition_id": str(competition_id),
        "season_id": str(season_id),
    })
    if doc and doc.get("expires_at", datetime.min) > datetime.utcnow():
        return doc["data"]
    return None


def cache_matches(provider: str, competition_id, season_id, data: list):
    """Cache match list for 6 hours."""
    db = _get_db()
    if db is None:
        return
    db["matches"].update_one(
        {"provider": provider, "competition_id": str(competition_id), "season_id": str(season_id)},
        {"$set": {
            "provider": provider,
            "competition_id": str(competition_id),
            "season_id": str(season_id),
            "data": data,
            "expires_at": datetime.utcnow() + timedelta(hours=6),
            "updated_at": datetime.utcnow(),
        }},
        upsert=True,
    )


# --- Match Events Cache ---

def get_cached_events(provider: str, match_id) -> Optional[pd.DataFrame]:
    """Get cached match events (permanent — match data doesn't change)."""
    db = _get_db()
    if db is None:
        return None
    doc = db["events"].find_one({"provider": provider, "match_id": str(match_id)})
    if doc:
        df = pd.DataFrame(doc["data"])
        df.attrs["sport"] = doc.get("sport", "football")
        return df
    return None


def cache_events(provider: str, match_id, df: pd.DataFrame, sport: str = "football"):
    """Cache match events permanently (finished matches never change)."""
    db = _get_db()
    if db is None:
        return
    # Convert DataFrame to list of dicts, handling NaN
    data = json.loads(df.to_json(orient="records"))
    db["events"].update_one(
        {"provider": provider, "match_id": str(match_id)},
        {"$set": {
            "provider": provider,
            "match_id": str(match_id),
            "sport": sport,
            "data": data,
            "event_count": len(data),
            "cached_at": datetime.utcnow(),
        }},
        upsert=True,
    )


# --- Full Analysis Cache ---

def get_cached_analysis(provider: str, match_id) -> Optional[dict]:
    """Get cached analysis result (permanent)."""
    db = _get_db()
    if db is None:
        return None
    doc = db["analyses"].find_one({"provider": provider, "match_id": str(match_id)})
    if doc:
        doc.pop("_id", None)
        return doc.get("report")
    return None


def cache_analysis(provider: str, match_id, report: dict):
    """Cache full analysis result permanently."""
    db = _get_db()
    if db is None:
        return
    db["analyses"].update_one(
        {"provider": provider, "match_id": str(match_id)},
        {"$set": {
            "provider": provider,
            "match_id": str(match_id),
            "analysis_id": report.get("analysis_id", ""),
            "teams": report.get("teams", []),
            "sport": report.get("sport", ""),
            "summary": report.get("summary", ""),
            "report": report,
            "created_at": datetime.utcnow(),
        }},
        upsert=True,
    )


def list_cached_analyses(limit: int = 20) -> list:
    """List recent cached analyses. Only shows analyses from the last 10 minutes."""
    db = _get_db()
    if db is None:
        return []

    cutoff = datetime.utcnow() - timedelta(minutes=10)

    # Clean up old entries from the recent list (keep the analysis data, just hide from recent)
    docs = db["analyses"].find(
        {"created_at": {"$gte": cutoff}},
        {"_id": 0, "analysis_id": 1, "match_id": 1, "sport": 1, "teams": 1, "summary": 1, "created_at": 1}
    ).sort("created_at", -1).limit(limit)
    results = []
    for d in docs:
        results.append({
            "analysis_id": d.get("analysis_id", ""),
            "match_id": d.get("match_id", ""),
            "sport": d.get("sport", ""),
            "teams": d.get("teams", []),
            "summary": (d.get("summary", "") or "")[:120],
            "stored_at": d.get("created_at", "").isoformat() if isinstance(d.get("created_at"), datetime) else str(d.get("created_at", "")),
        })
    return results
