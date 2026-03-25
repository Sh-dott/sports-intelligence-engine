"""
Analysis Storage Module
Simple file-based JSON storage for analysis results.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime

STORAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "analyses"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def save_analysis(report_dict: dict) -> str:
    """Save analysis result and return analysis ID."""
    analysis_id = str(uuid.uuid4())[:8]
    report_dict["analysis_id"] = analysis_id
    report_dict["stored_at"] = datetime.now().isoformat()

    path = STORAGE_DIR / f"{analysis_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=str)

    return analysis_id


def load_analysis(analysis_id: str) -> dict | None:
    """Load stored analysis."""
    path = STORAGE_DIR / f"{analysis_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_analyses() -> list[dict]:
    """List all stored analyses with summary metadata."""
    results = []
    for path in sorted(STORAGE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "analysis_id": path.stem,
                "match_id": data.get("match_id", ""),
                "sport": data.get("sport", ""),
                "teams": data.get("teams", []),
                "summary": data.get("summary", "")[:120],
                "stored_at": data.get("stored_at", ""),
            })
        except Exception:
            continue
    return results


def delete_analysis(analysis_id: str) -> bool:
    """Delete stored analysis."""
    path = STORAGE_DIR / f"{analysis_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
