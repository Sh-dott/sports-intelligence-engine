"""
File-based cache for match data.
Stores normalized DataFrames as parquet files to avoid re-fetching from APIs.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


class MatchCache:
    """Simple file-based parquet cache for match data."""

    def __init__(self, provider_name: str):
        self.cache_dir = CACHE_DIR / provider_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, match_id) -> Path:
        return self.cache_dir / f"{match_id}.parquet"

    def has(self, match_id) -> bool:
        return self._path(match_id).exists()

    def get(self, match_id) -> Optional[pd.DataFrame]:
        """Load cached match data if available."""
        path = self._path(match_id)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                return df
            except Exception:
                # Corrupted cache file — remove and re-fetch
                path.unlink(missing_ok=True)
        return None

    def put(self, match_id, df: pd.DataFrame) -> None:
        """Cache match data to parquet."""
        try:
            df.to_parquet(self._path(match_id), index=False)
        except Exception as e:
            print(f"[cache] Warning: Could not cache match {match_id}: {e}")

    def clear(self, match_id=None) -> None:
        """Clear specific match or all cached data."""
        if match_id:
            self._path(match_id).unlink(missing_ok=True)
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()

    def list_cached(self) -> list[str]:
        """List all cached match IDs."""
        return [f.stem for f in self.cache_dir.glob("*.parquet")]
