from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_track_metadata(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load optional read-only playback metadata, without importing EMMR logic."""
    if path is None:
        return {}
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("track metadata must be a JSON list")
    result: dict[str, dict[str, Any]] = {}
    for raw in payload:
        if not isinstance(raw, dict) or not raw.get("song_id"):
            continue
        result[str(raw["song_id"])] = {
            "audio_url": str(raw.get("audio_url") or raw.get("song_url") or ""),
            "song_path": str(raw.get("song_path") or ""),
            "display_tags": list(raw.get("display_tags") or []),
        }
    return result
