from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Catalog:
    song_id: np.ndarray
    pred_va: np.ndarray
    tag_embedding: np.ndarray
    tag_prob: np.ndarray
    metadata: dict[str, dict]
    playable: np.ndarray

    @classmethod
    def load(cls, catalog_path: Path, metadata_path: Path) -> "Catalog":
        if not catalog_path.is_file():
            raise FileNotFoundError(f"Music Catalog not found: {catalog_path}")
        with np.load(catalog_path, allow_pickle=False) as data:
            required = {"song_id", "pred_va", "tag_embedding", "tag_prob"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"Music Catalog missing fields: {sorted(missing)}")
            song_id = np.asarray(data["song_id"])
            pred_va = np.asarray(data["pred_va"], dtype=np.float32)
            tag_embedding = np.asarray(data["tag_embedding"], dtype=np.float32)
            tag_prob = np.asarray(data["tag_prob"], dtype=np.float32)
        if song_id.shape != (55525,) or pred_va.shape != (55525, 2):
            raise ValueError("unexpected frozen Music Catalog shape")
        if tag_embedding.shape != (55525, 256) or tag_prob.shape != (55525, 183):
            raise ValueError("unexpected Music Catalog content-feature shape")
        if not np.isfinite(pred_va).all() or np.any(pred_va < -1) or np.any(pred_va > 1):
            raise ValueError("catalog pred_va must be finite and in [-1, 1]")
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else []
        metadata = {
            str(item["song_id"]): {
                "audio_url": str(item.get("audio_url") or ""),
                "song_path": str(item.get("song_path") or ""),
                "display_tags": list(item.get("display_tags") or []),
            }
            for item in raw_metadata if isinstance(item, dict) and item.get("song_id")
        }
        norms = np.linalg.norm(tag_embedding, axis=1, keepdims=True)
        tag_embedding = tag_embedding / np.maximum(norms, 1.0e-8)
        playable = np.asarray([bool(metadata.get(str(item), {}).get("audio_url")) for item in song_id], dtype=bool)
        return cls(song_id, pred_va, tag_embedding, tag_prob, metadata, playable)

    def public_track(self, index: int) -> dict:
        song_id = str(self.song_id[index])
        return {
            "song_id": song_id,
            "audio_url": self.metadata.get(song_id, {}).get("audio_url", ""),
            "display_tags": self.metadata.get(song_id, {}).get("display_tags", []),
            "pred_va": [float(self.pred_va[index, 0]), float(self.pred_va[index, 1])],
        }
