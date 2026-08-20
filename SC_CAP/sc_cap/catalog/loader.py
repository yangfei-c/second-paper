from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = (
    "song_id", "split", "pred_va", "genre_prob", "instrument_prob",
    "mood_prob", "tag_prob", "va_embedding", "tag_embedding",
)


@dataclass(frozen=True)
class MusicCatalog:
    """Read-only, deployment-facing view of the frozen MSMMR catalog."""

    song_id: np.ndarray
    split: np.ndarray
    pred_va: np.ndarray
    genre_prob: np.ndarray
    instrument_prob: np.ndarray
    mood_prob: np.ndarray
    tag_prob: np.ndarray
    va_embedding: np.ndarray
    tag_embedding: np.ndarray

    def __len__(self) -> int:
        return int(self.song_id.shape[0])

    def row(self, index: int) -> dict[str, object]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return {
            "song_id": str(self.song_id[index]),
            "split": str(self.split[index]),
            "pred_va": self.pred_va[index].astype(float).tolist(),
            "genre_prob": self.genre_prob[index],
            "instrument_prob": self.instrument_prob[index],
            "mood_prob": self.mood_prob[index],
            "tag_prob": self.tag_prob[index],
            "va_embedding": self.va_embedding[index],
            "tag_embedding": self.tag_embedding[index],
        }

    def as_public_row(self, index: int) -> dict[str, object]:
        """Return JSON-safe fields used by planner/API responses."""
        return {"song_id": str(self.song_id[index]),
                "split": str(self.split[index]),
                "pred_va": self.pred_va[index].astype(float).tolist()}


def load_catalog(path: str | Path, *, expected_count: int | None = 55525) -> MusicCatalog:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Music Catalog not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_FIELDS if key not in data.files]
        if missing:
            raise ValueError(f"Catalog missing fields: {missing}")
        arrays = {key: np.asarray(data[key]) for key in REQUIRED_FIELDS}
    n = len(arrays["song_id"])
    if expected_count is not None and n != expected_count:
        raise ValueError(f"Catalog size mismatch: expected {expected_count}, got {n}")
    if arrays["pred_va"].shape != (n, 2):
        raise ValueError(f"pred_va must have shape ({n}, 2), got {arrays['pred_va'].shape}")
    for key, dim in (("genre_prob", 87), ("instrument_prob", 40), ("mood_prob", 56),
                     ("tag_prob", 183), ("va_embedding", 256), ("tag_embedding", 256)):
        if arrays[key].shape != (n, dim):
            raise ValueError(f"{key} must have shape ({n}, {dim}), got {arrays[key].shape}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if arrays["split"].shape != (n,):
        raise ValueError("split must be one-dimensional")
    if not np.isfinite(arrays["pred_va"]).all() or (arrays["pred_va"] < -1).any() or (arrays["pred_va"] > 1).any():
        raise ValueError("pred_va must be finite and in [-1, 1]")
    return MusicCatalog(**arrays)
