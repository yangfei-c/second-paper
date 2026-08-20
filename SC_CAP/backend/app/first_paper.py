from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, XLMRobertaForSequenceClassification


FIRST_STRATEGIES = {
    "calm": [-0.2437, -0.3755],
    "energize": [0.1588, 0.2711],
    "comfort": [-0.1940, -0.3179],
}
FIRST_TAGS = [
    "ambient", "classical", "electronic", "folk", "hiphop", "jazz", "metal", "pop", "rock", "soundtrack",
    "bass", "drums", "guitar", "piano", "strings", "synthesizer", "voice",
    "dark", "dream", "energetic", "epic", "film", "happy", "love", "relaxing", "sad",
]
FIRST_STRATEGY_TAGS = {
    "maintain": ["auto"],
    "calm": ["ambient", "relaxing", "piano", "dream", "classical"],
    "energize": ["energetic", "electronic", "rock", "pop", "happy"],
    "comfort": ["piano", "sad", "classical", "dream", "relaxing"],
}


class _LegacyXLMRobertaVA(XLMRobertaForSequenceClassification):
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output.logits = torch.tanh(output.logits)
        return output


class LegacyFirstTextVA:
    """Minimal loader for the frozen first-paper checkpoint; no training path."""

    def __init__(self, checkpoint_path: Path, device_name: str = "cuda") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"first-paper Text-VA checkpoint not found: {self.checkpoint_path}")
        config = AutoConfig.from_pretrained("xlm-roberta-base", num_labels=2, local_files_only=True)
        model = _LegacyXLMRobertaVA(config)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        normalized = {str(key).removeprefix("model."): value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(normalized, strict=False)
        if unexpected:
            raise ValueError(f"first-paper checkpoint has unexpected state keys: {unexpected[:3]}")
        if any(not key.startswith("classifier") for key in missing):
            raise ValueError(f"first-paper checkpoint is incomplete: {missing[:3]}")
        self.model = model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", local_files_only=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def predict(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        self._load()
        assert self.model is not None and self.tokenizer is not None
        encoded = self.tokenizer(text.strip(), max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.inference_mode():
            output = self.model(input_ids=encoded["input_ids"].to(self.device), attention_mask=encoded["attention_mask"].to(self.device))
        return [float(value) for value in output.logits[0].detach().cpu().tolist()]


@dataclass
class FirstPaperSession:
    session_id: str
    state: dict
    recommendations: list[dict]
    cursor: int = 0


class FirstPaperService:
    """The existing first-paper VA+Tag ranking, retained independently of paper two."""

    def __init__(self, music_db_path: Path, text_checkpoint: Path, device: str) -> None:
        raw = json.loads(music_db_path.read_text(encoding="utf-8"))
        self.tracks = [self._normalize(item) for item in raw if isinstance(item, dict)]
        self.tracks = [item for item in self.tracks if item is not None]
        if not self.tracks:
            raise ValueError("first-paper music library is empty")
        self.z_matrix = np.asarray([item["va"] for item in self.tracks], dtype=np.float32)
        self.y_matrix = np.asarray([item["tag26"] for item in self.tracks], dtype=np.float32)
        self.prototypes = self._compute_tag_prototypes()
        self.text_va = LegacyFirstTextVA(text_checkpoint, device)
        self.sessions: dict[str, FirstPaperSession] = {}

    @staticmethod
    def _normalize(item: dict) -> dict | None:
        vector = item.get("semantic_vector_28d")
        if not isinstance(vector, list) or len(vector) < 28:
            return None
        try:
            values = [float(value) for value in vector[:28]]
        except (TypeError, ValueError):
            return None
        return {"song_id": str(item.get("song_id", "Unknown")),
                "audio_url": str(item.get("audio_url") or item.get("song_url") or ""),
                "display_tags": list(item.get("display_tags") or []),
                "va": values[:2], "tag26": values[2:28]}

    @staticmethod
    def _clip(va: np.ndarray) -> np.ndarray:
        return np.clip(va, -1.0, 1.0)

    def _compute_tag_prototypes(self, top_percent: float = 0.05) -> np.ndarray:
        prototypes = []
        for column in range(self.y_matrix.shape[1]):
            probabilities = self.y_matrix[:, column]
            selected = probabilities >= np.quantile(probabilities, 1.0 - top_percent)
            weights = probabilities[selected]
            points = self.z_matrix[selected]
            if float(weights.sum()) <= 1.0e-8:
                prototypes.append(self.z_matrix.mean(axis=0))
            else:
                prototypes.append((points * weights[:, None]).sum(axis=0) / (weights.sum() + 1.0e-8))
        return np.asarray(prototypes, dtype=np.float32)

    def _prior(self, strategy: str, mu_q: np.ndarray) -> tuple[np.ndarray, list[str]]:
        labels = FIRST_STRATEGY_TAGS[strategy]
        if labels == ["auto"]:
            distances = np.sum((self.prototypes - mu_q[None, :]) ** 2, axis=1)
            labels = [FIRST_TAGS[index] for index in np.argsort(distances, kind="stable")[:5]]
        prior = np.zeros(len(FIRST_TAGS), dtype=np.float32)
        selected = [label for label in labels if label in FIRST_TAGS]
        if selected:
            indices = [FIRST_TAGS.index(label) for label in selected]
            distances = np.sum((self.prototypes[indices] - mu_q[None, :]) ** 2, axis=1)
            logits = -10.0 * distances
            weights = np.exp(logits - logits.max())
            weights = weights / (weights.sum() + 1.0e-8)
            for index, weight in zip(indices, weights):
                prior[index] = float(weight)
        return prior, selected

    def recommend(self, text: str, strategy: str, strength: float = 0.5) -> dict:
        if strategy not in {*FIRST_STRATEGIES, "maintain"}:
            raise ValueError("unknown strategy")
        z_q = np.asarray(self.text_va.predict(text), dtype=np.float32)
        anchor = z_q if strategy == "maintain" else np.asarray(FIRST_STRATEGIES[strategy], dtype=np.float32)
        mu_q = self._clip((1.0 - float(strength)) * z_q + float(strength) * anchor)
        prior, tags = self._prior(strategy, mu_q)
        curr_dist = np.sum((self.z_matrix - z_q[None, :]) ** 2, axis=1)
        target_dist = np.sum((self.z_matrix - mu_q[None, :]) ** 2, axis=1)
        scores = -curr_dist - target_dist + self.y_matrix @ prior
        order = np.argsort(-scores, kind="stable")[:10]
        recommendations = [self._track_response(int(index), float(scores[index])) for index in order]
        session = FirstPaperSession(uuid.uuid4().hex, {"z_q": z_q.tolist(), "mu_q": mu_q.tolist(),
            "strategy": strategy, "strength": float(strength), "selected_strategy_tags": tags}, recommendations)
        self.sessions[session.session_id] = session
        return self._response(session)

    def next(self, session_id: str) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("session not found")
        if session.cursor >= len(session.recommendations) - 1:
            raise ValueError("no more candidates")
        session.cursor += 1
        return self._response(session)

    def _track_response(self, index: int, score: float) -> dict:
        item = self.tracks[index]
        return {"song_id": item["song_id"], "audio_url": item["audio_url"],
                "display_tags": item["display_tags"], "va": item["va"], "score": score}

    @staticmethod
    def _response(session: FirstPaperSession) -> dict:
        state, track = session.state, session.recommendations[session.cursor]
        labels = {"comfort": "安慰", "calm": "平静", "energize": "激活", "maintain": "保持"}
        tag_text = "、".join(track["display_tags"][:3]) or "当前情绪目标"
        return {"session_id": session.session_id, "track": track,
                "initial_va": state["z_q"], "target_va": state["mu_q"],
                "strategy": state["strategy"], "strength": state["strength"],
                "has_next": session.cursor < len(session.recommendations) - 1,
                "explanation": f"这首音乐更贴近你的{labels[state['strategy']]}目标，并匹配 {tag_text} 等特征。"}
