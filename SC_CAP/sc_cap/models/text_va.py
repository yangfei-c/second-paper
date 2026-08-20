from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


class TextVAInference:
    """Lazy MSMMR Text-VA loader; no checkpoint is loaded at import time."""

    def __init__(self, checkpoint: str | Path, msmmr_root: str | Path,
                 device: str = "cpu", allow_fallback: bool = False):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.msmmr_root = Path(msmmr_root).expanduser().resolve()
        self.device_name = device
        self.allow_fallback = allow_fallback
        self._model = None
        self._ready = False

    def _load(self) -> None:
        if self._ready:
            return
        try:
            import torch
            if str(self.msmmr_root) not in sys.path:
                sys.path.insert(0, str(self.msmmr_root))
            from msmmr.training.checkpoints import load_model_state
            from msmmr.training.factories import build_text_model
            from msmmr.training.utilities import get_device
            with self.checkpoint.open("rb") as handle:
                checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
            train_config = checkpoint.get("train_config")
            if checkpoint.get("task_name") != "text_va" or not isinstance(train_config, dict):
                raise ValueError("invalid MSMMR Text-VA checkpoint")
            device = get_device(self.device_name)
            self._model = build_text_model(train_config, device)
            load_model_state(self._model, checkpoint)
            self._model.eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
            self._device = device
        except Exception:
            if not self.allow_fallback:
                raise
            self._model = None
        self._ready = True

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        self._load()
        if self._model is None:
            # Explicit smoke-test fallback; never used unless requested.
            value = float(np.tanh((sum(ord(c) for c in text) % 200 - 100) / 90.0))
            arousal = float(np.tanh((len(text) - 20) / 20.0))
            return {"text_pred_v": value, "text_pred_a": arousal,
                    "text_embedding": None, "fallback": True}
        import torch
        with torch.inference_mode():
            output = self._model([text.strip()], device=self._device)
        va = output["va_predictions"][0].detach().cpu().numpy().astype(float)
        return {"text_pred_v": float(va[0]), "text_pred_a": float(va[1]),
                "text_embedding": output["text_embedding"][0].detach().cpu().numpy().astype(float).tolist(),
                "fallback": False}
