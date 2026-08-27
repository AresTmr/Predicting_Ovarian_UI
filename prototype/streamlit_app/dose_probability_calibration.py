from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import json

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_ROOT = (
    REPO_ROOT
    / "models"
    / "layer1_strategy"
    / "calibration"
    / "ui_reduced_holdout"
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_path(target: str) -> Path:
    return CALIBRATION_ROOT / f"{target.lower()}_temperature.json"


def apply_temperature_to_logits(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    value = float(temperature)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Calibration temperature must be positive, got {temperature!r}")
    return logits / value


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    scaled = values / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


@lru_cache(maxsize=3)
def load_temperature_calibration(
    target: str,
    labels: tuple[str, ...],
    checkpoint_path_string: str,
) -> dict[str, Any]:
    path = calibration_path(target)
    if not path.exists():
        raise FileNotFoundError(f"Dose probability calibration artifact not found: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "dose-temperature-calibration-v1":
        raise ValueError(f"Unsupported calibration schema in {path}")
    if str(artifact.get("target")) != str(target):
        raise ValueError(f"Calibration target mismatch in {path}")
    if list(artifact.get("labels") or []) != list(labels):
        raise ValueError(f"Calibration class order mismatch in {path}")
    checkpoint_path = Path(checkpoint_path_string)
    expected_hash = str(artifact.get("checkpoint_sha256") or "")
    if not checkpoint_path.exists() or file_sha256(checkpoint_path) != expected_hash:
        raise ValueError(f"Calibration checkpoint provenance mismatch for {target}")
    deployed_temperature = float(artifact.get("deployed_temperature", np.nan))
    if not np.isfinite(deployed_temperature) or deployed_temperature <= 0:
        raise ValueError(f"Invalid deployed temperature in {path}")
    return artifact


def target_calibration(
    target: str,
    labels: Sequence[str],
    checkpoint_path: Path,
) -> dict[str, Any]:
    return load_temperature_calibration(
        str(target),
        tuple(str(label) for label in labels),
        str(checkpoint_path.resolve()),
    )
