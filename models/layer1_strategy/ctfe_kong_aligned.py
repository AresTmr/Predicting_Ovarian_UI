from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS, FORBIDDEN_FEATURE_FRAGMENTS
from models.layer1_strategy.ctfe_neural import build_raw_neural_ctfe_arrays, prepare_neural_ctfe_arrays

KONG_SEMANTIC_LABELS = ["stop", "low_dose", "medium_low", "medium_high", "high_dose"]
KONG_LABEL_MAP = dict(zip(CTFE_DOSE_LABELS, KONG_SEMANTIC_LABELS))
DEFAULT_KNN_GRID = (10, 25, 50, 100)
DEFAULT_VOTE_MODES = ("uniform", "distance")


def kong_dose_display_label(internal_label: str) -> str:
    return KONG_LABEL_MAP.get(str(internal_label), str(internal_label))


def apply_kong_afc_sensitivity_filter(frame: pd.DataFrame) -> pd.DataFrame:
    if "afc" not in frame.columns:
        raise ValueError("Kong AFC sensitivity analysis requires afc.")
    afc = pd.to_numeric(frame["afc"], errors="coerce")
    return frame.loc[afc.between(7, 30, inclusive="both")].copy()


def validate_kong_feature_columns(columns: Iterable[str]) -> None:
    bad = [str(column) for column in columns if any(fragment in str(column).lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)]
    if bad:
        raise ValueError(f"Kong-aligned CTFE features include future/outcome columns: {bad[:20]}")


def apply_kong_sliding_window_stage(gn_day: pd.Series) -> pd.Series:
    values = pd.to_numeric(gn_day, errors="coerce")
    result = pd.Series("unknown", index=values.index, dtype=object)
    result.loc[values.notna() & values.lt(13)] = "pre_window"
    result.loc[values.notna() & values.ge(13)] = "sliding_window"
    return result


def compose_ctfe_sliding_window_predictions(
    frame: pd.DataFrame,
    *,
    day_col: str = "Day",
    ctfe_col: str = "ctfe_prediction",
    sliding_window_col: str = "sliding_window_prediction",
    sw_start_day: int = 13,
) -> pd.Series:
    """Compose Kong-style CTFE-sw predictions using the SW head from ``sw_start_day`` onward."""

    required = {day_col, ctfe_col, sliding_window_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing CTFE sliding-window columns: {missing}")
    day = pd.to_numeric(frame[day_col], errors="coerce")
    result = frame[ctfe_col].astype(object).copy()
    candidate = frame[sliding_window_col].astype(object)
    use_window = day.ge(int(sw_start_day)) & candidate.notna()
    result.loc[use_window] = candidate.loc[use_window]
    return result


def _label_accuracy(subset: pd.DataFrame, *, truth_col: str, prediction_col: str) -> float:
    valid = subset[truth_col].notna() & subset[prediction_col].notna()
    if not bool(valid.any()):
        return float("nan")
    truth = subset.loc[valid, truth_col].astype(str).to_numpy()
    prediction = subset.loc[valid, prediction_col].astype(str).to_numpy()
    return float(np.mean(truth == prediction))


def ctfe_daily_accuracy_table(
    predictions: pd.DataFrame,
    *,
    day_col: str = "Day",
    truth_col: str = "truth",
    ctfe_col: str = "ctfe_prediction",
    ctfe_sw_col: str = "ctfe_sw_prediction",
    min_day: int = 1,
    max_day: int = 20,
    at_risk_counts: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Build the Kong Table-3-style daily CTFE/CTFE-sw accuracy table."""

    required = {day_col, truth_col, ctfe_col, ctfe_sw_col}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing CTFE daily accuracy columns: {missing}")
    day_values = pd.to_numeric(predictions[day_col], errors="coerce")
    rows: list[dict[str, Any]] = []
    for day in range(int(min_day), int(max_day) + 1):
        subset = predictions.loc[day_values.eq(day)].copy()
        rows.append(
            {
                "Day": int(day),
                "Daily count": int(at_risk_counts.get(day, len(subset)) if at_risk_counts is not None else len(subset)),
                "CTFE": _label_accuracy(subset, truth_col=truth_col, prediction_col=ctfe_col),
                "CTFE-sw": _label_accuracy(subset, truth_col=truth_col, prediction_col=ctfe_sw_col),
            }
        )
    return pd.DataFrame(rows)


class DenseTDNNBlock(nn.Module):
    """Compact densely connected TDNN encoder for masked monitoring sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.project = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.tdnn1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.tdnn2 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.merge = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x0 = torch.relu(self.project(sequence.transpose(1, 2)))
        x1 = self.dropout(torch.relu(self.tdnn1(x0)))
        x2 = self.dropout(torch.relu(self.tdnn2(torch.cat([x0, x1], dim=1))))
        out = torch.relu(self.merge(torch.cat([x0, x1, x2], dim=1))).transpose(1, 2)
        return out * mask.unsqueeze(-1)


class MaskedMomentPooling(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1.0)
        mean = (encoded * weight).sum(dim=1) / denom
        centered = (encoded - mean.unsqueeze(1)) * weight
        variance = (centered.pow(2).sum(dim=1) / denom).clamp_min(1e-6)
        std = variance.sqrt()
        skew = centered.pow(3).sum(dim=1) / denom / std.pow(3)
        kurtosis = centered.pow(4).sum(dim=1) / denom / variance.pow(2)
        moments = torch.cat([mean, variance, skew.clamp(-10.0, 10.0), kurtosis.clamp(0.0, 20.0)], dim=1)
        return self.out(moments)


class KongAlignedCTFENetwork(nn.Module):
    """D-TDNN-style dual encoder with AddGate embedding for Kong-aligned comparison."""

    def __init__(self, static_dim: int, dynamic_dim: int, hidden_dim: int = 96, num_classes: int = 5, dropout: float = 0.10):
        super().__init__()
        self.cross_encoder = DenseTDNNBlock(static_dim + dynamic_dim, hidden_dim, dropout)
        self.dynamic_encoder = DenseTDNNBlock(dynamic_dim, hidden_dim, dropout)
        self.cross_pool = MaskedMomentPooling(hidden_dim, dropout)
        self.dynamic_pool = MaskedMomentPooling(hidden_dim, dropout)
        self.add_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static_sequence = static_x.unsqueeze(1).expand(-1, dynamic_x.shape[1], -1)
        cross_encoded = self.cross_encoder(torch.cat([static_sequence, dynamic_x], dim=-1), mask)
        dynamic_encoded = self.dynamic_encoder(dynamic_x, mask)
        cross_h = self.cross_pool(cross_encoded, mask)
        dynamic_h = self.dynamic_pool(dynamic_encoded, mask)
        gate = self.add_gate(torch.cat([dynamic_h, cross_h], dim=1))
        embedding = dynamic_h + gate * cross_h
        return self.classifier(embedding), embedding


@dataclass
class KongScopeResult:
    scope: str
    output_dir: str
    metrics: pd.DataFrame
    daily_metrics: pd.DataFrame
    predictions: pd.DataFrame
    training_curve: pd.DataFrame
    selected_knn: dict[str, Any]
    class_distribution: pd.DataFrame
    confusion_matrix: pd.DataFrame
    summary: dict[str, Any]


def evaluate_probability_head(y_true: np.ndarray, probabilities: np.ndarray, *, scope: str, split: str, head: str) -> dict[str, Any]:
    prediction = np.asarray(probabilities).argmax(axis=1)
    report = classification_report(
        y_true,
        prediction,
        labels=list(range(len(KONG_SEMANTIC_LABELS))),
        target_names=KONG_SEMANTIC_LABELS,
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "scope": scope,
        "head": head,
        "split": split,
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(range(len(KONG_SEMANTIC_LABELS))))),
    }
    for label in KONG_SEMANTIC_LABELS:
        stats = report.get(label, {})
        result[f"precision_{label}"] = float(stats.get("precision", 0.0))
        result[f"recall_{label}"] = float(stats.get("recall", 0.0))
        result[f"f1_{label}"] = float(stats.get("f1-score", 0.0))
        result[f"support_{label}"] = int(stats.get("support", 0))
    return result


def select_cosine_knn_head(metrics: pd.DataFrame) -> dict[str, Any]:
    valid = metrics[(metrics["split"].eq("valid")) & metrics["head"].astype(str).str.startswith("cosine_knn")].copy()
    if valid.empty:
        raise ValueError("No validation cosine-KNN candidates available.")
    tie_break_columns = [column for column in ["weighted_f1", "macro_f1", "accuracy"] if column in valid.columns]
    best = valid.sort_values(tie_break_columns, ascending=False).iloc[0].to_dict()
    best["selection_split"] = "valid"
    best["test_used_for_selection"] = False
    return best


def _loader(data: Any, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.tensor(data.static[mask], dtype=torch.float32),
        torch.tensor(data.dynamic[mask], dtype=torch.float32),
        torch.tensor(data.mask[mask], dtype=torch.float32),
        torch.tensor(data.y[mask], dtype=torch.long),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def _predict(model: nn.Module, data: Any, mask: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for static_x, dynamic_x, obs_mask, _ in _loader(data, mask, 1024, False):
            score, embedding = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            logits.append(score.cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    raw = np.vstack(logits)
    raw = raw - raw.max(axis=1, keepdims=True)
    probabilities = np.exp(raw)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, np.vstack(embeddings)


def _vote_neighbor_probabilities(distances: np.ndarray, indices: np.ndarray, train_y: np.ndarray, *, k: int, vote_mode: str) -> np.ndarray:
    if vote_mode not in DEFAULT_VOTE_MODES:
        raise ValueError(f"Unknown vote_mode: {vote_mode}")
    use_k = min(int(k), indices.shape[1])
    labels = train_y[indices[:, :use_k]]
    neighbor_distances = distances[:, :use_k]
    weights = np.ones_like(neighbor_distances) if vote_mode == "uniform" else 1.0 / np.maximum(neighbor_distances, 1e-8)
    probabilities = np.zeros((len(indices), len(KONG_SEMANTIC_LABELS)), dtype=float)
    for class_id in range(len(KONG_SEMANTIC_LABELS)):
        probabilities[:, class_id] = (weights * labels.__eq__(class_id)).sum(axis=1)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def build_cosine_knn_probability_grid(
    train_embedding: np.ndarray,
    train_y: np.ndarray,
    query_embedding: np.ndarray,
    *,
    k_grid: Iterable[int] = DEFAULT_KNN_GRID,
) -> dict[tuple[int, str], np.ndarray]:
    k_values = [int(k) for k in k_grid]
    max_k = min(max(k_values), len(train_embedding))
    neighbor = NearestNeighbors(n_neighbors=max_k, metric="cosine", algorithm="brute", n_jobs=-1).fit(train_embedding)
    distances, indices = neighbor.kneighbors(query_embedding)
    return {
        (int(k), vote_mode): _vote_neighbor_probabilities(distances, indices, train_y, k=int(k), vote_mode=vote_mode)
        for k in k_values
        for vote_mode in DEFAULT_VOTE_MODES
    }


def cosine_knn_probabilities(train_embedding: np.ndarray, train_y: np.ndarray, query_embedding: np.ndarray, *, k: int, vote_mode: str) -> np.ndarray:
    return build_cosine_knn_probability_grid(train_embedding, train_y, query_embedding, k_grid=[int(k)])[(int(k), vote_mode)]


def _daily_metric_table(frame: pd.DataFrame, y: np.ndarray, probabilities: np.ndarray, *, scope: str, head: str, split: str) -> pd.DataFrame:
    working = frame.copy().reset_index(drop=True)
    working["truth"] = y
    working["prediction"] = probabilities.argmax(axis=1)
    working["window_stage"] = apply_kong_sliding_window_stage(working["gn_day"])
    rows: list[dict[str, Any]] = []
    day = pd.to_numeric(working["gn_day"], errors="coerce")
    working["gn_day_band"] = pd.cut(day, bins=[-np.inf, 3, 6, 9, 12, np.inf], labels=["d0_3", "d4_6", "d7_9", "d10_12", "d13_plus"])
    for group_col in ["window_stage", "gn_day_band"]:
        for group_value, subset in working.groupby(group_col, dropna=False):
            if subset.empty:
                continue
            rows.append({
                "scope": scope,
                "head": head,
                "split": split,
                "group_col": group_col,
                "group_value": str(group_value),
                "sample_count": int(len(subset)),
                "accuracy": float(accuracy_score(subset["truth"], subset["prediction"])),
                "macro_f1": float(f1_score(subset["truth"], subset["prediction"], average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(subset["truth"], subset["prediction"], average="weighted", zero_division=0)),
            })
    return pd.DataFrame(rows)


def train_kong_aligned_scope(
    frame: pd.DataFrame,
    *,
    split_manifest_path: str | Path,
    output_dir: str | Path,
    scope: str,
    afc_sensitivity: bool = False,
    max_visits: int = 10,
    hidden_dim: int = 96,
    dropout: float = 0.10,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 14,
    seed: int = 20260424,
    device_name: str = "cpu",
    k_grid: Iterable[int] = DEFAULT_KNN_GRID,
) -> KongScopeResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cohort = apply_kong_afc_sensitivity_filter(frame) if afc_sensitivity else frame.copy()
    raw = build_raw_neural_ctfe_arrays(cohort, split_manifest_path=split_manifest_path, max_visits=max_visits)
    validate_kong_feature_columns(raw.static_features + raw.dynamic_features)
    data = prepare_neural_ctfe_arrays(raw)
    train_mask = data.split == "train"
    valid_mask = data.split == "valid"
    test_mask = data.split == "test"
    if not train_mask.any() or not valid_mask.any() or not test_mask.any():
        raise ValueError("Kong-aligned experiment requires train, valid and test rows.")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    model = KongAlignedCTFENetwork(data.static.shape[1], data.dynamic.shape[2], hidden_dim=hidden_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    training = _loader(data, train_mask, batch_size, True)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -1.0
    wait = 0
    curve: list[dict[str, Any]] = []
    for epoch in tqdm(range(1, epochs + 1), desc=f"Kong CTFE v2 {scope}"):
        model.train()
        loss_values: list[float] = []
        for static_x, dynamic_x, obs_mask, y in training:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            loss_values.append(float(loss.detach().cpu()))
        train_proba, _ = _predict(model, data, train_mask, device)
        valid_proba, _ = _predict(model, data, valid_mask, device)
        train_score = f1_score(data.y[train_mask], train_proba.argmax(axis=1), average="weighted", zero_division=0)
        valid_score = f1_score(data.y[valid_mask], valid_proba.argmax(axis=1), average="weighted", zero_division=0)
        curve.append({"scope": scope, "epoch": epoch, "train_loss": float(np.mean(loss_values)), "train_weighted_f1": float(train_score), "valid_weighted_f1": float(valid_score)})
        if valid_score > best_score + 1e-4:
            best_score = float(valid_score)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    all_mask = np.ones(len(data.y), dtype=bool)
    softmax, embedding = _predict(model, data, all_mask, device)
    metrics: list[dict[str, Any]] = []
    predictions = data.row_index.copy().reset_index(drop=True)
    predictions["scope"] = scope
    predictions["true_semantic_label"] = [KONG_SEMANTIC_LABELS[value] for value in data.y]
    for split, mask in [("train", train_mask), ("valid", valid_mask), ("test", test_mask)]:
        metrics.append(evaluate_probability_head(data.y[mask], softmax[mask], scope=scope, split=split, head="softmax"))
    train_embedding = embedding[train_mask]
    train_y = data.y[train_mask]
    cached: dict[tuple[str, int, str], np.ndarray] = {}
    for split, mask in [("valid", valid_mask), ("test", test_mask)]:
        probability_grid = build_cosine_knn_probability_grid(train_embedding, train_y, embedding[mask], k_grid=k_grid)
        for (k, vote_mode), probability in probability_grid.items():
            cached[(split, int(k), vote_mode)] = probability
            metrics.append(evaluate_probability_head(data.y[mask], probability, scope=scope, split=split, head=f"cosine_knn_k{int(k)}_{vote_mode}"))
    metric_frame = pd.DataFrame(metrics)
    selected = select_cosine_knn_head(metric_frame)
    selected_parts = str(selected["head"]).split("_")
    selected_k = int(selected_parts[2].replace("k", ""))
    selected_mode = selected_parts[3]
    selected["k"] = selected_k
    selected["vote_mode"] = selected_mode
    daily: list[pd.DataFrame] = []
    for split, mask in [("valid", valid_mask), ("test", test_mask)]:
        selected_probability = cached[(split, selected_k, selected_mode)]
        predictions.loc[mask, "selected_knn_prediction"] = [KONG_SEMANTIC_LABELS[value] for value in selected_probability.argmax(axis=1)]
        predictions.loc[mask, "softmax_prediction"] = [KONG_SEMANTIC_LABELS[value] for value in softmax[mask].argmax(axis=1)]
        daily.append(_daily_metric_table(data.row_index.loc[mask].reset_index(drop=True), data.y[mask], selected_probability, scope=scope, head=str(selected["head"]), split=split))
        daily.append(_daily_metric_table(data.row_index.loc[mask].reset_index(drop=True), data.y[mask], softmax[mask], scope=scope, head="softmax", split=split))
    test_probability = cached[("test", selected_k, selected_mode)]
    cm = confusion_matrix(data.y[test_mask], test_probability.argmax(axis=1), labels=list(range(len(KONG_SEMANTIC_LABELS))))
    distribution = pd.DataFrame({
        "scope": scope,
        "semantic_label": KONG_SEMANTIC_LABELS,
        "train_count": np.bincount(data.y[train_mask], minlength=5),
        "valid_count": np.bincount(data.y[valid_mask], minlength=5),
        "test_count": np.bincount(data.y[test_mask], minlength=5),
    })
    destination = Path(output_dir) / scope
    destination.mkdir(parents=True, exist_ok=True)
    curve_frame = pd.DataFrame(curve)
    daily_frame = pd.concat(daily, ignore_index=True)
    metric_frame.to_csv(destination / "kong_aligned_metrics.csv", index=False)
    daily_frame.to_csv(destination / "kong_aligned_daily_metrics.csv", index=False)
    predictions.to_csv(destination / "kong_aligned_predictions.csv", index=False)
    curve_frame.to_csv(destination / "kong_aligned_training_curve.csv", index=False)
    distribution.to_csv(destination / "kong_aligned_class_distribution.csv", index=False)
    cm_frame = pd.DataFrame(cm, index=KONG_SEMANTIC_LABELS, columns=KONG_SEMANTIC_LABELS)
    cm_frame.to_csv(destination / "kong_aligned_confusion_matrix.csv")
    torch.save({"state_dict": model.state_dict(), "static_dim": data.static.shape[1], "dynamic_dim": data.dynamic.shape[2], "hidden_dim": hidden_dim, "dropout": dropout, "max_visits": max_visits, "semantic_labels": KONG_SEMANTIC_LABELS, "best_epoch": best_epoch}, destination / "kong_aligned_model.pt")
    joblib.dump({"train_embedding": train_embedding, "train_labels": train_y, "selected_knn": selected, "static_features": data.static_features, "dynamic_features": data.dynamic_features}, destination / "kong_aligned_embedding_knn.joblib")
    summary = {"scope": scope, "best_epoch": int(best_epoch), "train_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum()), "test_rows": int(test_mask.sum()), "selected_knn_head": selected["head"], "selection_split": "valid", "test_used_for_selection": False}
    return KongScopeResult(scope, str(destination), metric_frame, daily_frame, predictions, curve_frame, selected, distribution, cm_frame, summary)
