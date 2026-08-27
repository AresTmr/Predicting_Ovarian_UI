from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from models.layer1_strategy.ctfe_auxiliary import (
    CTFE_DOSE_LABELS,
    CTFE_DYNAMIC_FEATURES,
    CTFE_ID_TO_DOSE,
    CTFE_STATIC_FEATURES,
    DEFAULT_MAX_VISITS,
    DEFAULT_SEED,
    FORBIDDEN_FEATURE_FRAGMENTS,
    create_ctfe_fsh_labels,
)
from models.layer1_strategy.ctfe_cost_sensitive import get_ctfe_cost_vector
from models.layer1_strategy.ctfe_stage import add_ctfe_stage_columns
from models.layer1_strategy.ctfe_sampling import build_training_sample_weights

CURRENT_CTFE_NEURAL_POINTER = Path("models/artifacts/current_layer1_ctfe_neural_run.txt")


@dataclass
class RawNeuralCTFEArrays:
    static: np.ndarray
    dynamic: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    row_index: pd.DataFrame
    split: np.ndarray
    static_features: list[str]
    dynamic_features: list[str]
    lengths: np.ndarray
    max_visits: int


@dataclass
class PreparedNeuralCTFEArrays:
    static: np.ndarray
    dynamic: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    row_index: pd.DataFrame
    split: np.ndarray
    static_features: list[str]
    dynamic_features: list[str]
    lengths: np.ndarray
    max_visits: int
    static_imputer: SimpleImputer
    static_scaler: StandardScaler
    dynamic_imputer: SimpleImputer
    dynamic_scaler: StandardScaler


class CTFEDeepNetwork(nn.Module):
    """Neural CTFE branch: static encoder + cross-feature gate + temporal GRU."""

    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        hidden_dim: int = 96,
        num_classes: int = 5,
        dropout: float = 0.15,
        encoder_type: str = "gru",
    ):
        super().__init__()
        if encoder_type not in {"gru", "tdnn"}:
            raise ValueError("encoder_type must be 'gru' or 'tdnn'")
        self.encoder_type = encoder_type
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_gate = nn.Sequential(nn.Linear(hidden_dim, dynamic_dim), nn.Sigmoid())
        self.dynamic_projection = nn.Sequential(nn.Linear(dynamic_dim, hidden_dim), nn.ReLU())
        if encoder_type == "gru":
            self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        else:
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        self.cross_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static_h = self.static_encoder(static_x)
        gate = self.feature_gate(static_h).unsqueeze(1)
        dynamic_gated = dynamic_x * gate * mask.unsqueeze(-1)
        dynamic_h = self.dynamic_projection(dynamic_gated)
        if self.encoder_type == "gru":
            temporal_out, _ = self.temporal_encoder(dynamic_h)
            lengths = mask.sum(dim=1).long().clamp_min(1)
            gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, temporal_out.size(-1))
            temporal_h = temporal_out.gather(1, gather_index).squeeze(1)
        else:
            temporal_out = self.temporal_encoder(dynamic_h.transpose(1, 2)).transpose(1, 2)
            temporal_out = temporal_out * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_h = temporal_out.sum(dim=1) / denom
            masked_out = temporal_out.masked_fill(mask.unsqueeze(-1) <= 0, -1e4)
            max_h = masked_out.max(dim=1).values
            temporal_h = 0.5 * (mean_h + max_h)
        fusion_gate = self.cross_gate(torch.cat([static_h, temporal_h], dim=1))
        embedding = fusion_gate * temporal_h + (1.0 - fusion_gate) * static_h
        logits = self.classifier(embedding)
        return logits, embedding


class CTFEFocalLoss(nn.Module):
    """Multi-class focal loss for imbalanced CTFE dose-class training."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone().float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        ce = -log_pt
        if self.weight is not None:
            ce = ce * self.weight[target]
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def _load_split(frame: pd.DataFrame, split_manifest_path: str | Path | None) -> pd.Series:
    if split_manifest_path is None:
        return pd.Series(["train"] * len(frame), index=frame.index)
    path = Path(split_manifest_path)
    if not path.exists() or "cycle_uid" not in frame.columns:
        return pd.Series(["train"] * len(frame), index=frame.index)
    manifest = pd.read_csv(path)
    if not {"cycle_uid", "split"}.issubset(manifest.columns):
        return pd.Series(["train"] * len(frame), index=frame.index)
    split_map = manifest.drop_duplicates("cycle_uid").set_index("cycle_uid")["split"]
    return frame["cycle_uid"].map(split_map).fillna("train")


def _numeric_columns_available(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])]


def _assert_no_forbidden(columns: list[str]) -> None:
    bad = [column for column in columns if any(fragment in column for fragment in FORBIDDEN_FEATURE_FRAGMENTS)]
    if bad:
        raise ValueError(f"Neural CTFE feature list contains leakage-prone columns: {bad[:20]}")


def _safe_numeric(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column, np.nan)]), errors="coerce").iloc[0]
    return float(value) if not pd.isna(value) else np.nan


def build_raw_neural_ctfe_arrays(
    frame: pd.DataFrame,
    *,
    split_manifest_path: str | Path | None = None,
    max_visits: int = DEFAULT_MAX_VISITS,
    eligible_only: bool = True,
) -> RawNeuralCTFEArrays:
    df = create_ctfe_fsh_labels(frame)
    if eligible_only and "has_next_visit" in df.columns:
        df = df[df["has_next_visit"].astype(bool)].copy()
    if eligible_only and "strategy_eligible_flag" in df.columns:
        df = df[df["strategy_eligible_flag"].astype(bool)].copy()
    df = df[df["ctfe_next_fsh_dose_class"].isin(CTFE_DOSE_LABELS)].copy()
    if df.empty:
        raise ValueError("No CTFE neural-eligible rows after filters.")
    if "cycle_uid" not in df.columns or "monitoring_order" not in df.columns:
        raise ValueError("Neural CTFE requires cycle_uid and monitoring_order.")

    df["split"] = _load_split(df, split_manifest_path).values
    df = df.sort_values(["cycle_uid", "monitoring_order", "visit_uid" if "visit_uid" in df.columns else "monitoring_order"]).reset_index(drop=True)
    static_features = _numeric_columns_available(df, CTFE_STATIC_FEATURES)
    dynamic_features = _numeric_columns_available(df, CTFE_DYNAMIC_FEATURES)
    _assert_no_forbidden(static_features + dynamic_features)
    groups = {cycle: group.sort_values("monitoring_order").reset_index(drop=True) for cycle, group in df.groupby("cycle_uid", sort=False)}

    static_rows: list[list[float]] = []
    dynamic_rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    lengths: list[int] = []
    row_records: list[dict[str, Any]] = []
    for _, query in df.iterrows():
        history = groups[query["cycle_uid"]][groups[query["cycle_uid"]]["monitoring_order"] <= query["monitoring_order"]].tail(max_visits)
        seq_len = int(len(history))
        sequence = np.full((max_visits, len(dynamic_features)), np.nan, dtype=float)
        mask = np.zeros(max_visits, dtype=float)
        for offset, (_, hist_row) in enumerate(history.iterrows(), start=0):
            sequence[offset, :] = [_safe_numeric(hist_row, column) for column in dynamic_features]
            mask[offset] = 1.0
        static_rows.append([_safe_numeric(query, column) for column in static_features])
        dynamic_rows.append(sequence)
        masks.append(mask)
        labels.append(int(query["ctfe_next_fsh_dose_class_id"]))
        lengths.append(seq_len)
        row_record = {
                "visit_uid": query.get("visit_uid", ""),
                "cycle_uid": query.get("cycle_uid", ""),
                "art_id": query.get("art_id", ""),
                "monitoring_order": query.get("monitoring_order", np.nan),
                "gn_day": query.get("gn_day", np.nan),
                "current_fsh_daily_dose": query.get("current_fsh_daily_dose", np.nan),
                "current_gn_dose": query.get("current_gn_dose", np.nan),
                "amh": query.get("amh", np.nan),
                "afc": query.get("afc", np.nan),
                "total_follicle_count": query.get("total_follicle_count", np.nan),
                "split": query.get("split", "train"),
                "ctfe_next_fsh_dose_class": query.get("ctfe_next_fsh_dose_class"),
                "next_fsh_daily_dose": query.get("next_fsh_daily_dose", np.nan),
            }
        row_records.append(row_record)
    return RawNeuralCTFEArrays(
        static=np.asarray(static_rows, dtype=float),
        dynamic=np.asarray(dynamic_rows, dtype=float),
        mask=np.asarray(masks, dtype=float),
        y=np.asarray(labels, dtype=int),
        row_index=add_ctfe_stage_columns(pd.DataFrame(row_records)),
        split=np.asarray(df["split"].tolist(), dtype=object),
        static_features=static_features,
        dynamic_features=dynamic_features,
        lengths=np.asarray(lengths, dtype=int),
        max_visits=max_visits,
    )


def prepare_neural_ctfe_arrays(raw: RawNeuralCTFEArrays) -> PreparedNeuralCTFEArrays:
    train_mask = raw.split == "train"
    static_imputer = SimpleImputer(strategy="median").fit(raw.static[train_mask])
    static_scaler = StandardScaler().fit(static_imputer.transform(raw.static[train_mask]))
    static = static_scaler.transform(static_imputer.transform(raw.static)).astype(np.float32)

    dyn_train_observed = raw.dynamic[train_mask][raw.mask[train_mask].astype(bool)]
    dynamic_imputer = SimpleImputer(strategy="median").fit(dyn_train_observed)
    dynamic_scaler = StandardScaler().fit(dynamic_imputer.transform(dyn_train_observed))
    flat_dynamic = raw.dynamic.reshape(-1, raw.dynamic.shape[-1])
    dynamic = dynamic_scaler.transform(dynamic_imputer.transform(flat_dynamic)).reshape(raw.dynamic.shape).astype(np.float32)
    dynamic = dynamic * raw.mask[:, :, None].astype(np.float32)
    return PreparedNeuralCTFEArrays(
        static=static,
        dynamic=dynamic,
        mask=raw.mask.astype(np.float32),
        y=raw.y,
        row_index=raw.row_index,
        split=raw.split,
        static_features=raw.static_features,
        dynamic_features=raw.dynamic_features,
        lengths=raw.lengths,
        max_visits=raw.max_visits,
        static_imputer=static_imputer,
        static_scaler=static_scaler,
        dynamic_imputer=dynamic_imputer,
        dynamic_scaler=dynamic_scaler,
    )


def _make_loader(
    data: PreparedNeuralCTFEArrays,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    *,
    sample_weights: np.ndarray | None = None,
    seed: int | None = None,
) -> DataLoader:
    tensor_data = TensorDataset(
        torch.tensor(data.static[mask], dtype=torch.float32),
        torch.tensor(data.dynamic[mask], dtype=torch.float32),
        torch.tensor(data.mask[mask], dtype=torch.float32),
        torch.tensor(data.y[mask], dtype=torch.long),
    )
    if sample_weights is None:
        return DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)
    if len(sample_weights) != len(tensor_data):
        raise ValueError("CTFE sampler weights must match selected training rows.")
    generator = torch.Generator()
    generator.manual_seed(int(seed if seed is not None else DEFAULT_SEED))
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return DataLoader(tensor_data, batch_size=batch_size, sampler=sampler, shuffle=False)


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(CTFE_DOSE_LABELS)).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _build_criterion(
    *,
    y_train: np.ndarray,
    class_weight_mode: str,
    class_cost_profile: str,
    loss_mode: str,
    focal_gamma: float,
    device: torch.device,
) -> nn.Module:
    if class_cost_profile != "none" and class_weight_mode != "none":
        raise ValueError("class_cost_profile cannot be combined with balanced class_weight_mode")
    if class_cost_profile != "none":
        class_weight = torch.as_tensor(get_ctfe_cost_vector(class_cost_profile), dtype=torch.float32, device=device)
    elif class_weight_mode == "balanced":
        class_weight = _class_weights(y_train).to(device)
    elif class_weight_mode == "none":
        class_weight = None
    else:
        raise ValueError("class_weight_mode must be 'none' or 'balanced'")

    if loss_mode == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weight)
    if loss_mode == "focal":
        return CTFEFocalLoss(gamma=focal_gamma, weight=class_weight)
    raise ValueError("loss_mode must be 'cross_entropy' or 'focal'")


def make_row_filter_mask(
    row_index: pd.DataFrame,
    *,
    row_filter_col: str | None = None,
    row_filter_values: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Build a boolean mask for stage/window-specific CTFE training."""

    if row_filter_col is None:
        return np.ones(len(row_index), dtype=bool)
    if row_filter_col not in row_index.columns:
        raise ValueError(f"Missing CTFE row filter column: {row_filter_col}")
    values = [str(value) for value in (row_filter_values or [])]
    if not values:
        raise ValueError("row_filter_values must be provided when row_filter_col is set.")
    return row_index[row_filter_col].astype(str).isin(values).to_numpy(dtype=bool)


def _predict(model: CTFEDeepNetwork, data: PreparedNeuralCTFEArrays, mask: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    embedding_list: list[np.ndarray] = []
    loader = _make_loader(data, mask, batch_size=1024, shuffle=False)
    with torch.no_grad():
        for static_x, dynamic_x, obs_mask, _ in loader:
            logits, embedding = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            logits_list.append(logits.cpu().numpy())
            embedding_list.append(embedding.cpu().numpy())
    logits_all = np.vstack(logits_list)
    logits_all = logits_all - logits_all.max(axis=1, keepdims=True)
    proba = np.exp(logits_all)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba, np.vstack(embedding_list)


def _metrics(y_true: np.ndarray, proba: np.ndarray, split: str) -> dict[str, Any]:
    pred = proba.argmax(axis=1)
    payload: dict[str, Any] = {
        "split": split,
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=list(range(len(CTFE_DOSE_LABELS))))),
    }
    report = classification_report(
        y_true,
        pred,
        labels=list(range(len(CTFE_DOSE_LABELS))),
        target_names=CTFE_DOSE_LABELS,
        output_dict=True,
        zero_division=0,
    )
    for label in CTFE_DOSE_LABELS:
        cls = report.get(label, {})
        payload[f"precision_{label}"] = float(cls.get("precision", 0.0))
        payload[f"recall_{label}"] = float(cls.get("recall", 0.0))
        payload[f"f1_{label}"] = float(cls.get("f1-score", 0.0))
        payload[f"support_{label}"] = int(cls.get("support", 0))
    return payload


def _prediction_frame(data: PreparedNeuralCTFEArrays, proba: np.ndarray) -> pd.DataFrame:
    frame = data.row_index.copy()
    frame["ctfe_neural_prediction"] = [CTFE_ID_TO_DOSE[int(i)] for i in proba.argmax(axis=1)]
    for idx, label in enumerate(CTFE_DOSE_LABELS):
        frame[f"prob_{label}"] = proba[:, idx]
    return frame


def train_neural_ctfe_model(
    *,
    input_path: str | Path = "data/processed/layer1_strategy_dataset.csv",
    split_manifest_path: str | Path = "data/splits/split_manifest_v1.csv",
    output_root: str | Path = "models/artifacts",
    run_id: str | None = None,
    max_visits: int = DEFAULT_MAX_VISITS,
    hidden_dim: int = 96,
    dropout: float = 0.15,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = DEFAULT_SEED,
    device_name: str = "auto",
    class_weight_mode: str = "none",
    class_cost_profile: str = "none",
    loss_mode: str = "cross_entropy",
    focal_gamma: float = 2.0,
    encoder_type: str = "gru",
    row_filter_col: str | None = None,
    row_filter_values: list[str] | tuple[str, ...] | None = None,
    sampling_mode: str = "none",
    update_pointer: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw = build_raw_neural_ctfe_arrays(pd.read_csv(input_path), split_manifest_path=split_manifest_path, max_visits=max_visits)
    data = prepare_neural_ctfe_arrays(raw)
    row_filter_mask = make_row_filter_mask(
        data.row_index,
        row_filter_col=row_filter_col,
        row_filter_values=row_filter_values,
    )
    train_mask = (data.split == "train") & row_filter_mask
    valid_mask = (data.split == "valid") & row_filter_mask
    test_mask = (data.split == "test") & row_filter_mask
    if train_mask.sum() == 0:
        raise ValueError(f"No CTFE training rows after row filter {row_filter_col}={row_filter_values}")
    if valid_mask.sum() == 0:
        valid_mask = train_mask
    if test_mask.sum() == 0:
        test_mask = valid_mask
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = CTFEDeepNetwork(
        static_dim=data.static.shape[1],
        dynamic_dim=data.dynamic.shape[2],
        hidden_dim=hidden_dim,
        num_classes=len(CTFE_DOSE_LABELS),
        dropout=dropout,
        encoder_type=encoder_type,
    ).to(device)
    criterion = _build_criterion(
        y_train=data.y[train_mask],
        class_weight_mode=class_weight_mode,
        class_cost_profile=class_cost_profile,
        loss_mode=loss_mode,
        focal_gamma=focal_gamma,
        device=device,
    )
    validation_criterion = _build_criterion(
        y_train=data.y[train_mask],
        class_weight_mode="none",
        class_cost_profile="none",
        loss_mode=loss_mode,
        focal_gamma=focal_gamma,
        device=device,
    )
    sample_weights, sampling_distribution = build_training_sample_weights(
        data.y,
        data.row_index,
        train_mask,
        sampling_mode=sampling_mode,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = _make_loader(
        data, train_mask, batch_size=batch_size, shuffle=sample_weights is None, sample_weights=sample_weights, seed=seed
    )
    valid_loader = _make_loader(data, valid_mask, batch_size=batch_size, shuffle=False)

    best_state = None
    best_valid_score = -float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, Any]] = []
    for epoch in tqdm(range(1, epochs + 1), desc="neural CTFE epoch"):
        model.train()
        train_losses: list[float] = []
        for static_x, dynamic_x, obs_mask, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        model.eval()
        valid_losses: list[float] = []
        with torch.no_grad():
            for static_x, dynamic_x, obs_mask, y in valid_loader:
                logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
                valid_losses.append(float(validation_criterion(logits, y.to(device)).detach().cpu().item()))
        train_proba, _ = _predict(model, data, train_mask, device)
        valid_proba, _ = _predict(model, data, valid_mask, device)
        train_metric = _metrics(data.y[train_mask], train_proba, "train")
        valid_metric = _metrics(data.y[valid_mask], valid_proba, "valid")
        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_weighted_f1": train_metric["weighted_f1"],
                "valid_weighted_f1": valid_metric["weighted_f1"],
                "train_macro_f1": train_metric["macro_f1"],
                "valid_macro_f1": valid_metric["macro_f1"],
            }
        )
        valid_score = float(valid_metric["weighted_f1"])
        if valid_score > best_valid_score + 1e-4:
            best_valid_score = valid_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    proba_all, embeddings_all = _predict(model, data, np.ones(len(data.y), dtype=bool), device)
    metrics = [
        _metrics(data.y[train_mask], proba_all[train_mask], "train"),
        _metrics(data.y[valid_mask], proba_all[valid_mask], "valid"),
        _metrics(data.y[test_mask], proba_all[test_mask], "test"),
    ]
    run_id = run_id or f"phase8_layer1_ctfe_neural_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_root) / run_id / "layer1_ctfe_neural"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "ctfe_neural_training_curve.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "ctfe_neural_metrics.csv", index=False)
    sampling_distribution.to_csv(output_dir / "ctfe_neural_sampling_distribution.csv", index=False)
    _prediction_frame(data, proba_all).to_csv(output_dir / "ctfe_neural_predictions.csv", index=False)
    cm = confusion_matrix(data.y[test_mask], proba_all[test_mask].argmax(axis=1), labels=list(range(len(CTFE_DOSE_LABELS))))
    pd.DataFrame(cm, index=CTFE_DOSE_LABELS, columns=CTFE_DOSE_LABELS).to_csv(output_dir / "ctfe_neural_confusion_matrix.csv")
    pd.DataFrame(
        {
            "dose_class": CTFE_DOSE_LABELS,
            "train_count": np.bincount(data.y[train_mask], minlength=len(CTFE_DOSE_LABELS)),
            "valid_count": np.bincount(data.y[valid_mask], minlength=len(CTFE_DOSE_LABELS)),
            "test_count": np.bincount(data.y[test_mask], minlength=len(CTFE_DOSE_LABELS)),
        }
    ).to_csv(output_dir / "ctfe_neural_class_distribution.csv", index=False)

    train_index = data.row_index.loc[train_mask].copy().reset_index(drop=True)
    train_embeddings = embeddings_all[train_mask]
    nn_index = NearestNeighbors(n_neighbors=min(50, len(train_index)), metric="euclidean").fit(train_embeddings)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "static_dim": data.static.shape[1],
            "dynamic_dim": data.dynamic.shape[2],
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "max_visits": max_visits,
            "dose_labels": CTFE_DOSE_LABELS,
            "static_features": data.static_features,
            "dynamic_features": data.dynamic_features,
            "run_id": run_id,
            "best_epoch": best_epoch,
            "loss_mode": loss_mode,
            "class_cost_profile": class_cost_profile,
            "class_cost_weights": get_ctfe_cost_vector(class_cost_profile).tolist(),
            "class_cost_scope": "train_loss_only",
            "focal_gamma": focal_gamma,
            "encoder_type": encoder_type,
            "row_filter_col": row_filter_col,
            "row_filter_values": list(row_filter_values or []),
            "sampling_mode": sampling_mode,
        },
        output_dir / "ctfe_neural_model.pt",
    )
    joblib.dump(
        {
            "static_imputer": data.static_imputer,
            "static_scaler": data.static_scaler,
            "dynamic_imputer": data.dynamic_imputer,
            "dynamic_scaler": data.dynamic_scaler,
            "row_index": train_index,
            "embeddings": train_embeddings,
            "nearest_neighbors": nn_index,
        },
        output_dir / "ctfe_neural_preprocess_and_knn.joblib",
    )
    summary = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "model_type": "neural_ctfe_gru",
        "device": str(device),
        "class_weight_mode": class_weight_mode,
        "class_cost_profile": class_cost_profile,
        "class_cost_weights": get_ctfe_cost_vector(class_cost_profile).tolist(),
        "class_cost_scope": "train_loss_only",
        "loss_mode": loss_mode,
        "focal_gamma": float(focal_gamma),
        "encoder_type": encoder_type,
        "row_filter_col": row_filter_col,
        "row_filter_values": list(row_filter_values or []),
        "sampling_mode": sampling_mode,
        "update_pointer": bool(update_pointer),
        "best_epoch": int(best_epoch),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "valid_weighted_f1": float(pd.DataFrame(metrics).query("split == 'valid'")["weighted_f1"].iloc[0]),
        "test_weighted_f1": float(pd.DataFrame(metrics).query("split == 'test'")["weighted_f1"].iloc[0]),
        "valid_log_loss": float(pd.DataFrame(metrics).query("split == 'valid'")["log_loss"].iloc[0]),
        "test_log_loss": float(pd.DataFrame(metrics).query("split == 'test'")["log_loss"].iloc[0]),
    }
    (output_dir / "ctfe_neural_summary.json").write_text(pd.Series(summary).to_json(force_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "ctfe_neural_summary.md").write_text(
        "\n".join(
            [
                "# Neural CTFE Layer1 Auxiliary Model",
                "",
                "This is a neural CTFE auxiliary branch. It does not replace the current Layer1 FSH/LH/HMG action models.",
                "",
                f"- run_id: `{run_id}`",
                f"- device: `{device}`",
                f"- loss_mode: `{loss_mode}`",
                f"- class_cost_profile: `{class_cost_profile}` weights `{get_ctfe_cost_vector(class_cost_profile).tolist()}` (`train_loss_only`; validation loss remains unweighted)",
                f"- focal_gamma: `{focal_gamma}`",
                f"- encoder_type: `{encoder_type}`",
                f"- row_filter: `{row_filter_col}={list(row_filter_values or [])}`",
                f"- sampling_mode: `{sampling_mode}` (train-only; validation/test retain observed distribution)",
                f"- best_epoch: `{best_epoch}`",
                f"- train/valid/test rows: `{summary['train_rows']}/{summary['valid_rows']}/{summary['test_rows']}`",
                f"- valid weighted-F1: `{summary['valid_weighted_f1']:.4f}`",
                f"- test weighted-F1: `{summary['test_weighted_f1']:.4f}`",
            ]
        ),
        encoding="utf-8",
    )
    if update_pointer:
        CURRENT_CTFE_NEURAL_POINTER.parent.mkdir(parents=True, exist_ok=True)
        CURRENT_CTFE_NEURAL_POINTER.write_text(run_id, encoding="utf-8")
    return summary
