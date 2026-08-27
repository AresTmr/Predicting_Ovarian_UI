#!/usr/bin/env python3
"""Phase 8.66 daily-expanded FSH absolute-dose model comparison runner.

This script trains comparison models under the same daily-expanded FSH dose
category task used by the current final CTFE(AddGate) paper row.  The final
CTFE(AddGate) row is read from the locked final-route comparison table by
default; this runner trains the comparator rows sequentially and rewrites the
two manuscript-style tables after every model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PHASE864_PATH = REPO_ROOT / "scripts" / "experiment" / "phase864_ctfe_kong_sw" / "run_phase864_ctfe_kong_sw.py"
spec = importlib.util.spec_from_file_location("phase864_ctfe_kong_sw_runner", PHASE864_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import Phase 8.64 runner from {PHASE864_PATH}")
phase864 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = phase864
spec.loader.exec_module(phase864)

from models.layer1_strategy.ctfe_kong_aligned import (  # noqa: E402
    DEFAULT_KNN_GRID,
    DenseTDNNBlock,
    KONG_SEMANTIC_LABELS,
    MaskedMomentPooling,
    build_cosine_knn_probability_grid,
)

LABELS = list(KONG_SEMANTIC_LABELS)
NUM_CLASSES = len(LABELS)
DEFAULT_MODELS = [
    "majority",
    "previous",
    "lasso",
    "lstm",
    "dtdnn",
    "gru",
    "tcn",
    "lstm_fc",
    "gru_fc",
    "tcn_fc",
    "gru_addgate",
    "gru_mag",
    "tcn_addgate",
    "tcn_mag",
    "lstm_addgate",
    "lstm_mag",
    "ctfe_lstm_fc",
    "lstm_fc_wfocal",
    "ctfe_lstm_fc_wfocal",
    "ctfe_lstm_mag",
    "ctfe_lstm_addgate",
    "dtdnn_fc",
    "dtdnn_mag",
    "dtdnn_addgate",
]
TABLE1_ROWS = [
    ("ET", "LSTM", "lstm"),
    ("ET", "D-TDNN", "dtdnn"),
    ("ET", "GRU", "gru"),
    ("ET", "TCN / 1D-CNN", "tcn"),
    ("ETS & ET", "LSTM(FC Layer)", "lstm_fc"),
    ("ETS & ET", "GRU(FC Layer)", "gru_fc"),
    ("ETS & ET", "GRU(AddGate)", "gru_addgate"),
    ("ETS & ET", "GRU(MAG)", "gru_mag"),
    ("ETS & ET", "TCN / 1D-CNN(FC Layer)", "tcn_fc"),
    ("ETS & ET", "TCN / 1D-CNN(AddGate)", "tcn_addgate"),
    ("ETS & ET", "TCN / 1D-CNN(MAG)", "tcn_mag"),
    ("ETS & ET", "LSTM(AddGate)", "lstm_addgate"),
    ("ETS & ET", "LSTM(MAG)", "lstm_mag"),
    ("ETS & ET", "CTFE-LSTM(FC Layer)", "ctfe_lstm_fc"),
    ("ETS & ET", "LSTM(FC Layer + weighted focal)", "lstm_fc_wfocal"),
    ("ETS & ET", "CTFE-LSTM(FC Layer + weighted focal)", "ctfe_lstm_fc_wfocal"),
    ("ETS & ET", "CTFE-LSTM(MAG)", "ctfe_lstm_mag"),
    ("ETS & ET", "CTFE-LSTM(AddGate)", "ctfe_lstm_addgate"),
    ("ETS & ET", "D-TDNN(FC Layer)", "dtdnn_fc"),
    ("ETS & ET", "D-TDNN(MAG)", "dtdnn_mag"),
    ("ETS & ET", "D-TDNN(AddGate)", "dtdnn_addgate"),
    ("ETS & ET", "CTFE(AddGate)", "ctfe_addgate_final"),
]
TABLE2_ROWS = [
    ("Majority baseline", "majority"),
    ("Previous-dose baseline", "previous"),
    ("LASSO logistic regression", "lasso"),
    ("CTFE(AddGate)", "ctfe_addgate_final"),
    ("CTFE-LSTM(FC Layer)", "ctfe_lstm_fc"),
    ("CTFE-LSTM(FC Layer + weighted focal)", "ctfe_lstm_fc_wfocal"),
    ("CTFE-LSTM(AddGate)", "ctfe_lstm_addgate"),
]
MODEL_DISPLAY = {key: model for _group, model, key in TABLE1_ROWS}
MODEL_DISPLAY.update({key: model for model, key in TABLE2_ROWS})
MODEL_SEED_OFFSET = {key: (idx + 1) * 137 for idx, key in enumerate(DEFAULT_MODELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train daily-expanded FSH comparison models and generate Table 1/Table 2.")
    parser.add_argument("--input", default="data/processed/layer1_strategy_dataset.csv")
    parser.add_argument("--baseline-input", default="data/processed/baseline_cycle_dataset.csv")
    parser.add_argument("--output-root", default="results/result1_deep_temporal_gn_dose_daily_expanded_model_comparison")
    parser.add_argument("--paper-table-dir", default="paper/tables/paper_claim_validation/phase866_daily_expanded_model_comparison")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Model keys or 'all'.")
    parser.add_argument("--min-day", type=int, default=1)
    parser.add_argument("--max-day", type=int, default=20)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--max-history-days", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--selection-metric", choices=["accuracy", "macro_f1", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--knn-grid", type=int, nargs="+", default=list(DEFAULT_KNN_GRID))
    parser.add_argument("--lasso-c", type=float, default=1.0)
    parser.add_argument("--lasso-max-iter", type=int, default=900)
    parser.add_argument("--ctfe-metrics-file", default="results/result1_deep_temporal_gn_dose_final_route_comparison/tables/final_route_comparison_fsh.csv")
    parser.add_argument("--ctfe-accuracy", type=float, default=None)
    parser.add_argument("--ctfe-macro-f1", type=float, default=None)
    parser.add_argument("--ctfe-weighted-f1", type=float, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def fsh_label_from_dose(value: Any) -> str | None:
    dose = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(dose):
        return None
    dose = float(dose)
    if dose <= 0:
        return "stop"
    if dose < 80:
        return "low_dose"
    if dose < 160:
        return "medium_low"
    if dose <= 240:
        return "medium_high"
    return "high_dose"


def metric_payload(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average="weighted", zero_division=0)) if len(y_true) else float("nan"),
    }


def classwise_rows(y_true: np.ndarray, y_pred: np.ndarray, *, model_key: str) -> list[dict[str, Any]]:
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=LABELS, output_dict=True, zero_division=0)
    rows = []
    for label in LABELS:
        stats = report.get(label, {})
        rows.append({
            "model_key": model_key,
            "class": label,
            "precision": float(stats.get("precision", 0.0)),
            "recall": float(stats.get("recall", 0.0)),
            "f1": float(stats.get("f1-score", 0.0)),
            "support": int(stats.get("support", 0)),
        })
    return rows


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray, *, model_key: str) -> list[dict[str, Any]]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)), normalize="true")
    rows = []
    for i, truth in enumerate(LABELS):
        for j, pred in enumerate(LABELS):
            rows.append({"model_key": model_key, "truth": truth, "prediction": pred, "normalized_count": float(cm[i, j])})
    return rows


def load_ctfe_final_metrics(args: argparse.Namespace) -> dict[str, Any]:
    manual = [args.ctfe_accuracy, args.ctfe_macro_f1, args.ctfe_weighted_f1]
    if all(value is not None for value in manual):
        return {
            "model_key": "ctfe_addgate_final",
            "model": "CTFE(AddGate)",
            "source": "manual CLI values",
            "accuracy": float(args.ctfe_accuracy),
            "macro_f1": float(args.ctfe_macro_f1),
            "weighted_f1": float(args.ctfe_weighted_f1),
        }
    metrics_path = Path(args.ctfe_metrics_file)
    if not metrics_path.is_absolute():
        metrics_path = REPO_ROOT / metrics_path
    table = pd.read_csv(metrics_path)
    subset = table[
        table["Task"].astype(str).eq("daily-expanded FSH dose category")
        & table["Model"].astype(str).eq("CTFE daily no-SW")
        & table["Sample weighting/eval"].astype(str).eq("day1-20")
    ].copy()
    if subset.empty:
        raise ValueError(f"Cannot find daily-expanded day1-20 CTFE row in {metrics_path}")
    row = subset.iloc[0]
    return {
        "model_key": "ctfe_addgate_final",
        "model": "CTFE(AddGate)",
        "source": str(metrics_path),
        "accuracy": float(row["Accuracy"]),
        "macro_f1": float(row["Macro-F1"]),
        "weighted_f1": float(row["Weighted-F1"]),
    }


def markdown_table(table: pd.DataFrame) -> str:
    headers = [str(c) for c in table.columns]
    rows = [["" if pd.isna(v) else str(v) for v in row] for row in table.to_numpy()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i]) for i in range(len(headers))]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    return "\n".join([fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |", *[fmt(row) for row in rows]])


def format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def render_tables(metrics_by_key: Mapping[str, Mapping[str, Any]], table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir.mkdir(parents=True, exist_ok=True)
    rows1 = []
    for group, model, key in TABLE1_ROWS:
        metric = metrics_by_key.get(key, {})
        rows1.append({
            "Model group": group,
            "Model": model,
            "Accuracy": format_metric(metric.get("accuracy")),
            "Macro-F1": format_metric(metric.get("macro_f1")),
            "Weighted-F1": format_metric(metric.get("weighted_f1")),
        })
    rows2 = []
    for model, key in TABLE2_ROWS:
        metric = metrics_by_key.get(key, {})
        rows2.append({
            "Model": model,
            "Accuracy": format_metric(metric.get("accuracy")),
            "Macro-F1": format_metric(metric.get("macro_f1")),
            "Weighted-F1": format_metric(metric.get("weighted_f1")),
        })
    table1 = pd.DataFrame(rows1)
    table2 = pd.DataFrame(rows2)
    table1.to_csv(table_dir / "table1_daily_expanded_architecture_comparison_fsh.csv", index=False)
    table2.to_csv(table_dir / "table2_daily_expanded_baseline_vs_final_ctfe_fsh.csv", index=False)
    (table_dir / "table1_daily_expanded_architecture_comparison_fsh.md").write_text(
        "# Table 1. Performance comparison of temporal encoding and fusion strategies for next FSH dose category prediction\n\n" + markdown_table(table1) + "\n",
        encoding="utf-8",
    )
    (table_dir / "table2_daily_expanded_baseline_vs_final_ctfe_fsh.md").write_text(
        "# Table 2. Performance comparison between baseline models and final deep temporal model\n\n" + markdown_table(table2) + "\n",
        encoding="utf-8",
    )
    raw1 = table1.copy()
    raw2 = table2.copy()
    for raw in (raw1, raw2):
        for col in ["Accuracy", "Macro-F1", "Weighted-F1"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw1.to_csv(table_dir / "table1_daily_expanded_architecture_comparison_fsh_raw.csv", index=False)
    raw2.to_csv(table_dir / "table2_daily_expanded_baseline_vs_final_ctfe_fsh_raw.csv", index=False)
    return table1, table2


def load_daily_arrays(args: argparse.Namespace) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    monitoring = pd.read_csv(args.input, low_memory=False)
    baseline = pd.read_csv(args.baseline_input, low_memory=False) if args.baseline_input else None
    frame, feature_groups, audit = phase864.prepare_daily_ctfe_frame(
        monitoring,
        baseline,
        min_day=int(args.min_day),
        max_day=int(args.max_day),
    )
    raw = phase864.build_daily_sequence_arrays(
        frame,
        static_features=feature_groups["static_features"],
        dynamic_features=feature_groups["dynamic_features"],
        max_history_days=int(args.max_history_days),
    )
    audit.update({
        "task": "daily-expanded FSH absolute dose category",
        "label_classes": LABELS,
        "sample_rows": int(len(raw.y)),
        "sample_cycles": int(raw.row_index["cycle_uid"].nunique()),
        "day_min": int(args.min_day),
        "day_max": int(args.max_day),
        "same_task_as_ctfe_final_row": True,
        "ctfe_final_metric_source": args.ctfe_metrics_file,
    })
    return raw, frame, audit


def grouped_oof_splits(groups: np.ndarray, y: np.ndarray, n_splits: int) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    n = min(int(n_splits), len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups=groups), start=1):
        yield fold, train_idx, test_idx


def last_dynamic_matrix(raw: Any) -> np.ndarray:
    dynamic = np.asarray(raw.dynamic, dtype=float)
    mask = np.asarray(raw.mask, dtype=bool)
    out = np.full((dynamic.shape[0], dynamic.shape[2]), np.nan, dtype=float)
    lengths = mask.sum(axis=1).astype(int)
    for i, length in enumerate(lengths):
        if length > 0:
            out[i, :] = dynamic[i, length - 1, :]
    return out


def save_prediction_frame(raw: Any, pred_ids: np.ndarray, *, model_key: str, path: Path) -> None:
    out = raw.row_index.copy()
    out["y_true_id"] = raw.y
    out["truth"] = [LABELS[int(v)] for v in raw.y]
    out["prediction_id"] = pred_ids.astype(int)
    out["prediction"] = [LABELS[int(v)] for v in pred_ids]
    out["model_key"] = model_key
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def summarize_model(raw: Any, pred_ids: np.ndarray, *, model_key: str, model_name: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metrics = metric_payload(raw.y, pred_ids)
    payload = {
        "model_key": model_key,
        "model": model_name,
        "sample_count": metrics["sample_count"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }
    if extra:
        payload.update(extra)
    return payload


def run_majority_baseline(raw: Any, args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    groups = raw.row_index["cycle_uid"].astype(str).to_numpy()
    pred = np.full(len(raw.y), -1, dtype=int)
    fold_rows = []
    for fold, train_idx, test_idx in grouped_oof_splits(groups, raw.y, int(args.oof_folds)):
        labels, counts = np.unique(raw.y[train_idx], return_counts=True)
        majority = int(labels[np.argmax(counts)])
        pred[test_idx] = majority
        fold_rows.append({"model_key": "majority", "fold": fold, "majority_class": LABELS[majority], **metric_payload(raw.y[test_idx], pred[test_idx])})
    save_prediction_frame(raw, pred, model_key="majority", path=out_dir / "predictions" / "majority_oof_predictions.csv")
    return summarize_model(raw, pred, model_key="majority", model_name="Majority baseline"), pd.DataFrame(fold_rows), classwise_rows(raw.y, pred, model_key="majority"), confusion_rows(raw.y, pred, model_key="majority")


def run_previous_dose_baseline(raw: Any, args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    groups = raw.row_index["cycle_uid"].astype(str).to_numpy()
    pred = np.full(len(raw.y), -1, dtype=int)
    fold_rows = []
    last_dyn = last_dynamic_matrix(raw)
    if "previous_fsh_daily_dose" not in raw.dynamic_features:
        raise ValueError("previous_fsh_daily_dose is unavailable in daily-expanded dynamic features.")
    dose_idx = raw.dynamic_features.index("previous_fsh_daily_dose")
    previous_label = np.full(len(raw.y), -1, dtype=int)
    for i, dose in enumerate(last_dyn[:, dose_idx]):
        label = fsh_label_from_dose(dose)
        if label in LABELS:
            previous_label[i] = LABELS.index(label)
    for fold, train_idx, test_idx in grouped_oof_splits(groups, raw.y, int(args.oof_folds)):
        labels, counts = np.unique(raw.y[train_idx], return_counts=True)
        fallback = int(labels[np.argmax(counts)])
        fold_pred = previous_label[test_idx].copy()
        fold_pred[fold_pred < 0] = fallback
        pred[test_idx] = fold_pred
        fold_rows.append({"model_key": "previous", "fold": fold, "fallback_class": LABELS[fallback], **metric_payload(raw.y[test_idx], pred[test_idx])})
    save_prediction_frame(raw, pred, model_key="previous", path=out_dir / "predictions" / "previous_oof_predictions.csv")
    return summarize_model(raw, pred, model_key="previous", model_name="Previous-dose baseline"), pd.DataFrame(fold_rows), classwise_rows(raw.y, pred, model_key="previous"), confusion_rows(raw.y, pred, model_key="previous")


def run_lasso_baseline(raw: Any, args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    groups = raw.row_index["cycle_uid"].astype(str).to_numpy()
    X = np.hstack([np.asarray(raw.static, dtype=float), last_dynamic_matrix(raw)])
    pred = np.full(len(raw.y), -1, dtype=int)
    fold_rows = []
    for fold, train_idx, test_idx in grouped_oof_splits(groups, raw.y, int(args.oof_folds)):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                penalty="l1",
                solver="saga",
                C=float(args.lasso_c),
                max_iter=int(args.lasso_max_iter),
                tol=1e-3,
                multi_class="multinomial",
                n_jobs=-1,
                random_state=int(args.seed) + int(fold),
            ),
        )
        model.fit(X[train_idx], raw.y[train_idx])
        pred[test_idx] = model.predict(X[test_idx]).astype(int)
        fold_rows.append({"model_key": "lasso", "fold": fold, **metric_payload(raw.y[test_idx], pred[test_idx])})
    save_prediction_frame(raw, pred, model_key="lasso", path=out_dir / "predictions" / "lasso_oof_predictions.csv")
    return summarize_model(raw, pred, model_key="lasso", model_name="LASSO logistic regression"), pd.DataFrame(fold_rows), classwise_rows(raw.y, pred, model_key="lasso"), confusion_rows(raw.y, pred, model_key="lasso")


class DynamicRNNClassifier(nn.Module):
    def __init__(self, dynamic_dim: int, hidden_dim: int, num_classes: int, dropout: float, *, cell: str):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(dynamic_dim, hidden_dim, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = mask.sum(dim=1).long().clamp_min(1).cpu()
        packed = pack_padded_sequence(dynamic_x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):
            h = hidden[0][-1]
        else:
            h = hidden[-1]
        embedding = self.drop(torch.relu(self.norm(h)))
        return self.classifier(embedding), embedding


class DynamicTDNNClassifier(nn.Module):
    def __init__(self, dynamic_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.encoder = DenseTDNNBlock(dynamic_dim, hidden_dim, dropout)
        self.pool = MaskedMomentPooling(hidden_dim, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(dynamic_x, mask)
        embedding = self.pool(encoded, mask)
        return self.classifier(embedding), embedding


class DynamicTCNClassifier(nn.Module):
    def __init__(self, dynamic_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dynamic_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.net(dynamic_x.transpose(1, 2)).transpose(1, 2) * mask.unsqueeze(-1)
        weight = mask.unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1.0)
        mean = (encoded * weight).sum(dim=1) / denom
        masked = encoded.masked_fill(~mask.bool().unsqueeze(-1), -1e4)
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(max_pool.lt(-1e3), torch.zeros_like(max_pool), max_pool)
        embedding = self.fuse(torch.cat([mean, max_pool], dim=1))
        return self.classifier(embedding), embedding


class FusionRNNFCClassifier(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.dynamic_rnn = nn.LSTM(dynamic_dim, hidden_dim, batch_first=True)
        self.cross_rnn = nn.LSTM(static_dim + dynamic_dim, hidden_dim, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _last(self, rnn: nn.LSTM, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp_min(1).cpu()
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = rnn(packed)
        return hidden[0][-1]

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static_sequence = static_x.unsqueeze(1).expand(-1, dynamic_x.shape[1], -1)
        dynamic_h = self._last(self.dynamic_rnn, dynamic_x, mask)
        cross_h = self._last(self.cross_rnn, torch.cat([static_sequence, dynamic_x], dim=-1), mask)
        embedding = self.fuse(torch.cat([dynamic_h, cross_h], dim=1))
        return self.classifier(embedding), embedding


class FusionGRUClassifier(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, hidden_dim: int, num_classes: int, dropout: float, *, mode: str):
        super().__init__()
        self.mode = mode
        self.dynamic_rnn = nn.GRU(dynamic_dim, hidden_dim, batch_first=True)
        self.cross_rnn = nn.GRU(static_dim + dynamic_dim, hidden_dim, batch_first=True)
        if mode == "fc":
            self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        elif mode == "addgate":
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        elif mode == "mag":
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.shift = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _last(self, rnn: nn.GRU, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp_min(1).cpu()
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = rnn(packed)
        return hidden[-1]

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static_sequence = static_x.unsqueeze(1).expand(-1, dynamic_x.shape[1], -1)
        dynamic_h = self._last(self.dynamic_rnn, dynamic_x, mask)
        cross_h = self._last(self.cross_rnn, torch.cat([static_sequence, dynamic_x], dim=-1), mask)
        joint = torch.cat([dynamic_h, cross_h], dim=1)
        if self.mode == "fc":
            embedding = self.fuse(joint)
        elif self.mode == "addgate":
            embedding = self.fuse(dynamic_h + self.gate(joint) * cross_h)
        else:
            embedding = self.fuse(dynamic_h + self.gate(joint) * self.shift(cross_h))
        return self.classifier(embedding), embedding


class FusionTCNClassifier(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, hidden_dim: int, num_classes: int, dropout: float, *, mode: str):
        super().__init__()
        self.mode = mode
        self.dynamic_net = self._make_net(dynamic_dim, hidden_dim, dropout)
        self.cross_net = self._make_net(static_dim + dynamic_dim, hidden_dim, dropout)
        self.dynamic_fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.cross_fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        if mode == "fc":
            self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        elif mode == "addgate":
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        elif mode == "mag":
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.shift = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _make_net(self, input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )

    def _encode(self, net: nn.Sequential, fuse: nn.Sequential, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = net(sequence.transpose(1, 2)).transpose(1, 2) * mask.unsqueeze(-1)
        weight = mask.unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1.0)
        mean = (encoded * weight).sum(dim=1) / denom
        masked = encoded.masked_fill(~mask.bool().unsqueeze(-1), -1e4)
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(max_pool.lt(-1e3), torch.zeros_like(max_pool), max_pool)
        return fuse(torch.cat([mean, max_pool], dim=1))

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static_sequence = static_x.unsqueeze(1).expand(-1, dynamic_x.shape[1], -1)
        dynamic_h = self._encode(self.dynamic_net, self.dynamic_fuse, dynamic_x, mask)
        cross_h = self._encode(self.cross_net, self.cross_fuse, torch.cat([static_sequence, dynamic_x], dim=-1), mask)
        joint = torch.cat([dynamic_h, cross_h], dim=1)
        if self.mode == "fc":
            embedding = self.fuse(joint)
        elif self.mode == "addgate":
            embedding = self.fuse(dynamic_h + self.gate(joint) * cross_h)
        else:
            embedding = self.fuse(dynamic_h + self.gate(joint) * self.shift(cross_h))
        return self.classifier(embedding), embedding


class FusionRNNAddGateClassifier(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.dynamic_rnn = nn.LSTM(dynamic_dim, hidden_dim, batch_first=True)
        self.cross_rnn = nn.LSTM(static_dim + dynamic_dim, hidden_dim, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _last(self, rnn: nn.LSTM, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp_min(1).cpu()
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = rnn(packed)
        return hidden[0][-1]

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static_sequence = static_x.unsqueeze(1).expand(-1, dynamic_x.shape[1], -1)
        dynamic_h = self._last(self.dynamic_rnn, dynamic_x, mask)
        cross_h = self._last(self.cross_rnn, torch.cat([static_sequence, dynamic_x], dim=-1), mask)
        joint = torch.cat([dynamic_h, cross_h], dim=1)
        embedding = self.fuse(dynamic_h + self.gate(joint) * cross_h)
        return self.classifier(embedding), embedding


class FusionRNNMAGClassifier(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.dynamic_rnn = nn.LSTM(dynamic_dim, hidden_dim, batch_first=True)
        self.cross_rnn = nn.LSTM(static_dim + dynamic_dim, hidden_dim, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.shift = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _last(self, rnn: nn.LSTM, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp_min(1).cpu()
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = rnn(packed)
        return hidden[0][-1]

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static_sequence = static_x.unsqueeze(1).expand(-1, dynamic_x.shape[1], -1)
        dynamic_h = self._last(self.dynamic_rnn, dynamic_x, mask)
        cross_h = self._last(self.cross_rnn, torch.cat([static_sequence, dynamic_x], dim=-1), mask)
        joint = torch.cat([dynamic_h, cross_h], dim=1)
        embedding = self.fuse(dynamic_h + self.gate(joint) * self.shift(cross_h))
        return self.classifier(embedding), embedding


class FusionTDNNClassifier(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, hidden_dim: int, num_classes: int, dropout: float, *, mode: str):
        super().__init__()
        self.mode = mode
        self.cross_encoder = DenseTDNNBlock(static_dim + dynamic_dim, hidden_dim, dropout)
        self.dynamic_encoder = DenseTDNNBlock(dynamic_dim, hidden_dim, dropout)
        self.cross_pool = MaskedMomentPooling(hidden_dim, dropout)
        self.dynamic_pool = MaskedMomentPooling(hidden_dim, dropout)
        if mode == "fc":
            self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        elif mode == "addgate":
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        elif mode == "mag":
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.shift = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static_sequence = static_x.unsqueeze(1).expand(-1, dynamic_x.shape[1], -1)
        cross_encoded = self.cross_encoder(torch.cat([static_sequence, dynamic_x], dim=-1), mask)
        dynamic_encoded = self.dynamic_encoder(dynamic_x, mask)
        cross_h = self.cross_pool(cross_encoded, mask)
        dynamic_h = self.dynamic_pool(dynamic_encoded, mask)
        joint = torch.cat([dynamic_h, cross_h], dim=1)
        if self.mode == "fc":
            embedding = self.fuse(joint)
        elif self.mode == "addgate":
            embedding = self.fuse(dynamic_h + self.gate(joint) * cross_h)
        else:
            embedding = self.fuse(dynamic_h + self.gate(joint) * self.shift(cross_h))
        return self.classifier(embedding), embedding


def make_neural_model(model_key: str, data: Any, args: argparse.Namespace) -> nn.Module:
    static_dim = int(data.static.shape[1])
    dynamic_dim = int(data.dynamic.shape[2])
    hidden = int(args.hidden_dim)
    dropout = float(args.dropout)
    if model_key == "lstm":
        return DynamicRNNClassifier(dynamic_dim, hidden, NUM_CLASSES, dropout, cell="lstm")
    if model_key == "gru":
        return DynamicRNNClassifier(dynamic_dim, hidden, NUM_CLASSES, dropout, cell="gru")
    if model_key == "dtdnn":
        return DynamicTDNNClassifier(dynamic_dim, hidden, NUM_CLASSES, dropout)
    if model_key == "tcn":
        return DynamicTCNClassifier(dynamic_dim, hidden, NUM_CLASSES, dropout)
    if model_key in {"lstm_fc", "ctfe_lstm_fc", "lstm_fc_wfocal", "ctfe_lstm_fc_wfocal"}:
        return FusionRNNFCClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout)
    if model_key == "gru_fc":
        return FusionGRUClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout, mode="fc")
    if model_key == "gru_addgate":
        return FusionGRUClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout, mode="addgate")
    if model_key == "gru_mag":
        return FusionGRUClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout, mode="mag")
    if model_key == "tcn_fc":
        return FusionTCNClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout, mode="fc")
    if model_key == "tcn_addgate":
        return FusionTCNClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout, mode="addgate")
    if model_key == "tcn_mag":
        return FusionTCNClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout, mode="mag")
    if model_key in {"lstm_addgate", "ctfe_lstm_addgate"}:
        return FusionRNNAddGateClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout)
    if model_key in {"lstm_mag", "ctfe_lstm_mag"}:
        return FusionRNNMAGClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout)
    if model_key == "dtdnn_fc":
        return FusionTDNNClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout, mode="fc")
    if model_key == "dtdnn_mag":
        return FusionTDNNClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout, mode="mag")
    if model_key == "dtdnn_addgate":
        return FusionTDNNClassifier(static_dim, dynamic_dim, hidden, NUM_CLASSES, dropout, mode="addgate")
    raise ValueError(f"Unsupported neural model key: {model_key}")


def loader(data: Any, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(data.static[mask], dtype=torch.float32),
        torch.tensor(data.dynamic[mask], dtype=torch.float32),
        torch.tensor(data.mask[mask], dtype=torch.float32),
        torch.tensor(data.y[mask], dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weight: torch.Tensor, gamma: float):
        super().__init__()
        self.register_buffer("class_weight", class_weight.float())
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        target_log_prob = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        weight = self.class_weight.gather(0, target)
        loss = -weight * torch.pow(1.0 - target_prob, self.gamma) * target_log_prob
        return loss.mean()


def make_class_balanced_loss(y: np.ndarray, mask: np.ndarray, args: argparse.Namespace, device: torch.device) -> nn.Module:
    counts = np.bincount(y[mask].astype(int), minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = np.power(counts.sum() / (counts * float(NUM_CLASSES)), float(args.class_weight_power))
    weights = weights / max(float(weights.mean()), 1e-8)
    return WeightedFocalLoss(torch.tensor(weights, dtype=torch.float32, device=device), gamma=float(args.focal_gamma))


def make_training_criterion(model_key: str, y: np.ndarray, mask: np.ndarray, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if model_key.endswith("_wfocal"):
        return make_class_balanced_loss(y, mask, args, device)
    return nn.CrossEntropyLoss()


def predict_model(model: nn.Module, data: Any, mask: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    embeddings = []
    with torch.no_grad():
        for static_x, dynamic_x, obs_mask, _y in loader(data, mask, 2048, False):
            logits, embedding = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            raw = logits.detach().cpu().numpy()
            raw = raw - raw.max(axis=1, keepdims=True)
            proba = np.exp(raw)
            proba = proba / proba.sum(axis=1, keepdims=True)
            probs.append(proba)
            embeddings.append(embedding.detach().cpu().numpy())
    return np.vstack(probs), np.vstack(embeddings)


def selection_tuple(metrics: Mapping[str, Any], selection_metric: str) -> tuple[float, float, float]:
    primary = float(metrics.get(selection_metric, -1.0))
    return (primary, float(metrics.get("weighted_f1", -1.0)), float(metrics.get("macro_f1", -1.0)))


def select_prediction_head(
    train_embedding: np.ndarray,
    train_y: np.ndarray,
    valid_embedding: np.ndarray,
    valid_y: np.ndarray,
    heldout_embedding: np.ndarray,
    valid_softmax: np.ndarray,
    heldout_softmax: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates: list[tuple[dict[str, Any], np.ndarray]] = []
    soft_pred = valid_softmax.argmax(axis=1)
    soft_metrics = {"head": "softmax", **metric_payload(valid_y, soft_pred)}
    candidates.append((soft_metrics, heldout_softmax.argmax(axis=1)))
    grid = build_cosine_knn_probability_grid(train_embedding, train_y, valid_embedding, k_grid=[int(k) for k in args.knn_grid])
    heldout_grid_cache: dict[tuple[int, str], np.ndarray] = {}
    for (k, vote_mode), valid_proba in grid.items():
        valid_pred = valid_proba.argmax(axis=1)
        row = {"head": f"cosine_knn_k{int(k)}_{vote_mode}", "k": int(k), "vote_mode": vote_mode, **metric_payload(valid_y, valid_pred)}
        heldout_grid_cache[(int(k), str(vote_mode))] = build_cosine_knn_probability_grid(train_embedding, train_y, heldout_embedding, k_grid=[int(k)])[(int(k), str(vote_mode))].argmax(axis=1)
        candidates.append((row, heldout_grid_cache[(int(k), str(vote_mode))]))
    best_row, heldout_pred = sorted(candidates, key=lambda item: selection_tuple(item[0], args.selection_metric), reverse=True)[0]
    best_row["selection_metric"] = args.selection_metric
    best_row["selection_split"] = "inner_valid"
    best_row["test_used_for_selection"] = False
    return heldout_pred.astype(int), best_row


def run_neural_model(raw: Any, model_key: str, args: argparse.Namespace, out_dir: Path, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    groups = raw.row_index["cycle_uid"].astype(str).to_numpy()
    pred = np.full(len(raw.y), -1, dtype=int)
    fold_rows = []
    curve_rows = []
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold, outer_train_idx, heldout_idx in grouped_oof_splits(groups, raw.y, int(args.oof_folds)):
        outer_train = np.zeros(len(raw.y), dtype=bool)
        heldout = np.zeros(len(raw.y), dtype=bool)
        outer_train[outer_train_idx] = True
        heldout[heldout_idx] = True
        inner_train, inner_valid = phase864._inner_train_valid_masks(groups, outer_train, seed=int(args.seed), fold=int(fold))
        data = phase864.prepare_arrays(raw, fit_mask=inner_train)
        torch.manual_seed(int(args.seed) + int(fold) * 101 + int(MODEL_SEED_OFFSET.get(model_key, 0)))
        np.random.seed(int(args.seed) + int(fold) * 101 + int(MODEL_SEED_OFFSET.get(model_key, 0)))
        model = make_neural_model(model_key, data, args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        criterion = make_training_criterion(model_key, data.y, inner_train, args, device)
        best_state: dict[str, torch.Tensor] | None = None
        best_metric = -float("inf")
        wait = 0
        for epoch in tqdm(range(1, int(args.epochs) + 1), desc=f"phase866 {model_key} fold{fold}"):
            model.train()
            losses: list[float] = []
            for static_x, dynamic_x, obs_mask, y in loader(data, inner_train, int(args.batch_size), True):
                optimizer.zero_grad(set_to_none=True)
                logits, _embedding = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
                loss = criterion(logits, y.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            valid_proba, _valid_embedding = predict_model(model, data, inner_valid, device)
            valid_pred = valid_proba.argmax(axis=1)
            valid_metrics = metric_payload(data.y[inner_valid], valid_pred)
            score = float(valid_metrics[str(args.selection_metric)])
            curve_rows.append({
                "model_key": model_key,
                "fold": int(fold),
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "valid_accuracy": float(valid_metrics["accuracy"]),
                "valid_macro_f1": float(valid_metrics["macro_f1"]),
                "valid_weighted_f1": float(valid_metrics["weighted_f1"]),
            })
            if score > best_metric + 1e-5:
                best_metric = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= int(args.patience):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        train_proba, train_embedding = predict_model(model, data, inner_train, device)
        valid_proba, valid_embedding = predict_model(model, data, inner_valid, device)
        heldout_proba, heldout_embedding = predict_model(model, data, heldout, device)
        heldout_pred, selected = select_prediction_head(
            train_embedding,
            data.y[inner_train],
            valid_embedding,
            data.y[inner_valid],
            heldout_embedding,
            valid_proba,
            heldout_proba,
            args=args,
        )
        pred[heldout] = heldout_pred
        fold_metric = {"model_key": model_key, "fold": int(fold), **metric_payload(data.y[heldout], heldout_pred)}
        fold_metric.update({f"selected_{k}": v for k, v in selected.items() if k not in {"sample_count"}})
        fold_rows.append(fold_metric)
        torch.save({
            "model_key": model_key,
            "fold": int(fold),
            "state_dict": model.state_dict(),
            "static_features": list(raw.static_features),
            "dynamic_features": list(raw.dynamic_features),
            "selected_head": selected,
            "args": vars(args),
        }, model_dir / f"{model_key}_fold{fold}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if (pred < 0).any():
        raise RuntimeError(f"Model {model_key} left {(pred < 0).sum()} OOF rows unpredicted.")
    save_prediction_frame(raw, pred, model_key=model_key, path=out_dir / "predictions" / f"{model_key}_oof_predictions.csv")
    return (
        summarize_model(raw, pred, model_key=model_key, model_name=MODEL_DISPLAY.get(model_key, model_key)),
        pd.DataFrame(fold_rows),
        classwise_rows(raw.y, pred, model_key=model_key),
        confusion_rows(raw.y, pred, model_key=model_key),
        pd.DataFrame(curve_rows),
    )


def expand_model_list(models: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in models:
        if item == "all":
            expanded.extend(DEFAULT_MODELS)
        else:
            expanded.append(item)
    seen: set[str] = set()
    ordered: list[str] = []
    allowed = set(DEFAULT_MODELS)
    for key in expanded:
        if key not in allowed:
            raise ValueError(f"Unknown model key {key}; allowed keys are {sorted(allowed)} or all.")
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def write_failed_logs(out_dir: Path, failures: list[dict[str, Any]]) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "failed_models_log.json", failures)
    lines = []
    for item in failures:
        lines.append(f"[{item.get('model_key')}] {item.get('error_type')}: {item.get('error')}")
        lines.append(str(item.get("traceback", "")))
        lines.append("")
    (log_dir / "failed_models_log.txt").write_text("\n".join(lines), encoding="utf-8")


def write_accumulators(out_dir: Path, metrics: list[dict[str, Any]], folds: list[pd.DataFrame], classwise: list[dict[str, Any]], confusions: list[dict[str, Any]], curves: list[pd.DataFrame]) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(table_dir / "daily_expanded_model_metrics_fsh_raw.csv", index=False)
    if folds:
        pd.concat(folds, ignore_index=True).to_csv(table_dir / "daily_expanded_model_fold_metrics_fsh.csv", index=False)
    else:
        pd.DataFrame().to_csv(table_dir / "daily_expanded_model_fold_metrics_fsh.csv", index=False)
    pd.DataFrame(classwise).to_csv(table_dir / "daily_expanded_model_classwise_metrics_fsh.csv", index=False)
    pd.DataFrame(confusions).to_csv(table_dir / "daily_expanded_model_confusion_matrix_normalized_fsh.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(table_dir / "daily_expanded_neural_training_curves_fsh.csv", index=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or f"phase866_daily_expanded_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_root) / run_id
    paper_table_dir = Path(args.paper_table_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    models = expand_model_list(list(args.models))
    raw, frame, audit = load_daily_arrays(args)
    frame.head(2000).to_csv(out_dir / "tables" / "daily_expanded_training_frame_sample.csv", index=False)
    class_distribution = pd.DataFrame({
        "class": LABELS,
        "sample_count": [int(np.sum(raw.y == i)) for i in range(NUM_CLASSES)],
    })
    class_distribution["sample_fraction"] = class_distribution["sample_count"] / max(1, int(len(raw.y)))
    class_distribution.to_csv(out_dir / "tables" / "class_distribution_fsh.csv", index=False)
    ctfe_metric = load_ctfe_final_metrics(args)
    metrics: list[dict[str, Any]] = [ctfe_metric]
    fold_frames: list[pd.DataFrame] = []
    classwise_all: list[dict[str, Any]] = []
    confusion_all: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    metrics_by_key: dict[str, dict[str, Any]] = {"ctfe_addgate_final": ctfe_metric}
    render_tables(metrics_by_key, paper_table_dir)
    render_tables(metrics_by_key, out_dir / "tables")
    config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(device),
        "models": models,
        "args": vars(args),
        "data_audit": audit,
        "ctfe_final_metric": ctfe_metric,
        "note": "Comparator models are trained on the same daily-expanded FSH absolute-dose category task as the final CTFE(AddGate) row.",
    }
    write_json(out_dir / "logs" / "experiment_config.json", config)
    for model_key in models:
        try:
            print(f"[phase866] running {model_key}", flush=True)
            if model_key == "majority":
                summary, fold_df, cls_rows, cm_rows = run_majority_baseline(raw, args, out_dir)
                curve_df = pd.DataFrame()
            elif model_key == "previous":
                summary, fold_df, cls_rows, cm_rows = run_previous_dose_baseline(raw, args, out_dir)
                curve_df = pd.DataFrame()
            elif model_key == "lasso":
                summary, fold_df, cls_rows, cm_rows = run_lasso_baseline(raw, args, out_dir)
                curve_df = pd.DataFrame()
            else:
                summary, fold_df, cls_rows, cm_rows, curve_df = run_neural_model(raw, model_key, args, out_dir, device)
            metrics.append(summary)
            metrics_by_key[model_key] = summary
            fold_frames.append(fold_df)
            classwise_all.extend(cls_rows)
            confusion_all.extend(cm_rows)
            if not curve_df.empty:
                curve_frames.append(curve_df)
            write_accumulators(out_dir, metrics, fold_frames, classwise_all, confusion_all, curve_frames)
            render_tables(metrics_by_key, paper_table_dir)
            render_tables(metrics_by_key, out_dir / "tables")
            print(f"[phase866] completed {model_key}: acc={summary['accuracy']:.4f}, macro_f1={summary['macro_f1']:.4f}, weighted_f1={summary['weighted_f1']:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failure = {
                "model_key": model_key,
                "model": MODEL_DISPLAY.get(model_key, model_key),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            write_failed_logs(out_dir, failures)
            render_tables(metrics_by_key, paper_table_dir)
            render_tables(metrics_by_key, out_dir / "tables")
            print(f"[phase866] FAILED {model_key}: {type(exc).__name__}: {exc}", flush=True)
            if args.fail_fast:
                raise
    write_failed_logs(out_dir, failures)
    write_accumulators(out_dir, metrics, fold_frames, classwise_all, confusion_all, curve_frames)
    table1, table2 = render_tables(metrics_by_key, paper_table_dir)
    render_tables(metrics_by_key, out_dir / "tables")
    result = {
        "run_id": run_id,
        "artifact_dir": str(out_dir),
        "paper_table_dir": str(paper_table_dir),
        "models_requested": models,
        "models_completed": sorted([key for key in metrics_by_key if key != "ctfe_addgate_final"]),
        "failed_models": failures,
        "table1": table1.to_dict(orient="records"),
        "table2": table2.to_dict(orient="records"),
    }
    write_json(out_dir / "logs" / "run_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
