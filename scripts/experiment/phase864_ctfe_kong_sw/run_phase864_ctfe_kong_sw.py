#!/usr/bin/env python3
"""Phase 8.64 Kong-style CTFE daily table with sliding-window routing.

This runner uses the existing no-leakage daily panel, trains a Kong-aligned
D-TDNN/AddGate neural encoder with a cosine-KNN head, and reports cycle-grouped
OOF daily accuracy for CTFE and CTFE-sw.  The sliding-window head is trained and
applied only from Day 13 onward.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.layer1_strategy.ctfe_auxiliary import CTFE_STATIC_FEATURES
from models.layer1_strategy.ctfe_kong_aligned import (
    DEFAULT_KNN_GRID,
    KONG_SEMANTIC_LABELS,
    KongAlignedCTFENetwork,
    build_cosine_knn_probability_grid,
    compose_ctfe_sliding_window_predictions,
    ctfe_daily_accuracy_table,
)
from models.layer1_strategy.kong_daily_no_sw import (
    build_daily_no_sw_panel,
    daily_no_sw_label_audit,
    select_daily_no_sw_feature_columns,
)

LABEL_TO_ID = {label: idx for idx, label in enumerate(KONG_SEMANTIC_LABELS)}
DAILY_CTFE_EXTRA_LEAKAGE_COLUMNS = {
    "current_fsh_share_gn",
    "visit_gn_total_amount",
    "total_medication_count",
    "pure_fsh_medication_count",
    "lh_medication_count",
    "hmg_medication_count",
    "lh_like_hmg_medication_count",
    "stimulation_gn_medication_count",
    "gn_per_total_follicle",
    "gn_per_mature_follicle",
    "follicle_per_1000iu_gn",
    "mature_follicle_per_1000iu_gn",
}


@dataclass
class DailySequenceArrays:
    static: np.ndarray
    dynamic: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    row_index: pd.DataFrame
    static_features: list[str]
    dynamic_features: list[str]


@dataclass
class PreparedDailyArrays:
    static: np.ndarray
    dynamic: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    row_index: pd.DataFrame
    static_features: list[str]
    dynamic_features: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 8.64 CTFE and CTFE-sw daily Kong-style table.")
    parser.add_argument("--input", default="data/processed/layer1_strategy_dataset.csv")
    parser.add_argument("--baseline-input", default="data/processed/baseline_cycle_dataset.csv")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default="artifacts/experiments/phase864_ctfe_kong_sw")
    parser.add_argument("--paper-table-dir", default="paper/tables/paper_claim_validation/phase864_ctfe_kong_sw")
    parser.add_argument("--paper-figure-dir", default="paper/figures/paper_claim_validation/phase864_ctfe_kong_sw")
    parser.add_argument("--min-day", type=int, default=1)
    parser.add_argument("--max-day", type=int, default=20)
    parser.add_argument("--sw-start-day", type=int, default=13)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--max-history-days", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--knn-grid", type=int, nargs="+", default=list(DEFAULT_KNN_GRID))
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _numeric_feature_columns(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])]


def prepare_daily_ctfe_frame(
    monitoring_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame | None = None,
    *,
    min_day: int = 1,
    max_day: int = 20,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    """Create a no-leakage daily CTFE frame from the daily no-SW panel."""

    panel = build_daily_no_sw_panel(monitoring_frame, baseline_frame, min_day=min_day, max_day=max_day)
    label_audit = daily_no_sw_label_audit(panel, day_col="Day")
    panel = panel[panel["daily_fsh_dose_bin"].isin(KONG_SEMANTIC_LABELS)].copy()
    if panel.empty:
        raise ValueError("No FSH-evaluable daily CTFE rows after panel construction.")

    panel["ctfe_truth"] = panel["daily_fsh_dose_bin"].astype(str)
    panel["ctfe_truth_id"] = panel["ctfe_truth"].map(LABEL_TO_ID).astype(int)
    panel["next_fsh_daily_dose"] = pd.to_numeric(panel["daily_fsh_daily_dose"], errors="coerce")
    panel["source_visit_uid"] = panel.get("visit_uid", pd.Series([pd.NA] * len(panel), index=panel.index)).astype(str)
    panel["Day"] = pd.to_numeric(panel["Day"], errors="coerce").astype(int)
    panel["evaluation_day"] = panel["Day"]
    panel["monitoring_order"] = panel["Day"]
    panel["gn_day"] = panel["Day"].astype(float)
    panel["visit_uid"] = panel["cycle_uid"].astype(str) + "__daily_day" + panel["Day"].astype(str)
    panel["baseline_only_feature_row"] = panel["baseline_only_feature_row"].astype(bool).astype(int)

    feature_candidates = [
        column
        for column in select_daily_no_sw_feature_columns(panel, targets=["daily_fsh_dose_bin", "ctfe_truth", "ctfe_truth_id"])
        if column not in DAILY_CTFE_EXTRA_LEAKAGE_COLUMNS
    ]
    for column in feature_candidates:
        if column in panel.columns and not pd.api.types.is_numeric_dtype(panel[column]):
            converted = pd.to_numeric(panel[column], errors="coerce")
            if converted.notna().any():
                panel[column] = converted
    numeric_features = _numeric_feature_columns(panel, feature_candidates)
    static_features = [column for column in CTFE_STATIC_FEATURES if column in numeric_features]
    if not static_features:
        panel["static_intercept"] = 1.0
        static_features = ["static_intercept"]
        numeric_features.append("static_intercept")
    dynamic_features = [column for column in numeric_features if column not in static_features]
    if "Day" in numeric_features:
        dynamic_features = ["Day"] + [column for column in dynamic_features if column != "Day"]
    if not dynamic_features:
        raise ValueError("No numeric dynamic features available for daily CTFE.")

    day_values = pd.to_numeric(panel["Day"], errors="coerce")
    audit = {
        "min_day": int(min_day),
        "max_day": int(max_day),
        "panel_rows_before_fsh_filter": int(label_audit.total_rows),
        "panel_rows_after_fsh_filter": int(len(panel)),
        "panel_cycles_after_fsh_filter": int(panel["cycle_uid"].nunique()),
        "daily_counts_before_fsh_filter": label_audit.daily_counts,
        "daily_counts_after_fsh_filter": {int(day): int(count) for day, count in day_values.value_counts().sort_index().items()},
        "static_feature_count": int(len(static_features)),
        "dynamic_feature_count": int(len(dynamic_features)),
        "static_features": static_features,
        "dynamic_features": dynamic_features,
        "leakage_excluded_same_day_dose_columns": [
            "current_fsh_daily_dose",
            "current_lh_daily_dose",
            "current_hmg_daily_dose",
            "daily_fsh_daily_dose",
            "daily_lh_daily_dose",
            "daily_hmg_daily_dose",
            "daily_fsh_dose_bin",
            *sorted(DAILY_CTFE_EXTRA_LEAKAGE_COLUMNS),
        ],
    }
    return panel.sort_values(["cycle_uid", "Day", "visit_uid"], kind="mergesort").reset_index(drop=True), {"static_features": static_features, "dynamic_features": dynamic_features}, audit


def _safe_numeric(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column, np.nan)]), errors="coerce").iloc[0]
    return float(value) if not pd.isna(value) else np.nan


def build_daily_sequence_arrays(
    frame: pd.DataFrame,
    *,
    static_features: list[str],
    dynamic_features: list[str],
    max_history_days: int,
) -> DailySequenceArrays:
    rows_static: list[list[float]] = []
    rows_dynamic: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    row_records: list[dict[str, Any]] = []
    max_history_days = int(max_history_days)
    ordered = frame.sort_values(["cycle_uid", "Day", "visit_uid"], kind="mergesort").reset_index(drop=True)
    for _, group in ordered.groupby("cycle_uid", sort=False):
        group = group.sort_values(["Day", "visit_uid"], kind="mergesort").reset_index(drop=True)
        static_matrix = group[static_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        dynamic_matrix = group[dynamic_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        day_values = pd.to_numeric(group["Day"], errors="coerce").to_numpy(dtype=int)
        label_values = pd.to_numeric(group["ctfe_truth_id"], errors="coerce").to_numpy(dtype=int)
        for pos, query in group.iterrows():
            start_pos = max(0, int(pos) + 1 - max_history_days)
            history = dynamic_matrix[start_pos : int(pos) + 1]
            sequence = np.full((max_history_days, len(dynamic_features)), np.nan, dtype=float)
            mask = np.zeros(max_history_days, dtype=float)
            sequence[: len(history), :] = history
            mask[: len(history)] = 1.0
            rows_static.append(static_matrix[int(pos), :].tolist())
            rows_dynamic.append(sequence)
            masks.append(mask)
            labels.append(int(label_values[int(pos)]))
            day = int(day_values[int(pos)])
            row_records.append(
                {
                    "visit_uid": query.get("visit_uid", ""),
                    "source_visit_uid": query.get("source_visit_uid", ""),
                    "cycle_uid": query.get("cycle_uid", ""),
                    "art_id": query.get("art_id", ""),
                    "Day": day,
                    "gn_day": float(query.get("gn_day", day)),
                    "monitoring_order": int(query.get("monitoring_order", day)),
                    "ctfe_truth": query.get("ctfe_truth", ""),
                    "daily_fsh_daily_dose": query.get("daily_fsh_daily_dose", np.nan),
                    "baseline_only_feature_row": int(query.get("baseline_only_feature_row", 0)),
                }
            )
    return DailySequenceArrays(
        static=np.asarray(rows_static, dtype=float),
        dynamic=np.asarray(rows_dynamic, dtype=float),
        mask=np.asarray(masks, dtype=float),
        y=np.asarray(labels, dtype=int),
        row_index=pd.DataFrame(row_records),
        static_features=static_features,
        dynamic_features=dynamic_features,
    )

def prepare_arrays(raw: DailySequenceArrays, *, fit_mask: np.ndarray) -> PreparedDailyArrays:
    if not bool(fit_mask.any()):
        raise ValueError("Cannot fit daily CTFE preprocessors without training rows.")
    static_imputer = SimpleImputer(strategy="median").fit(raw.static[fit_mask])
    static_scaler = StandardScaler().fit(static_imputer.transform(raw.static[fit_mask]))
    static = static_scaler.transform(static_imputer.transform(raw.static)).astype(np.float32)

    observed = raw.dynamic[fit_mask][raw.mask[fit_mask].astype(bool)]
    dynamic_imputer = SimpleImputer(strategy="median").fit(observed)
    dynamic_scaler = StandardScaler().fit(dynamic_imputer.transform(observed))
    flat = raw.dynamic.reshape(-1, raw.dynamic.shape[-1])
    dynamic = dynamic_scaler.transform(dynamic_imputer.transform(flat)).reshape(raw.dynamic.shape).astype(np.float32)
    dynamic = dynamic * raw.mask[:, :, None].astype(np.float32)
    return PreparedDailyArrays(static, dynamic, raw.mask.astype(np.float32), raw.y, raw.row_index.copy(), raw.static_features, raw.dynamic_features)


def _loader(data: PreparedDailyArrays, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_data = TensorDataset(
        torch.tensor(data.static[mask], dtype=torch.float32),
        torch.tensor(data.dynamic[mask], dtype=torch.float32),
        torch.tensor(data.mask[mask], dtype=torch.float32),
        torch.tensor(data.y[mask], dtype=torch.long),
    )
    return DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)


def _predict(model: nn.Module, data: PreparedDailyArrays, mask: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for static_x, dynamic_x, obs_mask, _ in _loader(data, mask, 1024, False):
            logits, embedding = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            raw = logits.cpu().numpy()
            raw = raw - raw.max(axis=1, keepdims=True)
            proba = np.exp(raw)
            proba = proba / proba.sum(axis=1, keepdims=True)
            probabilities.append(proba)
            embeddings.append(embedding.cpu().numpy())
    return np.vstack(probabilities), np.vstack(embeddings)


def _metric_payload(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(KONG_SEMANTIC_LABELS))), average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(len(KONG_SEMANTIC_LABELS))), average="weighted", zero_division=0)) if len(y_true) else float("nan"),
    }


def _select_knn(
    train_embedding: np.ndarray,
    train_y: np.ndarray,
    valid_embedding: np.ndarray,
    valid_y: np.ndarray,
    *,
    k_grid: Iterable[int],
) -> tuple[dict[str, Any], np.ndarray]:
    grid = build_cosine_knn_probability_grid(train_embedding, train_y, valid_embedding, k_grid=k_grid)
    rows: list[dict[str, Any]] = []
    for (k, vote_mode), probability in grid.items():
        pred = probability.argmax(axis=1)
        metrics = _metric_payload(valid_y, pred)
        rows.append({"k": int(k), "vote_mode": str(vote_mode), **metrics})
    selected = sorted(rows, key=lambda item: (item["accuracy"], item["weighted_f1"], item["macro_f1"], -item["k"]), reverse=True)[0]
    selected_probability = grid[(int(selected["k"]), str(selected["vote_mode"]))]
    selected["head"] = f"cosine_knn_k{int(selected['k'])}_{selected['vote_mode']}"
    return selected, selected_probability


def _query_knn(train_embedding: np.ndarray, train_y: np.ndarray, query_embedding: np.ndarray, *, selected: Mapping[str, Any]) -> np.ndarray:
    grid = build_cosine_knn_probability_grid(train_embedding, train_y, query_embedding, k_grid=[int(selected["k"])])
    return grid[(int(selected["k"]), str(selected["vote_mode"]))]


def train_predict_scope(
    raw: DailySequenceArrays,
    *,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    query_mask: np.ndarray,
    scope: str,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not bool(train_mask.any()) or not bool(valid_mask.any()) or not bool(query_mask.any()):
        return None
    torch.manual_seed(int(args.seed) + int(fold) * 17 + (13 if scope == "ctfe_sw_window" else 0))
    np.random.seed(int(args.seed) + int(fold) * 17 + (13 if scope == "ctfe_sw_window" else 0))
    data = prepare_arrays(raw, fit_mask=train_mask)
    model = KongAlignedCTFENetwork(
        data.static.shape[1],
        data.dynamic.shape[2],
        hidden_dim=int(args.hidden_dim),
        num_classes=len(KONG_SEMANTIC_LABELS),
        dropout=float(args.dropout),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    wait = 0
    curves: list[dict[str, Any]] = []
    for epoch in tqdm(range(1, int(args.epochs) + 1), desc=f"phase864 {scope} fold{fold}"):
        model.train()
        losses: list[float] = []
        for static_x, dynamic_x, obs_mask, y in _loader(data, train_mask, int(args.batch_size), True):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_proba, _ = _predict(model, data, valid_mask, device)
        valid_pred = valid_proba.argmax(axis=1)
        valid_score = float(accuracy_score(data.y[valid_mask], valid_pred))
        curves.append({"fold": int(fold), "scope": scope, "epoch": int(epoch), "train_loss": float(np.mean(losses)), "valid_accuracy": valid_score})
        if valid_score > best_score + 1e-4:
            best_score = valid_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(args.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    train_proba, train_embedding = _predict(model, data, train_mask, device)
    valid_proba, valid_embedding = _predict(model, data, valid_mask, device)
    query_proba, query_embedding = _predict(model, data, query_mask, device)
    selected, valid_knn = _select_knn(train_embedding, data.y[train_mask], valid_embedding, data.y[valid_mask], k_grid=args.knn_grid)
    query_knn = _query_knn(train_embedding, data.y[train_mask], query_embedding, selected=selected)
    valid_pred = valid_knn.argmax(axis=1)
    query_pred = query_knn.argmax(axis=1)
    metrics = [
        {"fold": int(fold), "scope": scope, "split": "inner_valid", "head": selected["head"], **_metric_payload(data.y[valid_mask], valid_pred)},
        {"fold": int(fold), "scope": scope, "split": "oof_heldout", "head": selected["head"], **_metric_payload(data.y[query_mask], query_pred)},
        {"fold": int(fold), "scope": scope, "split": "softmax_heldout", "head": "softmax", **_metric_payload(data.y[query_mask], query_proba.argmax(axis=1))},
    ]
    return query_pred, metrics, curves


def _inner_train_valid_masks(groups: np.ndarray, outer_train_mask: np.ndarray, *, seed: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    train_indices = np.flatnonzero(outer_train_mask)
    if len(np.unique(groups[train_indices])) < 3:
        return outer_train_mask.copy(), outer_train_mask.copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=int(seed) + int(fold))
    rel_train, rel_valid = next(splitter.split(train_indices, groups=groups[train_indices]))
    inner_train = np.zeros(len(groups), dtype=bool)
    inner_valid = np.zeros(len(groups), dtype=bool)
    inner_train[train_indices[rel_train]] = True
    inner_valid[train_indices[rel_valid]] = True
    return inner_train, inner_valid


def build_oof_predictions(raw: DailySequenceArrays, args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = raw.row_index["cycle_uid"].astype(str).to_numpy()
    n_splits = min(int(args.oof_folds), len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    ctfe_pred = np.full(len(raw.y), pd.NA, dtype=object)
    sw_window_pred = np.full(len(raw.y), pd.NA, dtype=object)
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    day = pd.to_numeric(raw.row_index["Day"], errors="coerce").to_numpy(dtype=float)
    for fold, (outer_train_idx, heldout_idx) in enumerate(splitter.split(np.zeros(len(raw.y)), raw.y, groups=groups), start=1):
        outer_train_mask = np.zeros(len(raw.y), dtype=bool)
        heldout_mask = np.zeros(len(raw.y), dtype=bool)
        outer_train_mask[outer_train_idx] = True
        heldout_mask[heldout_idx] = True
        inner_train_mask, inner_valid_mask = _inner_train_valid_masks(groups, outer_train_mask, seed=int(args.seed), fold=fold)

        all_result = train_predict_scope(
            raw,
            train_mask=inner_train_mask,
            valid_mask=inner_valid_mask,
            query_mask=heldout_mask,
            scope="ctfe_all_days",
            fold=fold,
            args=args,
            device=device,
        )
        if all_result is None:
            raise RuntimeError(f"Fold {fold} failed to train all-day CTFE model.")
        all_query_pred, all_metrics, all_curves = all_result
        ctfe_pred[heldout_mask] = [KONG_SEMANTIC_LABELS[int(value)] for value in all_query_pred]
        metrics.extend(all_metrics)
        curves.extend(all_curves)

        late_mask = day >= float(args.sw_start_day)
        sw_result = train_predict_scope(
            raw,
            train_mask=inner_train_mask & late_mask,
            valid_mask=inner_valid_mask & late_mask,
            query_mask=heldout_mask & late_mask,
            scope="ctfe_sw_window",
            fold=fold,
            args=args,
            device=device,
        )
        if sw_result is not None:
            sw_query_pred, sw_metrics, sw_curves = sw_result
            sw_window_pred[heldout_mask & late_mask] = [KONG_SEMANTIC_LABELS[int(value)] for value in sw_query_pred]
            metrics.extend(sw_metrics)
            curves.extend(sw_curves)

    predictions = raw.row_index.copy()
    predictions["truth"] = predictions["ctfe_truth"].astype(str)
    predictions["ctfe_prediction"] = ctfe_pred
    predictions["sliding_window_prediction"] = sw_window_pred
    predictions["ctfe_sw_prediction"] = compose_ctfe_sliding_window_predictions(
        predictions,
        ctfe_col="ctfe_prediction",
        sliding_window_col="sliding_window_prediction",
        sw_start_day=int(args.sw_start_day),
    )
    return predictions, pd.DataFrame(metrics), pd.DataFrame(curves)


def markdown_table(table: pd.DataFrame) -> str:
    display = table.copy()
    for column in ["CTFE", "CTFE-sw"]:
        display[column] = display[column].map(lambda value: "NA" if pd.isna(value) else f"{float(value):.3f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(headers[index]) for index in range(len(headers))]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    return "\n".join([fmt(headers), "| " + " | ".join("-" * width for width in widths) + " |", *[fmt(row) for row in rows]])


def render_table(table: pd.DataFrame, output_png: Path, output_pdf: Path, *, note: str) -> None:
    import matplotlib.pyplot as plt

    display = table.copy()
    for column in ["CTFE", "CTFE-sw"]:
        display[column] = display[column].map(lambda value: "NA" if pd.isna(value) else f"{float(value):.3f}")
    display["Day"] = display["Day"].astype(str)
    display["Daily count"] = display["Daily count"].astype(str)
    height = max(3.2, 0.34 * (len(display) + 2))
    fig, ax = plt.subplots(figsize=(6.2, height))
    ax.axis("off")
    ax.text(0.0, 1.04, "Table 3. Daily accuracy results for the prediction", transform=ax.transAxes, ha="left", va="bottom", fontsize=10.0, fontweight="bold")
    mpl_table = ax.table(cellText=display.values, colLabels=display.columns, loc="upper left", cellLoc="center", colLoc="center", bbox=[0.0, 0.08, 1.0, 0.90])
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8.0)
    for (row, _col), cell in mpl_table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f4f7fb")
        else:
            cell.set_facecolor("#ffffff")
    ax.text(0.0, 0.015, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8, color="#334155")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or f"phase864_ctfe_kong_sw_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_id
    table_dir = Path(args.paper_table_dir)
    figure_dir = Path(args.paper_figure_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    monitoring = pd.read_csv(args.input, low_memory=False)
    baseline = pd.read_csv(args.baseline_input, low_memory=False) if args.baseline_input else None
    frame, feature_groups, audit = prepare_daily_ctfe_frame(monitoring, baseline, min_day=int(args.min_day), max_day=int(args.max_day))
    raw = build_daily_sequence_arrays(frame, static_features=feature_groups["static_features"], dynamic_features=feature_groups["dynamic_features"], max_history_days=int(args.max_history_days))
    predictions, metrics, curves = build_oof_predictions(raw, args, device)
    at_risk_counts = {int(day): int(count) for day, count in predictions["Day"].value_counts().sort_index().items()}
    table = ctfe_daily_accuracy_table(predictions, min_day=int(args.min_day), max_day=int(args.max_day), at_risk_counts=at_risk_counts)

    audit.update(
        {
            "run_id": run_id,
            "device": str(device),
            "oof_folds": int(args.oof_folds),
            "sw_start_day": int(args.sw_start_day),
            "max_history_days": int(args.max_history_days),
            "model": "KongAlignedCTFENetwork_DTDNN_AddGate_cosineKNN",
            "prediction_rows": int(len(predictions)),
            "prediction_cycles": int(predictions["cycle_uid"].nunique()),
        }
    )

    frame.head(2000).to_csv(run_dir / "phase864_daily_ctfe_frame_sample.csv", index=False)
    predictions.to_csv(run_dir / "phase864_ctfe_kong_sw_oof_predictions.csv", index=False)
    metrics.to_csv(run_dir / "phase864_ctfe_kong_sw_fold_metrics.csv", index=False)
    curves.to_csv(run_dir / "phase864_ctfe_kong_sw_training_curves.csv", index=False)
    table.to_csv(run_dir / "table3_ctfe_kong_sw_daily_accuracy.csv", index=False)
    table.to_csv(table_dir / "table3_ctfe_kong_sw_daily_accuracy.csv", index=False)
    md = "# Table 3. Daily accuracy results for the prediction\n\n" + markdown_table(table) + "\n"
    (run_dir / "table3_ctfe_kong_sw_daily_accuracy.md").write_text(md, encoding="utf-8")
    (table_dir / "table3_ctfe_kong_sw_daily_accuracy.md").write_text(md, encoding="utf-8")
    note = "The sliding window (sw) starts to work from day 13. Accuracy uses cycle-grouped OOF predictions from the no-leakage daily CTFE panel."
    render_table(table, figure_dir / "table3_ctfe_kong_sw_daily_accuracy.png", figure_dir / "table3_ctfe_kong_sw_daily_accuracy.pdf", note=note)
    render_table(table, run_dir / "table3_ctfe_kong_sw_daily_accuracy.png", run_dir / "table3_ctfe_kong_sw_daily_accuracy.pdf", note=note)
    write_json(run_dir / "data_model_audit.json", audit)
    write_json(
        run_dir / "model_manifest.json",
        {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "platform": platform.platform(),
            "input": args.input,
            "baseline_input": args.baseline_input,
            "artifact_dir": str(run_dir),
            "table_csv": str(table_dir / "table3_ctfe_kong_sw_daily_accuracy.csv"),
            "table_markdown": str(table_dir / "table3_ctfe_kong_sw_daily_accuracy.md"),
            "table_png": str(figure_dir / "table3_ctfe_kong_sw_daily_accuracy.png"),
            "table_pdf": str(figure_dir / "table3_ctfe_kong_sw_daily_accuracy.pdf"),
            "data_audit": str(run_dir / "data_model_audit.json"),
            "fold_metrics": str(run_dir / "phase864_ctfe_kong_sw_fold_metrics.csv"),
            "oof_predictions": str(run_dir / "phase864_ctfe_kong_sw_oof_predictions.csv"),
        },
    )
    result = {"run_id": run_id, "artifact_dir": str(run_dir), "table_rows": table.to_dict(orient="records"), "table_csv": str(table_dir / "table3_ctfe_kong_sw_daily_accuracy.csv"), "table_png": str(figure_dir / "table3_ctfe_kong_sw_daily_accuracy.png"), "audit": audit}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
