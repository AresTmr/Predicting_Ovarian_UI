from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from models.layer1_strategy.action_model import (
    FUTURE_PREFIXES,
    ID_COLUMNS,
    LAYER1_ACTION_LEAKAGE_COLUMNS,
    augment_layer1_action_features,
)

FSH_DOSE_LABELS = ["stop", "low_dose", "medium_low", "medium_high", "high_dose"]
GRID_DOSE_VALUES = [0, 75, 150, 225, 300, 375]
GRID_DOSE_LABELS = [f"dose_{value}" for value in GRID_DOSE_VALUES]

ABSOLUTE_DOSE_TARGETS = {
    "fsh": "next_fsh_dose_bin",
    "lh": "next_lh_dose_bin",
    "hmg": "next_hmg_dose_bin",
}

ABSOLUTE_DOSE_LABELS = {
    "fsh": FSH_DOSE_LABELS,
    "lh": GRID_DOSE_LABELS,
    "hmg": GRID_DOSE_LABELS,
}

ABSOLUTE_DOSE_SOURCE_COLUMNS = {
    "fsh": "next_fsh_daily_dose",
    "lh": "next_lh_daily_dose",
    "hmg": "next_hmg_daily_dose",
}

ABSOLUTE_DOSE_LEAKAGE_COLUMNS = set(ABSOLUTE_DOSE_TARGETS.values()) | {
    f"pred_{target}" for target in ABSOLUTE_DOSE_TARGETS.values()
}


@dataclass(frozen=True)
class AbsoluteDoseLabelAudit:
    total_rows: int
    complete_case_rows: int
    off_grid_counts: dict[str, int]
    missing_counts: dict[str, int]


def fsh_absolute_dose_label(value: Any) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    dose = float(value)
    if dose <= 0:
        return "stop"
    if dose < 80:
        return "low_dose"
    if dose < 160:
        return "medium_low"
    if dose <= 240:
        return "medium_high"
    return "high_dose"


def grid_absolute_dose_label(value: Any, *, allowed: Sequence[int] = GRID_DOSE_VALUES, tolerance: float = 1e-6) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    dose = float(value)
    for allowed_value in allowed:
        if abs(dose - float(allowed_value)) <= tolerance:
            return f"dose_{allowed_value}"
    return pd.NA


def add_absolute_dose_labels(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for drug, source_col in ABSOLUTE_DOSE_SOURCE_COLUMNS.items():
        if source_col not in df.columns:
            raise ValueError(f"Missing required next-dose column for {drug}: {source_col}")
    df["next_fsh_dose_bin"] = df["next_fsh_daily_dose"].map(fsh_absolute_dose_label)
    df["next_lh_dose_bin"] = df["next_lh_daily_dose"].map(grid_absolute_dose_label)
    df["next_hmg_dose_bin"] = df["next_hmg_daily_dose"].map(grid_absolute_dose_label)
    return df


def absolute_dose_label_audit(frame: pd.DataFrame) -> AbsoluteDoseLabelAudit:
    labeled = add_absolute_dose_labels(frame)
    off_grid_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    for drug, target in ABSOLUTE_DOSE_TARGETS.items():
        source_col = ABSOLUTE_DOSE_SOURCE_COLUMNS[drug]
        source = pd.to_numeric(labeled[source_col], errors="coerce")
        missing_counts[drug] = int(source.isna().sum())
        off_grid_counts[drug] = int(source.notna().sum() - labeled[target].notna().sum())
    complete_mask = np.logical_and.reduce([labeled[target].notna().to_numpy() for target in ABSOLUTE_DOSE_TARGETS.values()])
    return AbsoluteDoseLabelAudit(
        total_rows=int(len(labeled)),
        complete_case_rows=int(complete_mask.sum()),
        off_grid_counts=off_grid_counts,
        missing_counts=missing_counts,
    )


def absolute_dose_complete_case(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = add_absolute_dose_labels(frame)
    mask = np.logical_and.reduce([labeled[target].notna().to_numpy() for target in ABSOLUTE_DOSE_TARGETS.values()])
    return labeled.loc[mask].copy()


def select_absolute_dose_feature_columns(frame: pd.DataFrame, targets: Iterable[str] = ABSOLUTE_DOSE_TARGETS.values()) -> list[str]:
    augmented = augment_layer1_action_features(frame)
    excluded = set(ID_COLUMNS) | set(LAYER1_ACTION_LEAKAGE_COLUMNS) | set(targets) | ABSOLUTE_DOSE_LEAKAGE_COLUMNS
    feature_columns: list[str] = []
    for column in augmented.columns:
        if column in excluded:
            continue
        if any(str(column).startswith(prefix) for prefix in FUTURE_PREFIXES):
            continue
        if str(column).endswith("_action"):
            continue
        if str(column).endswith("_dose_bin"):
            continue
        feature_columns.append(column)
    return feature_columns


def evaluation_day_from_gn_day(gn_day: pd.Series) -> pd.Series:
    values = pd.to_numeric(gn_day, errors="coerce")
    return (values + 1).round().astype("Int64")


def evaluate_absolute_dose_predictions(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    labels: Sequence[str],
) -> dict[str, Any]:
    truth = pd.Series(y_true, dtype="object")
    pred = pd.Series(y_pred, dtype="object")
    report = classification_report(truth, pred, labels=list(labels), output_dict=True, zero_division=0)
    metrics: dict[str, Any] = {
        "sample_count": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, labels=list(labels), average="weighted", zero_division=0)),
    }
    for label in labels:
        payload = report.get(label, {})
        metrics[f"precision_{label}"] = float(payload.get("precision", 0.0))
        metrics[f"recall_{label}"] = float(payload.get("recall", 0.0))
        metrics[f"f1_{label}"] = float(payload.get("f1-score", 0.0))
        metrics[f"support_{label}"] = int(payload.get("support", 0))
    return metrics


def kong_style_daily_accuracy_table(
    predictions: pd.DataFrame,
    *,
    day_col: str = "evaluation_day",
    min_day: int = 1,
    max_day: int = 19,
    at_risk_counts: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = predictions.copy()
    frame[day_col] = pd.to_numeric(frame[day_col], errors="coerce")
    frame = frame[frame[day_col].between(min_day, max_day, inclusive="both")].copy()
    if at_risk_counts is None:
        days = sorted(int(day) for day in frame[day_col].dropna().unique())
    else:
        days = list(range(int(min_day), int(max_day) + 1))
    for day in days:
        subset = frame[frame[day_col].eq(day)]
        daily_count = int(at_risk_counts.get(day, 0)) if at_risk_counts is not None else int(len(subset))
        if subset.empty:
            rows.append(
                {
                    "Day": int(day),
                    "Daily count": daily_count,
                    "FSH": np.nan,
                    "LH": np.nan,
                    "HMG": np.nan,
                    "Three-drug exact": np.nan,
                }
            )
            continue
        rows.append(
            {
                "Day": int(day),
                "Daily count": daily_count,
                "FSH": float((subset["truth_fsh"].astype(str) == subset["pred_fsh"].astype(str)).mean()),
                "LH": float((subset["truth_lh"].astype(str) == subset["pred_lh"].astype(str)).mean()),
                "HMG": float((subset["truth_hmg"].astype(str) == subset["pred_hmg"].astype(str)).mean()),
                "Three-drug exact": float(
                    (
                        (subset["truth_fsh"].astype(str) == subset["pred_fsh"].astype(str))
                        & (subset["truth_lh"].astype(str) == subset["pred_lh"].astype(str))
                        & (subset["truth_hmg"].astype(str) == subset["pred_hmg"].astype(str))
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows, columns=["Day", "Daily count", "FSH", "LH", "HMG", "Three-drug exact"])


def daily_at_risk_cycle_counts(
    frame: pd.DataFrame,
    *,
    day_col: str = "evaluation_day",
    cycle_col: str = "cycle_uid",
    min_day: int = 1,
    max_day: int = 19,
) -> dict[int, int]:
    if cycle_col not in frame.columns:
        raise ValueError(f"Missing cycle identifier column: {cycle_col}")
    if day_col not in frame.columns:
        raise ValueError(f"Missing day column: {day_col}")
    working = frame[[cycle_col, day_col]].copy()
    working[day_col] = pd.to_numeric(working[day_col], errors="coerce")
    max_day_by_cycle = working.dropna(subset=[day_col]).groupby(cycle_col)[day_col].max()
    return {
        int(day): int(max_day_by_cycle.ge(day).sum())
        for day in range(int(min_day), int(max_day) + 1)
    }


def label_display_map() -> Mapping[str, Mapping[str, str]]:
    return {
        "fsh": {
            "stop": "0 / stop",
            "low_dose": "<80 IU",
            "medium_low": "80-160 IU",
            "medium_high": "160-240 IU",
            "high_dose": ">240 IU",
        },
        "lh": {f"dose_{value}": f"{value} IU" for value in GRID_DOSE_VALUES},
        "hmg": {f"dose_{value}": f"{value} IU" for value in GRID_DOSE_VALUES},
    }
