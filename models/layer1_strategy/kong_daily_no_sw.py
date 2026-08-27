from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from models.layer1_strategy.absolute_dose_heads import (
    ABSOLUTE_DOSE_LABELS,
    FSH_DOSE_LABELS,
    GRID_DOSE_LABELS,
    fsh_absolute_dose_label,
    grid_absolute_dose_label,
    kong_style_daily_accuracy_table,
)
from models.layer1_strategy.action_model import FUTURE_PREFIXES, ID_COLUMNS, LAYER1_ACTION_LEAKAGE_COLUMNS

DAILY_DOSE_TARGETS = {
    "fsh": "daily_fsh_dose_bin",
    "lh": "daily_lh_dose_bin",
    "hmg": "daily_hmg_dose_bin",
}

DAILY_DOSE_LABELS = {
    "fsh": FSH_DOSE_LABELS,
    "lh": GRID_DOSE_LABELS,
    "hmg": GRID_DOSE_LABELS,
}

CURRENT_DOSE_SOURCE_COLUMNS = {
    "fsh": "current_fsh_daily_dose",
    "lh": "current_lh_daily_dose",
    "hmg": "current_hmg_daily_dose",
}

DAILY_DOSE_SOURCE_COLUMNS = {
    "fsh": "daily_fsh_daily_dose",
    "lh": "daily_lh_daily_dose",
    "hmg": "daily_hmg_daily_dose",
}

DAILY_NO_SW_LEAKAGE_COLUMNS = (
    set(DAILY_DOSE_TARGETS.values())
    | set(DAILY_DOSE_SOURCE_COLUMNS.values())
    | set(CURRENT_DOSE_SOURCE_COLUMNS.values())
    | {
        "current_fsh_dose",
        "current_lh_dose",
        "current_hmg_dose",
        "current_gn_dose",
        "current_lh_like_hmg_daily_dose",
        "current_lh_like_hmg_dose",
        "current_gn_per_follicle",
        "current_gn_per_mature_follicle",
        "current_e2_per_gn",
        "current_lh_like_share",
        "initial_gn_dose",
        "cumulative_fsh_dose",
        "cumulative_lh_dose",
        "cumulative_hmg_dose",
        "cumulative_lh_like_hmg_dose",
        "cumulative_gn_dose",
        "lh_like_share_current_gn",
        "gn_dose_change_vs_previous",
        "gn_dose_ratio_vs_previous",
        "source_evaluation_day",
        "source_gn_day",
        "source_monitoring_order",
        "days_since_source_visit",
    }
)


@dataclass(frozen=True)
class DailyNoSWLabelAudit:
    total_rows: int
    complete_case_rows: int
    off_grid_counts: dict[str, int]
    missing_counts: dict[str, int]
    daily_counts: dict[int, int]
    complete_case_daily_counts: dict[int, int]


def _evaluation_day_from_gn_day(gn_day: pd.Series) -> pd.Series:
    values = pd.to_numeric(gn_day, errors="coerce")
    return (values + 1).round().astype("Int64")


def _last_same_day_snapshots(frame: pd.DataFrame, *, min_day: int, max_day: int) -> pd.DataFrame:
    missing = sorted({"cycle_uid", *CURRENT_DOSE_SOURCE_COLUMNS.values()} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns for daily no-SW panel: {missing}")
    df = frame.copy()
    if "strategy_eligible_flag" in df.columns:
        df = df[df["strategy_eligible_flag"].astype(bool)].copy()
    if "evaluation_day" not in df.columns:
        if "gn_day" not in df.columns:
            raise ValueError("Daily no-SW panel requires either evaluation_day or gn_day.")
        df["evaluation_day"] = _evaluation_day_from_gn_day(df["gn_day"])
    df["source_evaluation_day"] = pd.to_numeric(df["evaluation_day"], errors="coerce")
    df = df[df["source_evaluation_day"].between(min_day, max_day, inclusive="both")].copy()
    if df.empty:
        return df
    sort_columns = ["cycle_uid", "source_evaluation_day"]
    if "monitoring_order" in df.columns:
        sort_columns.append("monitoring_order")
    if "visit_uid" in df.columns:
        sort_columns.append("visit_uid")
    df = df.sort_values(sort_columns, kind="mergesort")
    return df.drop_duplicates(["cycle_uid", "source_evaluation_day"], keep="last").copy()


def _baseline_lookup(baseline_frame: pd.DataFrame | None) -> dict[Any, Mapping[str, Any]]:
    if baseline_frame is None or baseline_frame.empty:
        return {}
    if "cycle_uid" not in baseline_frame.columns:
        raise ValueError("baseline_frame must contain cycle_uid when provided.")
    baseline = baseline_frame.drop_duplicates("cycle_uid", keep="last").set_index("cycle_uid", drop=False)
    return {cycle_uid: row.to_dict() for cycle_uid, row in baseline.iterrows()}


def _label_daily_doses(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for drug, current_col in CURRENT_DOSE_SOURCE_COLUMNS.items():
        value = row.get(current_col, pd.NA)
        result[DAILY_DOSE_SOURCE_COLUMNS[drug]] = value
    result["daily_fsh_dose_bin"] = fsh_absolute_dose_label(result["daily_fsh_daily_dose"])
    result["daily_lh_dose_bin"] = grid_absolute_dose_label(result["daily_lh_daily_dose"])
    result["daily_hmg_dose_bin"] = grid_absolute_dose_label(result["daily_hmg_daily_dose"])
    return result


def _empty_like(columns: Sequence[str]) -> dict[str, Any]:
    return {column: pd.NA for column in columns}


def build_daily_no_sw_panel(
    monitoring_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame | None = None,
    *,
    min_day: int = 1,
    max_day: int = 19,
) -> pd.DataFrame:
    """Expand visit-level stimulation states into a Kong-style daily panel without sliding windows.

    Rows before a cycle's first in-range monitoring state use baseline covariates only as features
    while the earliest observed prescription state supplies the daily dose label.
    """
    snapshots = _last_same_day_snapshots(monitoring_frame, min_day=min_day, max_day=max_day)
    baseline_by_cycle = _baseline_lookup(baseline_frame)
    baseline_columns = list(baseline_frame.columns) if baseline_frame is not None else []
    output_columns = list(
        dict.fromkeys(
            [
                *baseline_columns,
                *snapshots.columns.tolist(),
                "Day",
                "source_gn_day",
                "source_monitoring_order",
                "days_since_source_visit",
                "baseline_only_feature_row",
                *DAILY_DOSE_SOURCE_COLUMNS.values(),
                *DAILY_DOSE_TARGETS.values(),
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    if snapshots.empty:
        return pd.DataFrame(columns=output_columns)

    for cycle_uid, group in snapshots.groupby("cycle_uid", sort=False):
        group = group.sort_values(["source_evaluation_day", "monitoring_order"] if "monitoring_order" in group.columns else ["source_evaluation_day"], kind="mergesort")
        max_cycle_day = int(min(max_day, pd.to_numeric(group["source_evaluation_day"], errors="coerce").max()))
        if max_cycle_day < min_day:
            continue
        first_snapshot = group.iloc[0]
        source_days = pd.to_numeric(group["source_evaluation_day"], errors="coerce").to_numpy(dtype=float)
        for day in range(int(min_day), max_cycle_day + 1):
            available_positions = np.flatnonzero(source_days <= float(day))
            baseline_only = len(available_positions) == 0
            label_source = first_snapshot if baseline_only else group.iloc[int(available_positions[-1])]
            if baseline_only:
                row = _empty_like(output_columns)
                row.update(baseline_by_cycle.get(cycle_uid, {}))
                row["cycle_uid"] = cycle_uid
                if "visit_uid" in label_source.index:
                    row["visit_uid"] = label_source.get("visit_uid")
            else:
                row = label_source.to_dict()
            source_day = int(label_source["source_evaluation_day"])
            row["Day"] = int(day)
            row["evaluation_day"] = int(day)
            row["source_evaluation_day"] = source_day
            row["source_gn_day"] = label_source.get("gn_day", pd.NA)
            row["source_monitoring_order"] = label_source.get("monitoring_order", pd.NA)
            row["days_since_source_visit"] = int(day) - source_day
            row["baseline_only_feature_row"] = bool(baseline_only)
            row.update(_label_daily_doses(label_source.to_dict()))
            rows.append(row)

    panel = pd.DataFrame(rows, columns=output_columns)
    if not panel.empty:
        panel = panel.sort_values(["Day", "cycle_uid"], kind="mergesort").reset_index(drop=True)
    return panel


def daily_no_sw_label_audit(frame: pd.DataFrame, *, day_col: str = "Day") -> DailyNoSWLabelAudit:
    missing_counts: dict[str, int] = {}
    off_grid_counts: dict[str, int] = {}
    for drug, source_col in DAILY_DOSE_SOURCE_COLUMNS.items():
        source = pd.to_numeric(frame[source_col], errors="coerce")
        target = frame[DAILY_DOSE_TARGETS[drug]]
        missing_counts[drug] = int(source.isna().sum())
        off_grid_counts[drug] = int(source.notna().sum() - target.notna().sum())
    complete_mask = np.logical_and.reduce([frame[target].notna().to_numpy() for target in DAILY_DOSE_TARGETS.values()])
    day_values = pd.to_numeric(frame[day_col], errors="coerce")
    complete_day_values = day_values.loc[complete_mask]
    return DailyNoSWLabelAudit(
        total_rows=int(len(frame)),
        complete_case_rows=int(complete_mask.sum()),
        off_grid_counts=off_grid_counts,
        missing_counts=missing_counts,
        daily_counts={int(day): int(count) for day, count in day_values.value_counts().sort_index().items() if pd.notna(day)},
        complete_case_daily_counts={
            int(day): int(count)
            for day, count in complete_day_values.value_counts().sort_index().items()
            if pd.notna(day)
        },
    )


def complete_daily_no_sw_cases(frame: pd.DataFrame) -> pd.DataFrame:
    mask = np.logical_and.reduce([frame[target].notna().to_numpy() for target in DAILY_DOSE_TARGETS.values()])
    return frame.loc[mask].copy()


def select_daily_no_sw_feature_columns(
    frame: pd.DataFrame,
    targets: Iterable[str] = DAILY_DOSE_TARGETS.values(),
) -> list[str]:
    excluded = set(ID_COLUMNS) | set(LAYER1_ACTION_LEAKAGE_COLUMNS) | set(targets) | DAILY_NO_SW_LEAKAGE_COLUMNS
    feature_columns: list[str] = []
    for column in frame.columns:
        name = str(column)
        if column in excluded:
            continue
        if any(name.startswith(prefix) for prefix in FUTURE_PREFIXES):
            continue
        if name.endswith("_action") or name.endswith("_dose_bin"):
            continue
        feature_columns.append(column)
    return feature_columns


def evaluate_daily_no_sw_predictions(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    labels: Sequence[str],
) -> dict[str, Any]:
    from models.layer1_strategy.absolute_dose_heads import evaluate_absolute_dose_predictions

    return evaluate_absolute_dose_predictions(y_true, y_pred, labels)


def daily_no_sw_accuracy_table(
    predictions: pd.DataFrame,
    *,
    day_col: str = "Day",
    min_day: int = 1,
    max_day: int = 19,
) -> pd.DataFrame:
    day_values = pd.to_numeric(predictions[day_col], errors="coerce") if day_col in predictions.columns else pd.Series(dtype=float)
    counts = day_values.value_counts().to_dict()
    at_risk_counts = {int(day): int(counts.get(day, 0)) for day in range(int(min_day), int(max_day) + 1)}
    return kong_style_daily_accuracy_table(
        predictions,
        day_col=day_col,
        min_day=min_day,
        max_day=max_day,
        at_risk_counts=at_risk_counts,
    )


def label_display_map() -> Mapping[str, Mapping[str, str]]:
    return {
        "fsh": {
            "stop": "0 / stop",
            "low_dose": "<80 IU",
            "medium_low": "80-160 IU",
            "medium_high": "160-240 IU",
            "high_dose": ">240 IU",
        },
        "lh": {label: label.replace("dose_", "") + " IU" for label in ABSOLUTE_DOSE_LABELS["lh"]},
        "hmg": {label: label.replace("dose_", "") + " IU" for label in ABSOLUTE_DOSE_LABELS["hmg"]},
    }
