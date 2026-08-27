from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS, create_ctfe_fsh_labels
from models.layer1_strategy.ctfe_stage import add_ctfe_stage_columns


def build_ctfe_audit_frame(
    frame: pd.DataFrame,
    *,
    split_manifest: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """Build auditable CTFE target rows without adding target values as model inputs."""

    audit = create_ctfe_fsh_labels(frame)
    if "has_next_visit" in audit.columns:
        audit = audit[audit["has_next_visit"].astype(bool)].copy()
    if "strategy_eligible_flag" in audit.columns:
        audit = audit[audit["strategy_eligible_flag"].astype(bool)].copy()
    audit = audit[audit["ctfe_next_fsh_dose_class"].isin(CTFE_DOSE_LABELS)].copy()
    if split_manifest is not None and "cycle_uid" in audit.columns:
        manifest = pd.read_csv(split_manifest) if isinstance(split_manifest, (str, Path)) else split_manifest.copy()
        if {"cycle_uid", "split"}.issubset(manifest.columns):
            mapping = manifest.drop_duplicates("cycle_uid").set_index("cycle_uid")["split"]
            audit["split"] = audit["cycle_uid"].map(mapping).fillna("unassigned")
    if "split" not in audit.columns:
        audit["split"] = "all"
    return add_ctfe_stage_columns(audit).reset_index(drop=True)


def summarize_ctfe_class_distribution(frame: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """Return complete five-class counts and shares, optionally within each group."""

    if group_col is None:
        counts = frame["ctfe_next_fsh_dose_class"].value_counts().reindex(CTFE_DOSE_LABELS, fill_value=0)
        total = int(counts.sum())
        return pd.DataFrame(
            {
                "dose_class": CTFE_DOSE_LABELS,
                "count": [int(counts[label]) for label in CTFE_DOSE_LABELS],
                "share": [float(counts[label] / total) if total else 0.0 for label in CTFE_DOSE_LABELS],
            }
        )
    rows: list[dict[str, object]] = []
    for group, subset in frame.groupby(group_col, dropna=False):
        counts = subset["ctfe_next_fsh_dose_class"].value_counts().reindex(CTFE_DOSE_LABELS, fill_value=0)
        total = int(counts.sum())
        for label in CTFE_DOSE_LABELS:
            rows.append(
                {
                    "group_col": group_col,
                    "group_value": str(group),
                    "dose_class": label,
                    "count": int(counts[label]),
                    "group_total": total,
                    "share_within_group": float(counts[label] / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_stage_class_distribution(frame: pd.DataFrame, stage_col: str = "ctfe_stage_group") -> pd.DataFrame:
    if stage_col not in frame.columns:
        raise ValueError(f"Missing CTFE stage column: {stage_col}")
    return summarize_ctfe_class_distribution(frame, group_col=stage_col)


def summarize_boundary_doses(frame: pd.DataFrame) -> pd.DataFrame:
    """List actual next FSH dose values feeding the five CTFE target bins."""

    if "next_fsh_daily_dose" not in frame.columns:
        raise ValueError("CTFE boundary audit requires next_fsh_daily_dose.")
    grouped = (
        frame.groupby(["ctfe_next_fsh_dose_class", "next_fsh_daily_dose"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    total = int(grouped["count"].sum())
    grouped["share"] = grouped["count"] / total if total else 0.0
    grouped["dose_class"] = pd.Categorical(grouped["ctfe_next_fsh_dose_class"], CTFE_DOSE_LABELS, ordered=True)
    return grouped.sort_values(["dose_class", "next_fsh_daily_dose"]).drop(columns=["dose_class"]).reset_index(drop=True)


def find_sparse_stage_classes(
    frame: pd.DataFrame,
    *,
    stage_cols: Iterable[str] = ("ctfe_stage_group", "gn_day_group", "split"),
    min_support: int = 30,
) -> pd.DataFrame:
    """Report label cells with insufficient support for stable evaluation/training."""

    tables = []
    for stage_col in stage_cols:
        if stage_col not in frame.columns:
            continue
        table = summarize_ctfe_class_distribution(frame, group_col=stage_col)
        table["min_support"] = int(min_support)
        table["is_sparse"] = table["count"] < int(min_support)
        tables.append(table[table["is_sparse"]].copy())
    if not tables:
        return pd.DataFrame(columns=["group_col", "group_value", "dose_class", "count", "group_total", "share_within_group", "min_support", "is_sparse"])
    return pd.concat(tables, ignore_index=True).sort_values(["group_col", "group_value", "count", "dose_class"]).reset_index(drop=True)

def build_repeated_split_protocol_metadata(seeds: Iterable[int]) -> dict[str, object]:
    """Document stability protocol so no reported test result is used for model selection."""

    return {
        "seeds": [int(seed) for seed in seeds],
        "selection_split": "valid",
        "test_used_for_selection": False,
        "updates_current_pointer": False,
        "split_grain": "cycle_uid",
        "method": "train GRU and TDNN per seed; choose stratified blend weights on validation only; report test metrics",
    }


def summarize_repeated_split_metrics(
    metrics: pd.DataFrame,
    *,
    variant: str = "stratified_ensemble",
    split: str = "test",
) -> pd.DataFrame:
    subset = metrics[(metrics["variant"].astype(str) == variant) & (metrics["split"].astype(str) == split)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["variant", "split", "n_splits"])
    row: dict[str, object] = {"variant": variant, "split": split, "n_splits": int(subset["seed"].nunique())}
    for metric in ["accuracy", "macro_f1", "weighted_f1", "log_loss"]:
        values = pd.to_numeric(subset[metric], errors="coerce").dropna()
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row[f"{metric}_min"] = float(values.min())
        row[f"{metric}_max"] = float(values.max())
    return pd.DataFrame([row])