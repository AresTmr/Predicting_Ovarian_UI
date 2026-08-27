from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS

DEFAULT_CTFE_STRATA = [
    "monitoring_order_group",
    "current_fsh_band",
    "current_gn_band",
    "ovarian_reserve_group",
    "follicle_load_group",
    "true_class",
]


def _series_or_na(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([pd.NA] * len(frame), index=frame.index)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_series_or_na(frame, column), errors="coerce")


def _dose_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bins = [-np.inf, 0, 80, 160, 240, np.inf]
    labels = ["0", "(0,80]", "(80,160]", "(160,240]", ">240"]
    band = pd.cut(numeric, bins=bins, labels=labels, right=True)
    return band.astype("object").fillna("unknown")


def assign_ctfe_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Add clinically interpretable CTFE post-hoc evaluation strata.

    These strata are for evaluation only. They use current-time features and the
    already-created true/predicted label columns, not future outcomes as model inputs.
    """

    result = frame.copy()
    if "true_class" not in result.columns:
        if "ctfe_next_fsh_dose_class" in result.columns:
            result["true_class"] = result["ctfe_next_fsh_dose_class"]
        elif "target" in result.columns:
            result["true_class"] = result["target"]
        else:
            raise ValueError("CTFE diagnostics require true_class or ctfe_next_fsh_dose_class.")
    if "pred_class" not in result.columns:
        if "ctfe_neural_prediction" in result.columns:
            result["pred_class"] = result["ctfe_neural_prediction"]
        elif "prediction" in result.columns:
            result["pred_class"] = result["prediction"]
        else:
            raise ValueError("CTFE diagnostics require pred_class or ctfe_neural_prediction.")
    if "split" not in result.columns:
        result["split"] = "all"

    monitoring_order = _numeric(result, "monitoring_order")
    result["monitoring_order_group"] = pd.cut(
        monitoring_order,
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["m1_2", "m3_4", "m5_6", "m7_plus"],
        right=True,
    ).astype("object").fillna("unknown")
    result["current_fsh_band"] = _dose_band(_numeric(result, "current_fsh_daily_dose"))
    result["current_gn_band"] = _dose_band(_numeric(result, "current_gn_dose"))

    amh = _numeric(result, "amh")
    afc = _numeric(result, "afc")
    reserve = pd.Series(["mid_reserve"] * len(result), index=result.index, dtype="object")
    reserve[(amh < 1.2) | (afc < 6)] = "low_reserve"
    reserve[(amh >= 5.0) | (afc >= 20)] = "high_reserve"
    reserve[amh.isna() & afc.isna()] = "unknown"
    result["ovarian_reserve_group"] = reserve

    follicle_count = _numeric(result, "total_follicle_count")
    load = pd.cut(
        follicle_count,
        bins=[-np.inf, 5, 15, np.inf],
        labels=["low_load", "mid_load", "high_load"],
        right=True,
    ).astype("object").fillna("unknown")
    result["follicle_load_group"] = load
    return result


def _score_subset(subset: pd.DataFrame, split: str, stratum_name: str, stratum_value: str) -> dict[str, object]:
    true = subset["true_class"].astype(str).to_numpy()
    pred = subset["pred_class"].astype(str).to_numpy()
    payload: dict[str, object] = {
        "split": split,
        "stratum_name": stratum_name,
        "stratum_value": stratum_value,
        "sample_count": int(len(subset)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=CTFE_DOSE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=CTFE_DOSE_LABELS, average="weighted", zero_division=0)),
    }
    for label in CTFE_DOSE_LABELS:
        payload[f"support_{label}"] = int((true == label).sum())
        payload[f"pred_count_{label}"] = int((pred == label).sum())
    return payload


def compute_stratified_metrics(
    frame: pd.DataFrame,
    strata_cols: Iterable[str] | None = None,
    *,
    min_count: int = 10,
) -> pd.DataFrame:
    prepared = assign_ctfe_strata(frame)
    strata = list(strata_cols or DEFAULT_CTFE_STRATA)
    rows: list[dict[str, object]] = []

    for split, split_frame in prepared.groupby("split", dropna=False):
        split_name = str(split)
        if len(split_frame) >= min_count:
            rows.append(_score_subset(split_frame, split_name, "overall", "overall"))
        for column in strata:
            if column not in split_frame.columns:
                continue
            for value, subset in split_frame.groupby(column, dropna=False):
                if len(subset) < min_count:
                    continue
                rows.append(_score_subset(subset, split_name, column, str(value)))
    return pd.DataFrame(rows)


def summarize_worst_strata(
    metrics: pd.DataFrame,
    *,
    split: str = "test",
    min_count: int = 30,
    top_n: int = 20,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    subset = metrics[(metrics["split"].astype(str) == split) & (metrics["sample_count"] >= min_count)].copy()
    if subset.empty:
        subset = metrics[metrics["sample_count"] >= min_count].copy()
    return subset.sort_values(["weighted_f1", "sample_count"], ascending=[True, False]).head(top_n).reset_index(drop=True)


def select_best_ctfe_run(
    results: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    baseline_run_id: str,
    baseline_valid_weighted_f1: float,
    baseline_test_weighted_f1: float,
    baseline_valid_macro_f1: float | None = None,
    baseline_test_macro_f1: float | None = None,
    max_valid_drop: float = 0.005,
    max_macro_drop: float = 0.005,
) -> dict[str, object]:
    """Select a CTFE candidate using validation metrics only.

    Test metrics are retained in the returned report but never participate in
    candidate qualification or sorting. This prevents test-set selection bias.
    ``max_valid_drop`` and test baseline arguments are kept for CLI/backward
    compatibility with older experiment records.
    """

    del baseline_test_macro_f1, max_valid_drop
    frame = pd.DataFrame(results).copy()
    baseline_payload: dict[str, object] = {
        "run_id": baseline_run_id,
        "accepted": False,
        "valid_weighted_f1": float(baseline_valid_weighted_f1),
        "test_weighted_f1": float(baseline_test_weighted_f1),
        "selection_split": "valid",
        "test_used_for_selection": False,
    }
    if frame.empty:
        return {**baseline_payload, "reason": "no_candidate_runs"}
    frame["valid_weighted_f1"] = pd.to_numeric(frame["valid_weighted_f1"], errors="coerce")
    frame["test_weighted_f1"] = pd.to_numeric(frame["test_weighted_f1"], errors="coerce")
    if "valid_macro_f1" in frame.columns:
        frame["valid_macro_f1"] = pd.to_numeric(frame["valid_macro_f1"], errors="coerce")
    if "test_macro_f1" in frame.columns:
        frame["test_macro_f1"] = pd.to_numeric(frame["test_macro_f1"], errors="coerce")

    qualified = frame[frame["valid_weighted_f1"] > float(baseline_valid_weighted_f1)].copy()
    if baseline_valid_macro_f1 is not None and "valid_macro_f1" in qualified.columns:
        qualified = qualified[
            qualified["valid_macro_f1"] >= float(baseline_valid_macro_f1) - float(max_macro_drop)
        ].copy()
    if qualified.empty:
        return {
            **baseline_payload,
            "reason": "no_candidate_improved_validation_with_macro_guardrail",
        }

    sort_columns = ["valid_weighted_f1"]
    if "valid_macro_f1" in qualified.columns:
        sort_columns.append("valid_macro_f1")
    qualified = qualified.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    best = qualified.iloc[0].to_dict()
    best.update(
        {
            "accepted": True,
            "reason": "validation_weighted_f1_improved_with_macro_guardrail",
            "selection_split": "valid",
            "test_used_for_selection": False,
        }
    )
    return best
