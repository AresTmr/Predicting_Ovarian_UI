from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS


PROBABILITY_COLUMNS = [f"prob_{label}" for label in CTFE_DOSE_LABELS]
ENSEMBLE_PROBABILITY_COLUMNS = [f"ensemble_prob_{label}" for label in CTFE_DOSE_LABELS]


def meets_ctfe_promotion_guardrail(
    candidate_valid_metrics: Mapping[str, float],
    current_valid_metrics: Mapping[str, float] | None,
    *,
    max_valid_macro_drop: float = 0.01,
    min_valid_weighted_gain: float = 0.0,
) -> bool:
    """Prevent a newly accepted experiment from downgrading the current CTFE pointer."""

    if current_valid_metrics is None:
        return True
    return bool(
        float(candidate_valid_metrics["weighted_f1"])
        >= float(current_valid_metrics["weighted_f1"]) + float(min_valid_weighted_gain)
        and float(candidate_valid_metrics["macro_f1"])
        >= float(current_valid_metrics["macro_f1"]) - float(max_valid_macro_drop)
    )


def load_ctfe_run_valid_metrics(output_root: str | Path, run_id: str) -> dict[str, float] | None:
    """Read validation metrics for a CTFE run regardless of ensemble type."""

    run_dir = Path(output_root) / run_id
    candidates = [
        (run_dir / "ctfe_stratified_ensemble_metrics.csv", "stratified_ensemble"),
        (run_dir / "ctfe_stage_specialized_metrics.csv", "stage_specialized"),
        (run_dir / "ctfe_ensemble_metrics.csv", "ensemble"),
        (run_dir / "layer1_ctfe_neural" / "ctfe_neural_metrics.csv", None),
    ]
    for metrics_path, variant in candidates:
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        rows = metrics[metrics["split"].astype(str).eq("valid")].copy()
        if variant is not None and "variant" in rows.columns:
            rows = rows[rows["variant"].astype(str).eq(variant)].copy()
        if rows.empty:
            continue
        row = rows.iloc[0]
        return {
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "weighted_f1": float(row["weighted_f1"]),
            "log_loss": float(row["log_loss"]),
        }
    return None


def _validate_prediction_frame(frame: pd.DataFrame) -> None:
    required = {"split", "ctfe_next_fsh_dose_class", *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing CTFE prediction columns: {missing}")


def _row_key_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    candidates = ["visit_uid", "cycle_uid", "monitoring_order"]
    return [column for column in candidates if column in left.columns and column in right.columns]


def _assert_same_rows(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if len(left) != len(right):
        raise ValueError("CTFE ensemble frames must have the same row count.")
    key_cols = _row_key_columns(left, right)
    if not key_cols:
        return
    if not left[key_cols].astype(str).reset_index(drop=True).equals(right[key_cols].astype(str).reset_index(drop=True)):
        raise ValueError(f"CTFE ensemble frames are not aligned by {key_cols}.")


def blend_ctfe_predictions(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    base_weight: float,
    prediction_col: str = "ctfe_ensemble_prediction",
) -> pd.DataFrame:
    """Blend two CTFE probability outputs with a fixed base-model weight."""

    _validate_prediction_frame(base)
    _validate_prediction_frame(candidate)
    _assert_same_rows(base, candidate)
    if not 0.0 <= float(base_weight) <= 1.0:
        raise ValueError("base_weight must be between 0 and 1.")
    result = base.copy().reset_index(drop=True)
    base_proba = base[PROBABILITY_COLUMNS].astype(float).to_numpy()
    candidate_proba = candidate[PROBABILITY_COLUMNS].astype(float).to_numpy()
    blended = float(base_weight) * base_proba + (1.0 - float(base_weight)) * candidate_proba
    blended = blended / np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)
    for idx, label in enumerate(CTFE_DOSE_LABELS):
        result[f"ensemble_prob_{label}"] = blended[:, idx]
    result[prediction_col] = np.asarray(CTFE_DOSE_LABELS, dtype=object)[blended.argmax(axis=1)]
    return result


def assign_ensemble_strata(frame: pd.DataFrame, *, stratum_col: str = "monitoring_order_group") -> pd.DataFrame:
    """Add validation-safe strata for CTFE ensemble selection.

    The default stratum only uses monitoring order, which is available at the
    current snapshot time. It is therefore safe for validation-only model
    selection and live inference.
    """

    result = frame.copy()
    if stratum_col in result.columns:
        result[stratum_col] = result[stratum_col].astype("object").fillna("unknown")
        return result
    if stratum_col != "monitoring_order_group":
        raise ValueError(f"Missing requested CTFE ensemble stratum column: {stratum_col}")
    if "monitoring_order" not in result.columns:
        result[stratum_col] = "unknown"
        return result
    monitoring_order = pd.to_numeric(result["monitoring_order"], errors="coerce")
    result[stratum_col] = pd.cut(
        monitoring_order,
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["m1_2", "m3_4", "m5_6", "m7_plus"],
        right=True,
    ).astype("object").fillna("unknown")
    return result


def apply_stratified_ctfe_blend(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    stratum_weights: dict[str, float],
    default_base_weight: float,
    stratum_col: str = "monitoring_order_group",
    prediction_col: str = "ctfe_stratified_ensemble_prediction",
) -> pd.DataFrame:
    """Blend CTFE probabilities with a validation-selected weight per stratum."""

    _validate_prediction_frame(base)
    _validate_prediction_frame(candidate)
    _assert_same_rows(base, candidate)
    if not 0.0 <= float(default_base_weight) <= 1.0:
        raise ValueError("default_base_weight must be between 0 and 1.")
    for value in stratum_weights.values():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("all stratum_weights values must be between 0 and 1.")

    result = assign_ensemble_strata(base, stratum_col=stratum_col).reset_index(drop=True)
    candidate = assign_ensemble_strata(candidate, stratum_col=stratum_col).reset_index(drop=True)
    if not result[stratum_col].astype(str).equals(candidate[stratum_col].astype(str)):
        raise ValueError(f"CTFE ensemble frames are not aligned by stratum column {stratum_col!r}.")

    base_proba = base[PROBABILITY_COLUMNS].astype(float).to_numpy()
    candidate_proba = candidate[PROBABILITY_COLUMNS].astype(float).to_numpy()
    row_weights = result[stratum_col].astype(str).map(lambda value: float(stratum_weights.get(value, default_base_weight))).to_numpy()
    blended = row_weights.reshape(-1, 1) * base_proba + (1.0 - row_weights.reshape(-1, 1)) * candidate_proba
    blended = blended / np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)
    for idx, label in enumerate(CTFE_DOSE_LABELS):
        result[f"ensemble_prob_{label}"] = blended[:, idx]
    result["ensemble_stratum"] = result[stratum_col].astype(str)
    result["ensemble_base_weight"] = row_weights
    result[prediction_col] = np.asarray(CTFE_DOSE_LABELS, dtype=object)[blended.argmax(axis=1)]
    return result


def evaluate_ctfe_prediction_frame(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    probability_prefix: str | None = None,
    split: str,
) -> dict[str, float | int | str]:
    subset = frame[frame["split"].astype(str) == split].copy()
    if subset.empty:
        raise ValueError(f"No CTFE rows available for split={split!r}")
    y_true = subset["ctfe_next_fsh_dose_class"].astype(str).to_numpy()
    y_pred = subset[prediction_col].astype(str).to_numpy()
    payload: dict[str, float | int | str] = {
        "split": split,
        "sample_count": int(len(subset)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="weighted", zero_division=0)),
    }
    if probability_prefix:
        proba = subset[[f"{probability_prefix}{label}" for label in CTFE_DOSE_LABELS]].astype(float).to_numpy()
        proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
        y_true_id = np.asarray([CTFE_DOSE_LABELS.index(value) for value in y_true], dtype=int)
        payload["log_loss"] = float(log_loss(y_true_id, proba, labels=list(range(len(CTFE_DOSE_LABELS)))))
    return payload


def search_ctfe_blend_weight(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    weights: Iterable[float],
    max_valid_macro_drop: float = 0.01,
    min_valid_weighted_gain: float = 0.0,
) -> dict[str, object]:
    """Select a blend weight using validation weighted-F1 with macro-F1 guardrail."""

    baseline = evaluate_ctfe_prediction_frame(
        base,
        prediction_col="ctfe_neural_prediction",
        probability_prefix="prob_",
        split="valid",
    )
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for weight in sorted({float(value) for value in weights}):
        blended = blend_ctfe_predictions(base, candidate, base_weight=weight)
        valid_metrics = evaluate_ctfe_prediction_frame(
            blended,
            prediction_col="ctfe_ensemble_prediction",
            probability_prefix="ensemble_prob_",
            split="valid",
        )
        row = {
            "base_weight": weight,
            **{f"valid_{key}": value for key, value in valid_metrics.items() if key != "split"},
        }
        row["passes_guardrail"] = bool(
            float(valid_metrics["weighted_f1"]) >= float(baseline["weighted_f1"]) + float(min_valid_weighted_gain)
            and float(valid_metrics["macro_f1"]) >= float(baseline["macro_f1"]) - float(max_valid_macro_drop)
        )
        rows.append(row)
        if not row["passes_guardrail"]:
            continue
        if best is None:
            best = row
            continue
        candidate_key = (float(valid_metrics["weighted_f1"]), float(valid_metrics["macro_f1"]), float(valid_metrics["accuracy"]))
        best_key = (float(best["valid_weighted_f1"]), float(best["valid_macro_f1"]), float(best["valid_accuracy"]))
        if candidate_key > best_key:
            best = row
    search = pd.DataFrame(rows).sort_values(["passes_guardrail", "valid_weighted_f1", "valid_macro_f1"], ascending=[False, False, False])
    if best is None:
        return {
            "accepted": False,
            "reason": "no_blend_weight_met_validation_guardrails",
            "base_weight": 1.0,
            "baseline_metrics": baseline,
            "selected_metrics": baseline,
            "search_results": search,
        }
    return {
        "accepted": bool(float(best["base_weight"]) != 1.0),
        "reason": "validation_weighted_f1_with_macro_guardrail",
        "base_weight": float(best["base_weight"]),
        "baseline_metrics": baseline,
        "selected_metrics": {
            "valid_accuracy": float(best["valid_accuracy"]),
            "valid_macro_f1": float(best["valid_macro_f1"]),
            "valid_weighted_f1": float(best["valid_weighted_f1"]),
            "valid_log_loss": float(best["valid_log_loss"]),
        },
        "search_results": search,
    }


def search_stratified_ctfe_blend_weights(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    weights: Iterable[float],
    stratum_col: str = "monitoring_order_group",
    min_valid_count: int = 50,
    default_base_weight: float = 1.0,
    max_valid_macro_drop: float = 0.01,
    min_valid_weighted_gain: float = 0.0,
) -> dict[str, object]:
    """Select CTFE blend weights per stratum using validation rows only."""

    base_prepared = assign_ensemble_strata(base, stratum_col=stratum_col)
    candidate_prepared = assign_ensemble_strata(candidate, stratum_col=stratum_col)
    _validate_prediction_frame(base_prepared)
    _validate_prediction_frame(candidate_prepared)
    _assert_same_rows(base_prepared, candidate_prepared)
    if not base_prepared[stratum_col].astype(str).reset_index(drop=True).equals(
        candidate_prepared[stratum_col].astype(str).reset_index(drop=True)
    ):
        raise ValueError(f"CTFE ensemble frames are not aligned by stratum column {stratum_col!r}.")

    baseline_frame = apply_stratified_ctfe_blend(
        base_prepared,
        candidate_prepared,
        stratum_weights={},
        default_base_weight=default_base_weight,
        stratum_col=stratum_col,
        prediction_col="ctfe_default_ensemble_prediction",
    )
    baseline = evaluate_ctfe_prediction_frame(
        baseline_frame,
        prediction_col="ctfe_default_ensemble_prediction",
        probability_prefix="ensemble_prob_",
        split="valid",
    )
    stratum_weights: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    valid_frame = base_prepared[base_prepared["split"].astype(str) == "valid"].copy()
    for stratum_value, subset in valid_frame.groupby(stratum_col, dropna=False):
        value = str(stratum_value)
        subset_index = subset.index
        if len(subset_index) < int(min_valid_count):
            stratum_weights[value] = float(default_base_weight)
            rows.append(
                {
                    "stratum": value,
                    "valid_count": int(len(subset_index)),
                    "base_weight": float(default_base_weight),
                    "accepted": False,
                    "reason": "below_min_valid_count",
                }
            )
            continue
        subset_baseline = evaluate_ctfe_prediction_frame(
            baseline_frame.loc[subset_index].copy(),
            prediction_col="ctfe_default_ensemble_prediction",
            probability_prefix="ensemble_prob_",
            split="valid",
        )
        best_row: dict[str, object] | None = None
        for weight in sorted({float(value) for value in weights}):
            blended_subset = blend_ctfe_predictions(
                base_prepared.loc[subset_index].copy(),
                candidate_prepared.loc[subset_index].copy(),
                base_weight=weight,
            )
            metrics = evaluate_ctfe_prediction_frame(
                blended_subset,
                prediction_col="ctfe_ensemble_prediction",
                probability_prefix="ensemble_prob_",
                split="valid",
            )
            passes = bool(
                float(metrics["weighted_f1"]) >= float(subset_baseline["weighted_f1"])
                and float(metrics["macro_f1"]) >= float(subset_baseline["macro_f1"]) - float(max_valid_macro_drop)
            )
            candidate_row = {
                "base_weight": weight,
                "accepted": passes,
                "valid_accuracy": float(metrics["accuracy"]),
                "valid_macro_f1": float(metrics["macro_f1"]),
                "valid_weighted_f1": float(metrics["weighted_f1"]),
                "valid_log_loss": float(metrics.get("log_loss", np.nan)),
            }
            if not passes:
                continue
            if best_row is None:
                best_row = candidate_row
                continue
            candidate_key = (
                float(candidate_row["valid_weighted_f1"]),
                float(candidate_row["valid_macro_f1"]),
                float(candidate_row["valid_accuracy"]),
            )
            best_key = (
                float(best_row["valid_weighted_f1"]),
                float(best_row["valid_macro_f1"]),
                float(best_row["valid_accuracy"]),
            )
            if candidate_key > best_key:
                best_row = candidate_row
        selected_weight = float(best_row["base_weight"]) if best_row is not None else float(default_base_weight)
        stratum_weights[value] = selected_weight
        selected_metrics = best_row or {
            "valid_accuracy": subset_baseline["accuracy"],
            "valid_macro_f1": subset_baseline["macro_f1"],
            "valid_weighted_f1": subset_baseline["weighted_f1"],
            "valid_log_loss": subset_baseline.get("log_loss", np.nan),
        }
        rows.append(
            {
                "stratum": value,
                "valid_count": int(len(subset_index)),
                "base_weight": selected_weight,
                "accepted": bool(best_row is not None and abs(selected_weight - float(default_base_weight)) > 1e-12),
                "reason": "stratum_weight_improved_or_matched_default" if best_row is not None else "default_weight_kept",
                "valid_accuracy": float(selected_metrics.get("valid_accuracy", baseline["accuracy"])),
                "valid_macro_f1": float(selected_metrics.get("valid_macro_f1", baseline["macro_f1"])),
                "valid_weighted_f1": float(selected_metrics.get("valid_weighted_f1", baseline["weighted_f1"])),
                "valid_log_loss": float(selected_metrics.get("valid_log_loss", baseline.get("log_loss", np.nan))),
            }
        )

    blended = apply_stratified_ctfe_blend(
        base_prepared,
        candidate_prepared,
        stratum_weights=stratum_weights,
        default_base_weight=default_base_weight,
        stratum_col=stratum_col,
    )
    selected_metrics = evaluate_ctfe_prediction_frame(
        blended,
        prediction_col="ctfe_stratified_ensemble_prediction",
        probability_prefix="ensemble_prob_",
        split="valid",
    )
    accepted = bool(
        any(abs(float(weight) - float(default_base_weight)) > 1e-12 for weight in stratum_weights.values())
        and float(selected_metrics["weighted_f1"]) >= float(baseline["weighted_f1"]) + float(min_valid_weighted_gain)
        and float(selected_metrics["macro_f1"]) >= float(baseline["macro_f1"]) - float(max_valid_macro_drop)
    )
    return {
        "accepted": accepted,
        "reason": "stratified_validation_weighted_f1_with_macro_guardrail" if accepted else "no_stratified_blend_met_global_guardrails",
        "stratum_col": stratum_col,
        "default_base_weight": float(default_base_weight),
        "stratum_weights": stratum_weights,
        "baseline_metrics": baseline,
        "selected_metrics": {
            "valid_accuracy": float(selected_metrics["accuracy"]),
            "valid_macro_f1": float(selected_metrics["macro_f1"]),
            "valid_weighted_f1": float(selected_metrics["weighted_f1"]),
            "valid_log_loss": float(selected_metrics.get("log_loss", np.nan)),
        },
        "search_results": pd.DataFrame(rows),
    }
