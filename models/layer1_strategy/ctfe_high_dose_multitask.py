from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS


CONTROL_VARIANT = "control_fusion"
CANDIDATE_VARIANT = "high_dose_multitask_candidate"
LOGISTIC_VARIANT = "high_dose_multitask_logistic"
CATBOOST_VARIANT = "high_dose_multitask_catboost"
MODEL_TYPES = [LOGISTIC_VARIANT, CATBOOST_VARIANT]

EARLY_STAGE_GROUPS = frozenset({"d0_3", "d4_6"})
UPPER_DOSE_LABELS = frozenset({"medium", "high"})
BOUNDARY_LABELS = frozenset({"medium", "high"})
TASK_NAMES = ["early_high", "upper_dose", "medium_high_boundary"]
DEFAULT_TASK_WEIGHTS = {
    "early_high": 0.40,
    "upper_dose": 0.20,
    "medium_high_boundary": 0.40,
}
TASK_WEIGHT_GRID = [
    {"early_high": 0.40, "upper_dose": 0.20, "medium_high_boundary": 0.40},
    {"early_high": 0.50, "upper_dose": 0.20, "medium_high_boundary": 0.30},
    {"early_high": 0.30, "upper_dose": 0.20, "medium_high_boundary": 0.50},
    {"early_high": 0.45, "upper_dose": 0.10, "medium_high_boundary": 0.45},
]

HIGH_DOSE_MULTITASK_FEATURES = [
    "age",
    "bmi",
    "afc",
    "amh",
    "basal_fsh",
    "basal_lh",
    "basal_e2",
    "basal_p",
    "cycle_day",
    "gn_day",
    "current_e2",
    "current_lh",
    "current_p",
    "current_fsh",
    "current_endometrium",
    "current_fsh_daily_dose",
    "current_lh_daily_dose",
    "current_hmg_daily_dose",
    "current_gn_dose",
    "cumulative_fsh_dose",
    "cumulative_gn_dose",
    "previous_fsh_daily_dose",
    "previous_gn_dose",
    "delta_e2",
    "delta_lh",
    "delta_p",
    "days_since_previous_visit",
    "visits_seen",
    "total_follicle_count",
    "mature_follicle_count",
    "follicle_count_10_12",
    "follicle_count_13_15",
    "follicle_count_16_18",
    "follicle_count_gt_18",
    "mature_follicle_share",
    "large_follicle_share",
    "gn_per_total_follicle",
    "gn_per_mature_follicle",
    "current_fsh_share_gn",
    "current_e2_per_gn",
]

FORBIDDEN_HIGH_DOSE_FRAGMENTS = (
    "next_",
    "target_",
    "observed_",
    "oocytes",
    "mii",
    "ohss",
    "clinical_pregnancy",
    "live_birth",
    "embryo",
    "transfer",
    "procedure",
)
ENSEMBLE_PROBABILITY_COLUMNS = [f"ensemble_prob_{label}" for label in CTFE_DOSE_LABELS]
MULTITASK_PROBABILITY_COLUMNS = [f"high_dose_multitask_prob_{label}" for label in CTFE_DOSE_LABELS]
LABEL_INDEX = {label: index for index, label in enumerate(CTFE_DOSE_LABELS)}


@dataclass
class ConstantBinaryHead:
    positive_probability: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), float(self.positive_probability), dtype=float)
        return np.column_stack([1.0 - positive, positive])


def validate_high_dose_feature_columns(feature_columns: Sequence[str]) -> None:
    bad = [
        str(name)
        for name in feature_columns
        if any(fragment in str(name).lower() for fragment in FORBIDDEN_HIGH_DOSE_FRAGMENTS)
    ]
    if bad:
        raise ValueError(f"High-dose multitask feature list contains leakage-prone columns: {bad}")


def resolve_high_dose_feature_columns(
    frame: pd.DataFrame,
    requested: Sequence[str] = HIGH_DOSE_MULTITASK_FEATURES,
) -> list[str]:
    validate_high_dose_feature_columns(requested)
    columns = [column for column in requested if column in frame.columns]
    if not columns:
        raise ValueError("No high-dose multitask feature columns are present in the frame.")
    return columns


def _ensure_prediction_columns(frame: pd.DataFrame) -> None:
    required = {
        "split",
        "cycle_uid",
        "gn_day_group",
        "ctfe_next_fsh_dose_class",
        "ctfe_stratified_ensemble_prediction",
        *ENSEMBLE_PROBABILITY_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"High-dose multitask input missing columns: {missing}")


def _require_train_only(frame: pd.DataFrame) -> None:
    if "split" not in frame.columns or not frame["split"].astype(str).eq("train").all():
        raise ValueError("High-dose multitask heads must be fitted from train rows only.")


def _normalise(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    probabilities = frame[list(columns)].astype(float).to_numpy()
    return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


def _task_row_mask(frame: pd.DataFrame, task_name: str, *, for_training: bool) -> np.ndarray:
    stage = frame["gn_day_group"].astype(str)
    truth = frame["ctfe_next_fsh_dose_class"].astype(str)
    early = stage.isin(EARLY_STAGE_GROUPS)
    if task_name == "early_high":
        return early.to_numpy()
    if task_name == "upper_dose":
        return early.to_numpy()
    if task_name == "medium_high_boundary":
        if for_training:
            return (early & truth.isin(BOUNDARY_LABELS)).to_numpy()
        return early.to_numpy()
    raise ValueError(f"Unknown high-dose multitask head: {task_name}")


def _task_target(frame: pd.DataFrame, task_name: str) -> pd.Series:
    truth = frame["ctfe_next_fsh_dose_class"].astype(str)
    if task_name == "early_high":
        return truth.eq("high").astype(int)
    if task_name == "upper_dose":
        return truth.isin(UPPER_DOSE_LABELS).astype(int)
    if task_name == "medium_high_boundary":
        return truth.eq("high").astype(int)
    raise ValueError(f"Unknown high-dose multitask head: {task_name}")


def _build_model(model_type: str, *, seed: int) -> object:
    if model_type == LOGISTIC_VARIANT:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        C=0.5,
                        solver="liblinear",
                        max_iter=500,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    if model_type == CATBOOST_VARIANT:
        return CatBoostClassifier(
            iterations=100,
            depth=3,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            random_seed=int(seed),
            thread_count=4,
        )
    raise ValueError(f"Unknown high-dose multitask model type: {model_type}")


def _balanced_cycle_sample_weight(subset: pd.DataFrame, y: pd.Series) -> np.ndarray:
    cycle_counts = subset["cycle_uid"].astype(str).map(subset["cycle_uid"].astype(str).value_counts()).to_numpy(dtype=float)
    cycle_weight = 1.0 / np.maximum(cycle_counts, 1.0)
    class_counts = y.value_counts().to_dict()
    class_weight = y.map(lambda value: len(y) / (2.0 * max(class_counts.get(int(value), 1), 1))).to_numpy(dtype=float)
    weights = cycle_weight * class_weight
    return weights / max(float(weights.mean()), 1e-12)


def _fit_binary_head(
    subset: pd.DataFrame,
    y: pd.Series,
    *,
    model_type: str,
    feature_columns: Sequence[str],
    seed: int,
) -> object:
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if subset.empty or positive == 0 or negative == 0:
        return ConstantBinaryHead(float(positive > 0))
    model = _build_model(model_type, seed=int(seed))
    sample_weight = _balanced_cycle_sample_weight(subset, y)
    if model_type == LOGISTIC_VARIANT:
        model.fit(subset[list(feature_columns)], y.to_numpy(dtype=int), classifier__sample_weight=sample_weight)
    elif model_type == CATBOOST_VARIANT:
        model.fit(subset[list(feature_columns)], y.to_numpy(dtype=int), sample_weight=sample_weight)
    else:
        model.fit(subset[list(feature_columns)], y.to_numpy(dtype=int))
    return model


def fit_high_dose_multitask_heads(
    training_frame: pd.DataFrame,
    *,
    model_type: str,
    feature_columns: Sequence[str] | None = None,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    _ensure_prediction_columns(training_frame)
    _require_train_only(training_frame)
    active_features = list(feature_columns or resolve_high_dose_feature_columns(training_frame))
    validate_high_dose_feature_columns(active_features)
    missing = sorted(set(active_features).difference(training_frame.columns))
    if missing:
        raise ValueError(f"High-dose multitask feature columns missing: {missing}")
    heads: dict[str, object] = {}
    support_rows: list[dict[str, object]] = []
    for offset, task_name in enumerate(TASK_NAMES):
        mask = _task_row_mask(training_frame, task_name, for_training=True)
        subset = training_frame.loc[mask].copy().reset_index(drop=True)
        y = _task_target(subset, task_name)
        heads[task_name] = _fit_binary_head(
            subset,
            y,
            model_type=model_type,
            feature_columns=active_features,
            seed=int(seed) + offset + 1,
        )
        support_rows.append(
            {
                "task_name": task_name,
                "model_type": model_type,
                "training_stage_groups": ",".join(sorted(EARLY_STAGE_GROUPS)),
                "sample_count": int(len(subset)),
                "positive_count": int(y.sum()),
                "negative_count": int(len(y) - int(y.sum())),
                "positive_rate": float(y.mean()) if len(y) else 0.0,
            }
        )
    return heads, pd.DataFrame(support_rows)


def add_high_dose_multitask_probabilities(
    frame: pd.DataFrame,
    heads: Mapping[str, object],
    *,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    _ensure_prediction_columns(frame)
    active_features = list(feature_columns or resolve_high_dose_feature_columns(frame))
    validate_high_dose_feature_columns(active_features)
    result = frame.copy().reset_index(drop=True)
    for task_name in TASK_NAMES:
        mask = _task_row_mask(result, task_name, for_training=False)
        probabilities = np.zeros(len(result), dtype=float)
        if mask.any():
            model = heads[task_name]
            probabilities[mask] = np.asarray(model.predict_proba(result.loc[mask, active_features]), dtype=float)[:, 1]
        result[f"multitask_prob_{task_name}"] = np.clip(probabilities, 0.0, 1.0)
    return result


def combine_multitask_high_probability(
    frame: pd.DataFrame,
    *,
    task_weights: Mapping[str, float] = DEFAULT_TASK_WEIGHTS,
) -> np.ndarray:
    weights = {task: float(task_weights.get(task, 0.0)) for task in TASK_NAMES}
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0.0:
        raise ValueError("At least one high-dose multitask weight must be positive.")
    probability = np.zeros(len(frame), dtype=float)
    for task_name in TASK_NAMES:
        column = f"multitask_prob_{task_name}"
        if column not in frame.columns:
            raise ValueError(f"Missing multitask probability column: {column}")
        probability += max(weights[task_name], 0.0) / total * pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.clip(probability, 0.0, 1.0)


def _high_probability_rank(probabilities: np.ndarray) -> np.ndarray:
    high_idx = LABEL_INDEX["high"]
    order = np.argsort(-probabilities, axis=1)
    return np.asarray([int(np.where(order[row] == high_idx)[0][0]) + 1 for row in range(probabilities.shape[0])])


def apply_high_dose_multitask_fusion(
    frame: pd.DataFrame,
    *,
    task_weights: Mapping[str, float],
    high_probability_threshold: float,
    maximum_control_margin: float,
    blend_weight: float,
    max_high_rank: int = 2,
    allowed_control_labels: Sequence[str] = ("medium",),
    prediction_col: str = "ctfe_high_dose_multitask_prediction",
) -> pd.DataFrame:
    _ensure_prediction_columns(frame)
    for value in [high_probability_threshold, maximum_control_margin, blend_weight]:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("High-dose multitask fusion parameters must be between zero and one.")
    if int(max_high_rank) < 1:
        raise ValueError("max_high_rank must be positive.")
    result = frame.copy().reset_index(drop=True)
    probabilities = _normalise(result, ENSEMBLE_PROBABILITY_COLUMNS)
    labels = np.asarray(CTFE_DOSE_LABELS, dtype=object)
    high_idx = LABEL_INDEX["high"]
    base_prediction = result["ctfe_stratified_ensemble_prediction"].astype(str).to_numpy()
    multitask_probability = combine_multitask_high_probability(result, task_weights=task_weights)
    result["multitask_high_probability"] = multitask_probability
    early = result["gn_day_group"].astype(str).isin(EARLY_STAGE_GROUPS).to_numpy()
    high_rank = _high_probability_rank(probabilities)
    top_base = probabilities.max(axis=1)
    high_base = probabilities[:, high_idx]
    margin = top_base - high_base
    eligible = (
        early
        & pd.Series(base_prediction).isin([str(label) for label in allowed_control_labels]).to_numpy()
        & (multitask_probability >= float(high_probability_threshold))
        & (high_rank <= int(max_high_rank))
        & (margin <= float(maximum_control_margin) + 1e-12)
    )
    candidate_probabilities = probabilities.copy()
    adjustment = np.full(len(result), "none", dtype=object)
    for row_index in np.flatnonzero(eligible):
        old_high = float(probabilities[row_index, high_idx])
        proposed_high = (1.0 - float(blend_weight)) * old_high + float(blend_weight) * float(multitask_probability[row_index])
        proposed_high = min(max(proposed_high, 0.0), 0.98)
        old_other = max(1.0 - old_high, 1e-12)
        new_other = max(1.0 - proposed_high, 0.0)
        row = candidate_probabilities[row_index].copy()
        row *= new_other / old_other
        row[high_idx] = proposed_high
        row = row / max(row.sum(), 1e-12)
        if int(row.argmax()) == high_idx:
            candidate_probabilities[row_index] = row
            adjustment[row_index] = f"{base_prediction[row_index]}_to_high"
    candidate_probabilities = candidate_probabilities / np.maximum(candidate_probabilities.sum(axis=1, keepdims=True), 1e-12)
    for index, label in enumerate(CTFE_DOSE_LABELS):
        result[f"high_dose_multitask_prob_{label}"] = candidate_probabilities[:, index]
    result[prediction_col] = labels[candidate_probabilities.argmax(axis=1)]
    result["high_dose_multitask_adjustment"] = adjustment
    result["high_dose_multitask_changed_prediction"] = result[prediction_col].astype(str).to_numpy() != base_prediction
    result["high_dose_multitask_high_rank"] = high_rank
    result["high_dose_multitask_control_margin"] = margin
    return result


def evaluate_high_dose_multitask_frame(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    probability_prefix: str = "high_dose_multitask_prob_",
) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("Cannot evaluate an empty high-dose multitask frame.")
    y_true = frame["ctfe_next_fsh_dose_class"].astype(str).to_numpy()
    y_pred = frame[prediction_col].astype(str).to_numpy()
    probability_columns = [f"{probability_prefix}{label}" for label in CTFE_DOSE_LABELS]
    probabilities = _normalise(frame, probability_columns)
    class_f1 = f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average=None, zero_division=0)
    precision, recall, _, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CTFE_DOSE_LABELS,
        zero_division=0,
    )
    distances = np.asarray([abs(LABEL_INDEX[a] - LABEL_INDEX[b]) for a, b in zip(y_true, y_pred)])
    metrics: dict[str, float | int] = {
        "sample_count": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="weighted", zero_division=0)),
        "log_loss": float(log_loss([LABEL_INDEX[value] for value in y_true], probabilities, labels=list(range(len(CTFE_DOSE_LABELS))))),
        "adjacent_errors": int((distances == 1).sum()),
        "far_errors": int((distances >= 2).sum()),
    }
    for label, value, p_value, r_value, support_value in zip(CTFE_DOSE_LABELS, class_f1, precision, recall, support):
        metrics[f"f1_{label}"] = float(value)
        metrics[f"precision_{label}"] = float(p_value)
        metrics[f"recall_{label}"] = float(r_value)
        metrics[f"support_{label}"] = int(support_value)
    if "high_dose_multitask_changed_prediction" in frame.columns:
        changed = frame["high_dose_multitask_changed_prediction"].astype(bool)
        adjustment = frame.get("high_dose_multitask_adjustment", pd.Series(["none"] * len(frame))).astype(str)
        to_high = changed & adjustment.str.endswith("_to_high")
        correct = to_high & frame["ctfe_next_fsh_dose_class"].astype(str).eq("high")
        wrong = to_high & ~frame["ctfe_next_fsh_dose_class"].astype(str).eq("high")
        metrics["changed_predictions"] = int(changed.sum())
        metrics["to_high_changes"] = int(to_high.sum())
        metrics["to_high_correct"] = int(correct.sum())
        metrics["to_high_wrong"] = int(wrong.sum())
        metrics["to_high_net_correct"] = int(correct.sum() - wrong.sum())
    metrics["predicted_high_count"] = int((pd.Series(y_pred).astype(str) == "high").sum())
    metrics["true_high_count"] = int((pd.Series(y_true).astype(str) == "high").sum())
    return metrics


def _control_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return evaluate_high_dose_multitask_frame(
        frame,
        prediction_col="ctfe_stratified_ensemble_prediction",
        probability_prefix="ensemble_prob_",
    )


def _weight_label(task_weights: Mapping[str, float]) -> str:
    return ",".join(f"{task}={float(task_weights.get(task, 0.0)):.2f}" for task in TASK_NAMES)


def search_high_dose_multitask_parameters(
    frame: pd.DataFrame,
    *,
    selection_mask: np.ndarray,
    task_weight_grid: Sequence[Mapping[str, float]] = TASK_WEIGHT_GRID,
    blend_weights: Iterable[float] = (0.30, 0.50, 0.70, 0.90),
    high_probability_thresholds: Iterable[float] = (0.45, 0.50, 0.55, 0.60, 0.65),
    maximum_control_margins: Iterable[float] = (0.04, 0.08, 0.12, 0.16, 0.20),
    max_high_ranks: Iterable[int] = (2, 3),
    maximum_log_loss_increase: float = 0.001,
    maximum_high_precision_drop: float = 0.02,
) -> dict[str, object]:
    mask = np.asarray(selection_mask, dtype=bool)
    if len(mask) != len(frame) or not mask.any():
        raise ValueError("High-dose multitask parameter search requires a non-empty aligned inner-tune mask.")
    tune = frame.loc[mask].reset_index(drop=True)
    baseline = _control_metrics(tune)
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for task_weights, blend_weight, threshold, margin, max_rank in product(
        task_weight_grid,
        blend_weights,
        high_probability_thresholds,
        maximum_control_margins,
        max_high_ranks,
    ):
        candidate = apply_high_dose_multitask_fusion(
            tune,
            task_weights=task_weights,
            high_probability_threshold=float(threshold),
            maximum_control_margin=float(margin),
            blend_weight=float(blend_weight),
            max_high_rank=int(max_rank),
        )
        metrics = evaluate_high_dose_multitask_frame(candidate, prediction_col="ctfe_high_dose_multitask_prediction")
        changed = int(metrics.get("changed_predictions", 0))
        passes = bool(
            changed > 0
            and float(metrics["weighted_f1"]) >= float(baseline["weighted_f1"]) - 1e-12
            and float(metrics["accuracy"]) >= float(baseline["accuracy"]) - 1e-12
            and float(metrics["macro_f1"]) >= float(baseline["macro_f1"]) - 1e-12
            and float(metrics["f1_high"]) > float(baseline["f1_high"])
            and float(metrics["recall_high"]) >= float(baseline["recall_high"]) - 1e-12
            and float(metrics["precision_high"]) >= float(baseline["precision_high"]) - float(maximum_high_precision_drop) - 1e-12
            and float(metrics["log_loss"]) <= float(baseline["log_loss"]) + float(maximum_log_loss_increase) + 1e-12
            and int(metrics["adjacent_errors"]) <= int(baseline["adjacent_errors"])
        )
        result = {
            "task_weight_label": _weight_label(task_weights),
            "task_weights": dict(task_weights),
            "blend_weight": float(blend_weight),
            "high_probability_threshold": float(threshold),
            "maximum_control_margin": float(margin),
            "max_high_rank": int(max_rank),
            "changed_predictions": changed,
            "passes_inner_tune_guardrail": passes,
            **{f"tune_{name}": value for name, value in metrics.items()},
        }
        rows.append(result)
        key = (
            float(result["tune_weighted_f1"]),
            float(result["tune_macro_f1"]),
            float(result["tune_f1_high"]),
            float(result["tune_recall_high"]),
            -float(result["tune_log_loss"]),
            int(result.get("tune_to_high_net_correct", 0)),
            -changed,
        )
        if passes and (
            best is None
            or key
            > (
                float(best["tune_weighted_f1"]),
                float(best["tune_macro_f1"]),
                float(best["tune_f1_high"]),
                float(best["tune_recall_high"]),
                -float(best["tune_log_loss"]),
                int(best.get("tune_to_high_net_correct", 0)),
                -int(best["changed_predictions"]),
            )
        ):
            best = result
    search_results = pd.DataFrame(rows).sort_values(
        ["passes_inner_tune_guardrail", "tune_weighted_f1", "tune_macro_f1", "tune_f1_high", "tune_log_loss"],
        ascending=[False, False, False, False, True],
    )
    if best is None:
        best = {
            "task_weight_label": _weight_label(DEFAULT_TASK_WEIGHTS),
            "task_weights": dict(DEFAULT_TASK_WEIGHTS),
            "blend_weight": 0.0,
            "high_probability_threshold": 1.0,
            "maximum_control_margin": 0.0,
            "max_high_rank": 1,
            "changed_predictions": 0,
        }
        accepted_for_validation = False
        reason = "no_nontrivial_inner_tune_guarded_high_dose_multitask_fusion"
    else:
        accepted_for_validation = True
        reason = "inner_tune_guarded_high_dose_multitask_fusion_selected"
    return {
        **best,
        "accepted_for_validation": accepted_for_validation,
        "reason": reason,
        "parameter_selection_split": "inner_tune",
        "test_used_for_selection": False,
        "baseline_metrics": baseline,
        "search_results": search_results,
    }


def summarize_high_dose_multitask_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "f1_high",
        "precision_high",
        "recall_high",
        "adjacent_errors",
        "far_errors",
        "changed_predictions",
        "to_high_changes",
        "to_high_net_correct",
        "predicted_high_count",
        "true_high_count",
    ]
    rows: list[dict[str, object]] = []
    for (scope, variant, split), group in metrics.groupby(["scope", "variant", "split"], sort=True):
        entry: dict[str, object] = {
            "scope": scope,
            "variant": variant,
            "split": split,
            "n_seeds": int(group["seed"].nunique()),
        }
        for name in metric_names:
            if name in group.columns:
                values = pd.to_numeric(group[name], errors="coerce")
                entry[f"{name}_mean"] = float(values.mean())
                entry[f"{name}_std"] = float(values.std(ddof=0))
        rows.append(entry)
    return pd.DataFrame(rows)


def assess_high_dose_multitask_promotion(
    metrics: pd.DataFrame,
    *,
    scope: str,
    min_improved_seeds: int,
    maximum_log_loss_increase: float = 0.001,
    maximum_high_precision_drop: float = 0.02,
) -> dict[str, object]:
    valid = metrics[(metrics["scope"].astype(str).eq(scope)) & (metrics["split"].astype(str).eq("valid"))].copy()
    grouped = valid.groupby("variant")[
        [
            "accuracy",
            "weighted_f1",
            "macro_f1",
            "log_loss",
            "f1_high",
            "precision_high",
            "recall_high",
            "adjacent_errors",
            "changed_predictions",
        ]
    ].mean()
    if CONTROL_VARIANT not in grouped.index or CANDIDATE_VARIANT not in grouped.index:
        raise ValueError(f"High-dose multitask comparison for {scope!r} requires both validation variants.")
    control = grouped.loc[CONTROL_VARIANT]
    candidate = grouped.loc[CANDIDATE_VARIANT]
    paired = valid.pivot_table(index="seed", columns="variant", values="weighted_f1", aggfunc="first").dropna()
    improved_seed_count = int((paired[CANDIDATE_VARIANT] >= paired[CONTROL_VARIANT]).sum())
    guards = {
        "weighted_f1_noninferior": bool(candidate["weighted_f1"] >= control["weighted_f1"] - 1e-12),
        "accuracy_noninferior": bool(candidate["accuracy"] >= control["accuracy"] - 1e-12),
        "macro_f1_noninferior": bool(candidate["macro_f1"] >= control["macro_f1"] - 1e-12),
        "high_f1_improved": bool(candidate["f1_high"] > control["f1_high"]),
        "high_recall_noninferior": bool(candidate["recall_high"] >= control["recall_high"] - 1e-12),
        "high_precision_tolerance_met": bool(candidate["precision_high"] >= control["precision_high"] - float(maximum_high_precision_drop) - 1e-12),
        "log_loss_tolerance_met": bool(candidate["log_loss"] <= control["log_loss"] + float(maximum_log_loss_increase) + 1e-12),
        "adjacent_errors_noninferior": bool(candidate["adjacent_errors"] <= control["adjacent_errors"]),
        "changed_rows_positive": bool(candidate["changed_predictions"] > 0),
        "paired_weighted_f1_seed_count": bool(improved_seed_count >= int(min_improved_seeds)),
    }
    accepted = bool(all(guards.values()))
    return {
        "scope": scope,
        "accepted": accepted,
        "reason": "all_validation_high_dose_multitask_guardrails_passed" if accepted else "validation_high_dose_multitask_guardrail_failed",
        "parameter_selection_split": "inner_tune",
        "selection_split": "valid",
        "test_used_for_selection": False,
        "maximum_log_loss_increase": float(maximum_log_loss_increase),
        "maximum_high_precision_drop": float(maximum_high_precision_drop),
        "min_improved_seeds": int(min_improved_seeds),
        "improved_seed_count": improved_seed_count,
        "guards": guards,
        "control_valid_metrics": {name: float(value) for name, value in control.to_dict().items()},
        "candidate_valid_metrics": {name: float(value) for name, value in candidate.to_dict().items()},
        "validation_delta_candidate_minus_control": {name: float(candidate[name] - control[name]) for name in grouped.columns},
    }
