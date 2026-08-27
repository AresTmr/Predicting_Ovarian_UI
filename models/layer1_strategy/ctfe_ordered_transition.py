from __future__ import annotations

from dataclasses import dataclass
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
CANDIDATE_VARIANT = "ordered_transition_candidate"
ORDINAL_LOGISTIC = "ordinal_logistic"
ORDINAL_CATBOOST = "ordinal_catboost"
MODEL_TYPES = [ORDINAL_LOGISTIC, ORDINAL_CATBOOST]
THRESHOLD_TARGETS = ["above_0", "above_80", "above_160", "above_240"]
TRANSITION_LABELS = ["decrease_bin", "maintain_bin", "increase_bin"]
ORDERED_FEATURES = [
    "afc",
    "amh",
    "gn_day",
    "current_e2",
    "current_fsh_daily_dose",
    "current_gn_dose",
    "total_follicle_count",
    "mature_follicle_count",
]
FORBIDDEN_ORDERED_FRAGMENTS = (
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
)
ENSEMBLE_PROBABILITY_COLUMNS = [f"ensemble_prob_{label}" for label in CTFE_DOSE_LABELS]
ORDERED_PROBABILITY_COLUMNS = [f"ordered_prob_{label}" for label in CTFE_DOSE_LABELS]
CANDIDATE_PROBABILITY_COLUMNS = [f"candidate_prob_{label}" for label in CTFE_DOSE_LABELS]
LABEL_INDEX = {label: index for index, label in enumerate(CTFE_DOSE_LABELS)}


@dataclass
class ConstantBinaryHead:
    positive_probability: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), float(self.positive_probability), dtype=float)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class ConstantMulticlassHead:
    label: str

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray([self.label], dtype=object)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.ones((len(frame), 1), dtype=float)


def validate_ordered_feature_columns(feature_columns: Sequence[str]) -> None:
    bad = [
        str(name)
        for name in feature_columns
        if any(fragment in str(name).lower() for fragment in FORBIDDEN_ORDERED_FRAGMENTS)
    ]
    if bad:
        raise ValueError(f"Ordered CTFE feature list contains leakage-prone columns: {bad}")


def _validate_features(frame: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    validate_ordered_feature_columns(feature_columns)
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Ordered CTFE feature columns missing: {missing}")


def _require_train_only(frame: pd.DataFrame) -> None:
    if "split" in frame.columns and not frame["split"].astype(str).eq("train").all():
        raise ValueError("Ordered CTFE fitting is restricted to train rows.")


def map_fsh_dose_class(values: pd.Series) -> pd.Series:
    dose = pd.to_numeric(values, errors="coerce")
    mapped = np.select(
        [dose.le(0), dose.gt(0) & dose.le(80), dose.gt(80) & dose.le(160), dose.gt(160) & dose.le(240), dose.gt(240)],
        ["stop", "decrease", "low", "medium", "high"],
        default="unknown",
    )
    return pd.Series(mapped, index=values.index, dtype="object")


def add_ordered_transition_targets(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ctfe_next_fsh_dose_class", "current_fsh_daily_dose"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Ordered CTFE targets require columns: {missing}")
    result = frame.copy()
    truth = result["ctfe_next_fsh_dose_class"].astype(str)
    invalid = sorted(set(truth).difference(CTFE_DOSE_LABELS))
    if invalid:
        raise ValueError(f"Unknown CTFE dose labels: {invalid}")
    result["current_fsh_class"] = map_fsh_dose_class(result["current_fsh_daily_dose"])
    if result["current_fsh_class"].eq("unknown").any():
        raise ValueError("Current FSH dose cannot be mapped to an ordered dose class.")
    true_index = truth.map(LABEL_INDEX).astype(int)
    current_index = result["current_fsh_class"].map(LABEL_INDEX).astype(int)
    result["above_0_target"] = true_index.ge(1).astype(int)
    result["above_80_target"] = true_index.ge(2).astype(int)
    result["above_160_target"] = true_index.ge(3).astype(int)
    result["above_240_target"] = true_index.ge(4).astype(int)
    delta = true_index - current_index
    result["transition_target"] = np.select(
        [delta.lt(0), delta.eq(0), delta.gt(0)],
        TRANSITION_LABELS,
        default="maintain_bin",
    )
    return result


def recover_five_class_probabilities(cumulative_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.asarray(cumulative_probabilities, dtype=float)
    if cumulative.ndim != 2 or cumulative.shape[1] != 4:
        raise ValueError("Ordered CTFE cumulative probabilities must have shape (n, 4).")
    cumulative = np.clip(cumulative, 0.0, 1.0)
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    probabilities = np.column_stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2] - cumulative[:, 3],
            cumulative[:, 3],
        ]
    )
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    return probabilities, cumulative


def _build_model(model_type: str, *, seed: int, multiclass: bool) -> object:
    if model_type == ORDINAL_LOGISTIC:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        C=0.5,
                        solver="lbfgs" if multiclass else "liblinear",
                        max_iter=500,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    if model_type == ORDINAL_CATBOOST:
        return CatBoostClassifier(
            iterations=150,
            depth=4,
            learning_rate=0.04,
            loss_function="MultiClass" if multiclass else "Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            random_seed=int(seed),
            thread_count=4,
        )
    raise ValueError(f"Unknown ordered CTFE model type: {model_type}")


def _fit_binary(frame: pd.DataFrame, target: pd.Series, *, model_type: str, feature_columns: Sequence[str], seed: int) -> object:
    positives = int(target.sum())
    if positives == 0 or positives == len(target):
        return ConstantBinaryHead(float(positives > 0))
    model = _build_model(model_type, seed=seed, multiclass=False)
    model.fit(frame[list(feature_columns)], target.astype(int).to_numpy())
    return model


def _fit_transition(frame: pd.DataFrame, target: pd.Series, *, model_type: str, feature_columns: Sequence[str], seed: int) -> object:
    if target.nunique() == 1:
        return ConstantMulticlassHead(str(target.iloc[0]))
    model = _build_model(model_type, seed=seed, multiclass=True)
    model.fit(frame[list(feature_columns)], target.astype(str).to_numpy())
    return model


def fit_ordered_transition_heads(
    training_frame: pd.DataFrame,
    *,
    model_type: str,
    feature_columns: Sequence[str] = ORDERED_FEATURES,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    _require_train_only(training_frame)
    _validate_features(training_frame, feature_columns)
    prepared = add_ordered_transition_targets(training_frame)
    heads: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for offset, target_name in enumerate(THRESHOLD_TARGETS):
        target = prepared[f"{target_name}_target"].astype(int)
        heads[target_name] = _fit_binary(
            prepared, target, model_type=model_type, feature_columns=feature_columns, seed=int(seed) + offset + 1
        )
        rows.append(
            {
                "head": target_name,
                "model_type": model_type,
                "sample_count": int(len(target)),
                "positive_count": int(target.sum()),
                "negative_count": int(len(target) - target.sum()),
            }
        )
    transition = prepared["transition_target"].astype(str)
    heads["transition"] = _fit_transition(
        prepared, transition, model_type=model_type, feature_columns=feature_columns, seed=int(seed) + 10
    )
    rows.append(
        {
            "head": "transition",
            "model_type": model_type,
            "sample_count": int(len(transition)),
            "positive_count": int(transition.eq("increase_bin").sum()),
            "negative_count": int(transition.ne("increase_bin").sum()),
        }
    )
    return heads, pd.DataFrame(rows)


def _aligned_multiclass_probability(model: object, frame: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(frame), dtype=float)
    classes = [str(item) for item in np.asarray(getattr(model, "classes_", labels), dtype=object)]
    aligned = np.zeros((len(frame), len(labels)), dtype=float)
    for index, label in enumerate(labels):
        if label in classes:
            aligned[:, index] = raw[:, classes.index(label)]
    return aligned


def score_ordered_transition_heads(
    frame: pd.DataFrame,
    heads: Mapping[str, object],
    *,
    feature_columns: Sequence[str] = ORDERED_FEATURES,
) -> pd.DataFrame:
    _validate_features(frame, feature_columns)
    result = add_ordered_transition_targets(frame).reset_index(drop=True)
    cumulative = np.column_stack(
        [
            np.asarray(heads[target].predict_proba(result[list(feature_columns)]), dtype=float)[:, 1]
            for target in THRESHOLD_TARGETS
        ]
    )
    probabilities, projected = recover_five_class_probabilities(cumulative)
    for index, target in enumerate(THRESHOLD_TARGETS):
        result[f"ordered_{target}_probability"] = projected[:, index]
    for index, label in enumerate(CTFE_DOSE_LABELS):
        result[f"ordered_prob_{label}"] = probabilities[:, index]
    transition = _aligned_multiclass_probability(heads["transition"], result[list(feature_columns)], TRANSITION_LABELS)
    for index, label in enumerate(TRANSITION_LABELS):
        result[f"transition_prob_{label}"] = transition[:, index]
    return result


def _normalise(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    probabilities = frame[list(columns)].astype(float).to_numpy()
    return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


def compose_ordered_transition_probabilities(
    frame: pd.DataFrame,
    *,
    transition_weight: float,
    control_weight: float,
    prediction_col: str = "ctfe_ordered_transition_prediction",
) -> pd.DataFrame:
    if not 0.0 <= float(transition_weight) <= 1.0 or not 0.0 <= float(control_weight) <= 1.0:
        raise ValueError("Ordered CTFE weights must be between zero and one.")
    required = set(ORDERED_PROBABILITY_COLUMNS + ENSEMBLE_PROBABILITY_COLUMNS)
    required.update(f"transition_prob_{label}" for label in TRANSITION_LABELS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Ordered CTFE scoring input missing columns: {missing}")
    result = frame.copy().reset_index(drop=True)
    if "current_fsh_class" not in result.columns:
        result["current_fsh_class"] = map_fsh_dose_class(result["current_fsh_daily_dose"])
    ordered = _normalise(result, ORDERED_PROBABILITY_COLUMNS)
    direction_probability = result[[f"transition_prob_{label}" for label in TRANSITION_LABELS]].astype(float).to_numpy()
    adjusted = ordered.copy()
    for row_index, current in enumerate(result["current_fsh_class"].astype(str)):
        current_index = LABEL_INDEX[current]
        for class_index in range(len(CTFE_DOSE_LABELS)):
            direction = 0 if class_index < current_index else 1 if class_index == current_index else 2
            adjusted[row_index, class_index] *= (
                (1.0 - float(transition_weight)) + float(transition_weight) * 3.0 * direction_probability[row_index, direction]
            )
    adjusted /= np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)
    control = _normalise(result, ENSEMBLE_PROBABILITY_COLUMNS)
    candidate = float(control_weight) * control + (1.0 - float(control_weight)) * adjusted
    candidate /= np.maximum(candidate.sum(axis=1, keepdims=True), 1e-12)
    for index, label in enumerate(CTFE_DOSE_LABELS):
        result[f"candidate_prob_{label}"] = candidate[:, index]
    result[prediction_col] = np.asarray(CTFE_DOSE_LABELS, dtype=object)[candidate.argmax(axis=1)]
    if "ctfe_stratified_ensemble_prediction" in result.columns:
        result["ordered_changed_prediction"] = result[prediction_col].astype(str).ne(
            result["ctfe_stratified_ensemble_prediction"].astype(str)
        )
    return result


def evaluate_ordered_frame(frame: pd.DataFrame, *, prediction_col: str, probability_prefix: str) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("Cannot evaluate an empty ordered CTFE frame.")
    true = frame["ctfe_next_fsh_dose_class"].astype(str).to_numpy()
    pred = frame[prediction_col].astype(str).to_numpy()
    probabilities = _normalise(frame, [f"{probability_prefix}{label}" for label in CTFE_DOSE_LABELS])
    individual = f1_score(true, pred, labels=CTFE_DOSE_LABELS, average=None, zero_division=0)
    distances = np.asarray([abs(LABEL_INDEX[a] - LABEL_INDEX[b]) for a, b in zip(true, pred)])
    metrics: dict[str, float | int] = {
        "sample_count": int(len(frame)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=CTFE_DOSE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=CTFE_DOSE_LABELS, average="weighted", zero_division=0)),
        "log_loss": float(log_loss([LABEL_INDEX[value] for value in true], probabilities, labels=list(range(len(CTFE_DOSE_LABELS))))),
        "adjacent_errors": int((distances == 1).sum()),
        "far_errors": int((distances >= 2).sum()),
    }
    for label, score in zip(CTFE_DOSE_LABELS, individual):
        metrics[f"f1_{label}"] = float(score)
    if "ordered_changed_prediction" in frame.columns:
        metrics["changed_predictions"] = int(frame["ordered_changed_prediction"].sum())
    return metrics


def evaluate_transition_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "transition_target" not in frame.columns:
        frame = add_ordered_transition_targets(frame)
    predicted = frame[[f"transition_prob_{label}" for label in TRANSITION_LABELS]].astype(float).to_numpy().argmax(axis=1)
    true = frame["transition_target"].astype(str).to_numpy()
    pred = np.asarray(TRANSITION_LABELS, dtype=object)[predicted]
    precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=TRANSITION_LABELS, zero_division=0)
    return pd.DataFrame(
        {"action": TRANSITION_LABELS, "precision": precision, "recall": recall, "f1": f1, "support": support.astype(int)}
    )


def search_ordered_fusion_parameters(
    frame: pd.DataFrame,
    *,
    selection_mask: np.ndarray,
    transition_weights: Iterable[float] = (0.0, 0.10, 0.25, 0.40),
    control_weights: Iterable[float] = (0.0, 0.25, 0.50, 0.75, 0.90, 1.0),
) -> dict[str, object]:
    mask = np.asarray(selection_mask, dtype=bool)
    if len(mask) != len(frame) or not mask.any():
        raise ValueError("Ordered CTFE selection requires aligned non-empty inner-tune rows.")
    tune = frame.loc[mask].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for transition_weight in transition_weights:
        for control_weight in control_weights:
            candidate = compose_ordered_transition_probabilities(
                tune, transition_weight=float(transition_weight), control_weight=float(control_weight)
            )
            metrics = evaluate_ordered_frame(
                candidate, prediction_col="ctfe_ordered_transition_prediction", probability_prefix="candidate_prob_"
            )
            rows.append(
                {
                    "transition_weight": float(transition_weight),
                    "control_weight": float(control_weight),
                    **{f"tune_{key}": value for key, value in metrics.items()},
                    "parameter_selection_split": "inner_tune",
                    "test_used_for_selection": False,
                }
            )
    search = pd.DataFrame(rows).sort_values(
        ["tune_weighted_f1", "tune_accuracy", "tune_macro_f1", "tune_adjacent_errors", "tune_log_loss"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    top = search.iloc[0].to_dict()
    top["search_results"] = search
    return top


def assess_ordered_promotion(metrics: pd.DataFrame, *, scope: str, min_improved_seeds: int) -> dict[str, object]:
    valid = metrics[(metrics["scope"].astype(str).eq(scope)) & (metrics["split"].astype(str).eq("valid"))].copy()
    deltas: list[dict[str, object]] = []
    for seed in sorted(valid["seed"].unique()):
        block = valid[valid["seed"].eq(seed)].set_index("variant")
        if CONTROL_VARIANT not in block.index or CANDIDATE_VARIANT not in block.index:
            continue
        control = block.loc[CONTROL_VARIANT]
        candidate = block.loc[CANDIDATE_VARIANT]
        passed = bool(
            float(candidate["weighted_f1"]) > float(control["weighted_f1"])
            and float(candidate["accuracy"]) >= float(control["accuracy"])
            and float(candidate["macro_f1"]) >= float(control["macro_f1"]) - 0.005
            and float(candidate["log_loss"]) <= float(control["log_loss"]) + 0.002
            and int(candidate["adjacent_errors"]) < int(control["adjacent_errors"])
            and float(candidate["f1_stop"]) >= float(control["f1_stop"]) - 0.02
            and float(candidate["f1_high"]) >= float(control["f1_high"]) - 0.02
        )
        deltas.append(
            {
                "seed": int(seed),
                "passed": passed,
                "weighted_f1_delta": float(candidate["weighted_f1"] - control["weighted_f1"]),
                "accuracy_delta": float(candidate["accuracy"] - control["accuracy"]),
                "macro_f1_delta": float(candidate["macro_f1"] - control["macro_f1"]),
                "log_loss_delta": float(candidate["log_loss"] - control["log_loss"]),
                "adjacent_errors_delta": int(candidate["adjacent_errors"] - control["adjacent_errors"]),
            }
        )
    delta_frame = pd.DataFrame(deltas)
    required = 1 if scope == "official" else int(min_improved_seeds)
    mean_positive = bool(
        not delta_frame.empty
        and float(delta_frame["weighted_f1_delta"].mean()) > 0.0
        and float(delta_frame["adjacent_errors_delta"].mean()) < 0.0
    )
    accepted = bool(len(delta_frame) >= required and int(delta_frame["passed"].sum()) >= required and mean_positive)
    return {
        "scope": scope,
        "accepted": accepted,
        "selection_split": "valid",
        "test_used_for_selection": False,
        "required_passing_seeds": required,
        "passing_seeds": int(delta_frame["passed"].sum()) if not delta_frame.empty else 0,
        "mean_weighted_f1_delta": float(delta_frame["weighted_f1_delta"].mean()) if not delta_frame.empty else np.nan,
        "mean_adjacent_errors_delta": float(delta_frame["adjacent_errors_delta"].mean()) if not delta_frame.empty else np.nan,
        "per_seed": deltas,
    }
