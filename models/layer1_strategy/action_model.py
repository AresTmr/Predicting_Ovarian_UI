from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from tqdm.auto import tqdm

from models.common.preprocessing import build_preprocessor, select_feature_columns

ACTION_LABELS = ["increase", "maintain", "decrease"]
ACTION_TO_ID = {label: idx for idx, label in enumerate(ACTION_LABELS)}
ID_TO_ACTION = {idx: label for label, idx in ACTION_TO_ID.items()}
DEFAULT_ACTION_THRESHOLD = 37.5
DEFAULT_SEED = 20260424

ID_COLUMNS = {
    "visit_uid",
    "canonical_visit_key",
    "cycle_uid",
    "cycle_id",
    "art_id",
    "monitoring_date",
    "cohort_flag_used",
}
LAYER1_ACTION_LEAKAGE_COLUMNS = {
    "next_gn_dose",
    "next_fsh_dose",
    "next_fsh_daily_dose",
    "next_lh_daily_dose",
    "next_lh_dose",
    "next_hmg_daily_dose",
    "next_hmg_dose",
    "next_lh_like_hmg_dose",
    "next_lh_like_hmg_daily_dose",
    "next_monitoring_order",
    "next_monitoring_date",
    "next_visit_interval_days",
    "observed_gn_dose_delta",
    "observed_action_label",
    "combined_gn_delta",
    "fsh_delta",
    "lh_delta",
    "hmg_delta",
    "lh_like_delta",
    "gn_action",
    "fsh_action",
    "lh_action",
    "hmg_action",
    "lh_like_action",
    "combined_gn_action",
    "has_next_visit",
    "strategy_eligible_flag",
    "split",
}
FUTURE_PREFIXES = ("target_", "next_", "observed_")


@dataclass(frozen=True)
class Layer1ActionTrainingResult:
    run_id: str
    output_dir: Path
    target: str
    best_model: str
    best_valid_macro_f1: float
    best_test_macro_f1: float | None
    metrics_path: Path
    predictions_path: Path


def action_from_delta(delta: float | int | None, threshold: float = DEFAULT_ACTION_THRESHOLD) -> str | pd.NA:
    if pd.isna(delta):
        return pd.NA
    value = float(delta)
    if value >= threshold:
        return "increase"
    if value <= -threshold:
        return "decrease"
    return "maintain"


def _first_existing(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, offset: float = 1.0) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / (den + offset)


def augment_layer1_action_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "current_gn_dose" in df.columns and "previous_gn_dose" in df.columns:
        df["gn_dose_change_vs_previous"] = pd.to_numeric(df["current_gn_dose"], errors="coerce") - pd.to_numeric(
            df["previous_gn_dose"], errors="coerce"
        )
        df["gn_dose_ratio_vs_previous"] = _safe_ratio(df["current_gn_dose"], df["previous_gn_dose"], offset=1.0)
    if "current_gn_dose" in df.columns and "total_follicle_count" in df.columns:
        df["current_gn_per_follicle"] = _safe_ratio(df["current_gn_dose"], df["total_follicle_count"], offset=1.0)
    if "current_gn_dose" in df.columns and "mature_follicle_count" in df.columns:
        df["current_gn_per_mature_follicle"] = _safe_ratio(df["current_gn_dose"], df["mature_follicle_count"], offset=1.0)
    if "current_e2" in df.columns and "current_gn_dose" in df.columns:
        df["current_e2_per_gn"] = _safe_ratio(df["current_e2"], df["current_gn_dose"], offset=1.0)
    if "current_lh_like_hmg_daily_dose" in df.columns and "current_gn_dose" in df.columns:
        df["current_lh_like_share"] = _safe_ratio(df["current_lh_like_hmg_daily_dose"], df["current_gn_dose"], offset=1.0)
    return df

def create_gn_action_labels(frame: pd.DataFrame, threshold: float = DEFAULT_ACTION_THRESHOLD) -> pd.DataFrame:
    """Create next-visit Gn adjustment direction labels from current and next doses.

    The split FSH and LH-like/HMG labels are emitted when matching next-dose columns exist.
    The combined label is always generated from current_gn_dose and next_gn_dose.
    """
    df = frame.copy()
    required = {"current_gn_dose", "next_gn_dose"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for combined_gn_action: {missing}")

    df["combined_gn_delta"] = pd.to_numeric(df["next_gn_dose"], errors="coerce") - pd.to_numeric(
        df["current_gn_dose"], errors="coerce"
    )
    df["combined_gn_action"] = df["combined_gn_delta"].map(lambda value: action_from_delta(value, threshold))
    df["gn_action"] = df["combined_gn_action"]

    current_fsh = _first_existing(df, ["current_fsh_daily_dose", "current_fsh_dose"])
    next_fsh = _first_existing(df, ["next_fsh_daily_dose", "next_fsh_dose"])
    if current_fsh and next_fsh:
        df["fsh_delta"] = pd.to_numeric(df[next_fsh], errors="coerce") - pd.to_numeric(df[current_fsh], errors="coerce")
        df["fsh_action"] = df["fsh_delta"].map(lambda value: action_from_delta(value, threshold))
    else:
        df["fsh_delta"] = pd.NA
        df["fsh_action"] = pd.NA

    current_lh = _first_existing(df, ["current_lh_daily_dose", "current_lh_dose"])
    next_lh = _first_existing(df, ["next_lh_daily_dose", "next_lh_dose"])
    if current_lh and next_lh:
        df["lh_delta"] = pd.to_numeric(df[next_lh], errors="coerce") - pd.to_numeric(df[current_lh], errors="coerce")
        df["lh_action"] = df["lh_delta"].map(lambda value: action_from_delta(value, threshold))
    else:
        df["lh_delta"] = pd.NA
        df["lh_action"] = pd.NA

    current_hmg = _first_existing(df, ["current_hmg_daily_dose", "current_hmg_dose"])
    next_hmg = _first_existing(df, ["next_hmg_daily_dose", "next_hmg_dose"])
    if current_hmg and next_hmg:
        df["hmg_delta"] = pd.to_numeric(df[next_hmg], errors="coerce") - pd.to_numeric(df[current_hmg], errors="coerce")
        df["hmg_action"] = df["hmg_delta"].map(lambda value: action_from_delta(value, threshold))
    else:
        df["hmg_delta"] = pd.NA
        df["hmg_action"] = pd.NA

    current_lh_like = _first_existing(df, ["current_lh_like_hmg_daily_dose", "current_lh_like_hmg_dose"])
    next_lh_like = _first_existing(df, ["next_lh_like_hmg_daily_dose", "next_lh_like_hmg_dose"])
    if current_lh_like and next_lh_like:
        df["lh_like_delta"] = pd.to_numeric(df[next_lh_like], errors="coerce") - pd.to_numeric(
            df[current_lh_like], errors="coerce"
        )
        df["lh_like_action"] = df["lh_like_delta"].map(lambda value: action_from_delta(value, threshold))
    else:
        df["lh_like_delta"] = pd.NA
        df["lh_like_action"] = pd.NA

    return df


def select_layer1_action_feature_columns(frame: pd.DataFrame, target: str = "combined_gn_action") -> list[str]:
    frame = augment_layer1_action_features(frame)
    excluded = set(ID_COLUMNS) | set(LAYER1_ACTION_LEAKAGE_COLUMNS) | {target}
    feature_columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if any(column.startswith(prefix) for prefix in FUTURE_PREFIXES):
            continue
        if column.endswith("_action"):
            continue
        feature_columns.append(column)
    return feature_columns


def evaluate_action_classifier(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    y_proba: pd.DataFrame | np.ndarray | None = None,
    labels: Sequence[str] = ACTION_LABELS,
) -> dict[str, Any]:
    y_true_s = pd.Series(y_true, dtype="object")
    y_pred_s = pd.Series(y_pred, dtype="object")
    report = classification_report(y_true_s, y_pred_s, labels=list(labels), output_dict=True, zero_division=0)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_s, y_pred_s)),
        "macro_f1": float(f1_score(y_true_s, y_pred_s, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_s, y_pred_s, labels=list(labels), average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true_s, y_pred_s, labels=list(labels), average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true_s, y_pred_s, labels=list(labels), average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(y_true_s, y_pred_s, labels=list(labels), average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(y_true_s, y_pred_s, labels=list(labels), average="weighted", zero_division=0)),
    }
    for label in labels:
        payload = report.get(label, {})
        metrics[f"precision_{label}"] = float(payload.get("precision", 0.0))
        metrics[f"recall_{label}"] = float(payload.get("recall", 0.0))
        metrics[f"f1_{label}"] = float(payload.get("f1-score", 0.0))
        metrics[f"support_{label}"] = int(payload.get("support", 0))
    if y_proba is not None:
        proba_df = _coerce_probability_frame(y_proba, labels)
        for label in labels:
            metrics[f"mean_pred_prob_{label}"] = float(proba_df[label].mean()) if label in proba_df else np.nan
    return metrics


def _coerce_probability_frame(y_proba: pd.DataFrame | np.ndarray, labels: Sequence[str] = ACTION_LABELS) -> pd.DataFrame:
    if isinstance(y_proba, pd.DataFrame):
        return y_proba.reindex(columns=list(labels), fill_value=0.0)
    arr = np.asarray(y_proba, dtype=float)
    if arr.ndim != 2:
        raise ValueError("y_proba must be a 2D array or DataFrame")
    if arr.shape[1] != len(labels):
        padded = np.zeros((arr.shape[0], len(labels)), dtype=float)
        padded[:, : min(arr.shape[1], len(labels))] = arr[:, : min(arr.shape[1], len(labels))]
        arr = padded
    return pd.DataFrame(arr, columns=list(labels))


def _load_split_frame(frame: pd.DataFrame, split_manifest_path: str | Path | None, seed: int) -> pd.Series:
    if split_manifest_path:
        path = Path(split_manifest_path)
        if path.exists():
            manifest = pd.read_csv(path)
            if {"cycle_uid", "split"}.issubset(manifest.columns) and "cycle_uid" in frame.columns:
                split_map = manifest.drop_duplicates("cycle_uid").set_index("cycle_uid")["split"]
                split = frame["cycle_uid"].map(split_map)
                if split.notna().any():
                    return split.fillna("train")
    rng = np.random.default_rng(seed)
    cycles = pd.Series(frame["cycle_uid"].dropna().unique()) if "cycle_uid" in frame.columns else pd.Series(range(len(frame)))
    shuffled = cycles.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    test = set(shuffled.iloc[: max(1, int(n * 0.15))])
    valid = set(shuffled.iloc[max(1, int(n * 0.15)) : max(2, int(n * 0.30))])
    if "cycle_uid" in frame.columns:
        return frame["cycle_uid"].map(lambda value: "test" if value in test else "valid" if value in valid else "train")
    return pd.Series(np.where(rng.random(len(frame)) < 0.15, "test", "train"), index=frame.index)


def _build_action_estimator(model_name: str, seed: int) -> Any:
    model_name = model_name.lower()
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=len(ACTION_LABELS),
            n_estimators=240,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(ACTION_LABELS),
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=220,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=seed,
            verbose=False,
        )
    raise ValueError(f"Unsupported Layer 1 action model: {model_name}")


def _fit_action_estimator(
    estimator: Any,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    sample_weight: np.ndarray | None,
) -> Any:
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if model_name == "lightgbm":
        fit_kwargs.update({"eval_set": [(x_valid, y_valid)], "eval_metric": "multi_logloss"})
    elif model_name == "xgboost":
        fit_kwargs.update({"eval_set": [(x_valid, y_valid)], "verbose": False})
    elif model_name == "catboost":
        fit_kwargs.update({"eval_set": (x_valid, y_valid)})
    estimator.fit(x_train, y_train, **fit_kwargs)
    return estimator


def _predict_labels_and_proba(estimator: Any, x: np.ndarray) -> tuple[list[str], pd.DataFrame]:
    pred_ids = np.asarray(estimator.predict(x)).reshape(-1).astype(int)
    pred_labels = [ID_TO_ACTION.get(int(idx), "maintain") for idx in pred_ids]
    if hasattr(estimator, "predict_proba"):
        raw_proba = np.asarray(estimator.predict_proba(x), dtype=float)
    else:
        raw_proba = np.zeros((len(pred_labels), len(ACTION_LABELS)), dtype=float)
        for row_idx, label in enumerate(pred_labels):
            raw_proba[row_idx, ACTION_TO_ID[label]] = 1.0
    proba = _coerce_probability_frame(raw_proba, ACTION_LABELS)
    return pred_labels, proba


def _feature_names(preprocessor: Any, fallback: Sequence[str]) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return list(fallback)


def _top_importance_frame(estimator: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(np.asarray(estimator.coef_, dtype=float)), axis=0)
    else:
        values = np.zeros(len(feature_names), dtype=float)
    n = min(len(values), len(feature_names))
    frame = pd.DataFrame({"feature": list(feature_names)[:n], "importance": values[:n]})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_action_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    output_path: str | Path,
    labels: Sequence[str] = ACTION_LABELS,
    title: str = "Layer 1 Gn adjustment action prediction",
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("True action")
    ax.set_xticks(range(len(labels)), labels=list(labels), rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=list(labels))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return cm_df


def _plot_prediction_distribution(y_true: Sequence[str], y_pred: Sequence[str], output_path: str | Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    true_counts = pd.Series(y_true).value_counts(normalize=True).reindex(ACTION_LABELS, fill_value=0.0)
    pred_counts = pd.Series(y_pred).value_counts(normalize=True).reindex(ACTION_LABELS, fill_value=0.0)
    dist = pd.DataFrame({"action": ACTION_LABELS, "true_share": true_counts.values, "pred_share": pred_counts.values})
    x = np.arange(len(ACTION_LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.bar(x - width / 2, dist["true_share"], width, label="True")
    ax.bar(x + width / 2, dist["pred_share"], width, label="Predicted")
    ax.set_xticks(x, ACTION_LABELS)
    ax.set_ylim(0, max(0.05, float(dist[["true_share", "pred_share"]].max().max()) * 1.25))
    ax.set_ylabel("Share")
    ax.set_title("Layer 1 Gn adjustment action prediction distribution")
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return dist


def _plot_importance_bar(importance: pd.DataFrame, output_path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#0ea5e9")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def explain_layer1_action_model_shap(
    estimator: Any,
    x_sample: np.ndarray,
    feature_names: Sequence[str],
    output_path: str | Path,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    output_path = Path(output_path)
    csv_path = Path(csv_path) if csv_path else output_path.with_suffix(".csv")
    try:
        import shap

        dense = x_sample.toarray() if hasattr(x_sample, "toarray") else np.asarray(x_sample)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(dense)
        if isinstance(shap_values, list):
            stacked = np.stack([np.asarray(values) for values in shap_values], axis=0)
            importance_values = np.mean(np.abs(stacked), axis=(0, 1))
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3 and arr.shape[-1] == len(ACTION_LABELS):
                importance_values = np.mean(np.abs(arr), axis=(0, 2))
            elif arr.ndim == 3:
                importance_values = np.mean(np.abs(arr), axis=(0, 1))
            else:
                importance_values = np.mean(np.abs(arr), axis=0)
        n = min(len(feature_names), len(importance_values))
        frame = pd.DataFrame({"feature": list(feature_names)[:n], "importance": importance_values[:n]})
        frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
        frame.to_csv(csv_path, index=False)
        _plot_importance_bar(frame, output_path, "Layer 1 Gn adjustment action prediction - SHAP top features")
        return frame
    except Exception as exc:
        frame = _top_importance_frame(estimator, feature_names)
        frame["explain_method"] = "feature_importance_fallback"
        frame["shap_error"] = str(exc)
        frame.to_csv(csv_path, index=False)
        _plot_importance_bar(frame, output_path, "Layer 1 Gn adjustment action prediction - feature importance fallback")
        return frame


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def train_layer1_action_model(
    input_path: str | Path = "data/processed/layer1_strategy_dataset.csv",
    split_manifest_path: str | Path | None = "data/splits/split_manifest_v1.csv",
    output_root: str | Path = "models/artifacts",
    target: str = "combined_gn_action",
    threshold: float = DEFAULT_ACTION_THRESHOLD,
    model_names: Sequence[str] = ("lightgbm", "xgboost", "catboost"),
    run_id: str | None = None,
    seed: int = DEFAULT_SEED,
) -> Layer1ActionTrainingResult:
    run_id = run_id or f"phase8_layer1_action_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_root) / run_id / "layer1_action"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    raw = augment_layer1_action_features(raw)
    labeled = create_gn_action_labels(raw, threshold=threshold)
    if target not in labeled.columns:
        raise ValueError(f"Unknown Layer 1 action target: {target}")
    data = labeled[labeled[target].isin(ACTION_LABELS)].copy()
    if "has_next_visit" in data.columns:
        data = data[data["has_next_visit"].astype(bool)].copy()
    if "strategy_eligible_flag" in data.columns:
        data = data[data["strategy_eligible_flag"].astype(bool)].copy()
    if data.empty:
        raise ValueError("No eligible samples for Layer 1 action training")

    data["split"] = _load_split_frame(data, split_manifest_path, seed).values
    data = augment_layer1_action_features(data)
    feature_columns = select_layer1_action_feature_columns(data, target=target)
    feature_bundle = select_feature_columns(data, feature_columns)
    if not feature_bundle.feature_columns:
        raise ValueError("No non-leakage Layer 1 action features were selected")

    train_df = data[data["split"] == "train"].copy()
    valid_df = data[data["split"] == "valid"].copy()
    test_df = data[data["split"] == "test"].copy()
    if valid_df.empty or test_df.empty:
        raise ValueError("Split manifest must provide non-empty valid and test splits")

    preprocessor = build_preprocessor(feature_bundle)
    x_train = preprocessor.fit_transform(train_df[feature_bundle.feature_columns])
    x_valid = preprocessor.transform(valid_df[feature_bundle.feature_columns])
    x_test = preprocessor.transform(test_df[feature_bundle.feature_columns])
    y_train = train_df[target].map(ACTION_TO_ID).astype(int).to_numpy()
    y_valid = valid_df[target].map(ACTION_TO_ID).astype(int).to_numpy()
    y_test = test_df[target].map(ACTION_TO_ID).astype(int).to_numpy()
    y_train_labels = train_df[target].astype(str).to_numpy()
    y_valid_labels = valid_df[target].astype(str).to_numpy()
    y_test_labels = test_df[target].astype(str).to_numpy()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    transformed_feature_names = _feature_names(preprocessor, feature_bundle.feature_columns)

    class_distribution = (
        data.groupby(["split", target]).size().rename("n").reset_index().sort_values(["split", target])
    )
    class_distribution.to_csv(output_dir / "layer1_gn_action_class_distribution.csv", index=False)

    metrics_rows: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    best_model_name: str | None = None
    best_valid_macro_f1 = -1.0

    for model_name in tqdm(list(model_names), desc="layer1 action models", unit="model"):
        model_name = model_name.lower()
        try:
            estimator = _build_action_estimator(model_name, seed)
            estimator = _fit_action_estimator(estimator, model_name, x_train, y_train, x_valid, y_valid, sample_weight)
        except Exception as exc:
            metrics_rows.append({"model_name": model_name, "split": "train", "status": "failed", "reason": str(exc)})
            continue
        trained[model_name] = estimator
        for split_name, x_split, y_split_labels in [
            ("train", x_train, y_train_labels),
            ("valid", x_valid, y_valid_labels),
            ("test", x_test, y_test_labels),
        ]:
            pred_labels, proba = _predict_labels_and_proba(estimator, x_split)
            metric_values = evaluate_action_classifier(y_split_labels, pred_labels, proba)
            metrics_rows.append({"model_name": model_name, "split": split_name, "status": "trained", **metric_values})
        valid_metric = [row for row in metrics_rows if row.get("model_name") == model_name and row.get("split") == "valid"][-1][
            "macro_f1"
        ]
        if float(valid_metric) > best_valid_macro_f1:
            best_valid_macro_f1 = float(valid_metric)
            best_model_name = model_name

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "layer1_gn_action_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if not best_model_name:
        raise RuntimeError(f"No Layer 1 action model trained successfully. See {metrics_path}")

    best_estimator = trained[best_model_name]
    best_test_pred, best_test_proba = _predict_labels_and_proba(best_estimator, x_test)
    test_metrics = evaluate_action_classifier(y_test_labels, best_test_pred, best_test_proba)
    confusion_df = plot_action_confusion_matrix(
        y_test_labels,
        best_test_pred,
        output_dir / "layer1_gn_action_confusion_matrix.png",
    )
    confusion_df.to_csv(output_dir / "layer1_gn_action_confusion_matrix.csv")
    distribution_df = _plot_prediction_distribution(
        y_test_labels,
        best_test_pred,
        output_dir / "layer1_gn_action_prediction_distribution.png",
    )
    distribution_df.to_csv(output_dir / "layer1_gn_action_prediction_distribution.csv", index=False)
    importance_df = _top_importance_frame(best_estimator, transformed_feature_names)
    importance_df.to_csv(output_dir / "layer1_gn_action_feature_importance.csv", index=False)
    _plot_importance_bar(
        importance_df,
        output_dir / "layer1_gn_action_feature_importance.png",
        "Layer 1 Gn adjustment action prediction - feature importance",
    )
    sample_n = min(300, x_test.shape[0])
    explain_layer1_action_model_shap(
        best_estimator,
        x_test[:sample_n],
        transformed_feature_names,
        output_dir / "layer1_gn_action_shap_top_features.png",
        output_dir / "layer1_gn_action_shap_top_features.csv",
    )

    predictions = test_df[[column for column in ["cycle_uid", "visit_uid", "monitoring_order", "current_gn_dose", "next_gn_dose"] if column in test_df.columns]].copy()
    predictions["target"] = y_test_labels
    predictions["prediction"] = best_test_pred
    for label in ACTION_LABELS:
        predictions[f"prob_{label}"] = best_test_proba[label].to_numpy()
    predictions_path = output_dir / "layer1_gn_action_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    bundle = {
        "estimator": best_estimator,
        "preprocessor": preprocessor,
        "feature_names": feature_bundle.feature_columns,
        "transformed_feature_names": transformed_feature_names,
        "target": target,
        "action_labels": ACTION_LABELS,
        "threshold": threshold,
        "model_name": best_model_name,
        "task_type": "multiclass_classification",
    }
    joblib.dump(bundle, output_dir / "layer1_gn_action_best_bundle.joblib")

    summary = {
        "run_id": run_id,
        "target": target,
        "threshold": threshold,
        "best_model": best_model_name,
        "best_valid_macro_f1": best_valid_macro_f1,
        "best_test_macro_f1": test_metrics.get("macro_f1"),
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "feature_count": len(feature_bundle.feature_columns),
        "sample_count": int(len(data)),
    }
    (output_dir / "run_summary.json").write_text(
        pd.Series(summary).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Layer 1 Gn adjustment action prediction",
        "",
        f"- run_id: `{run_id}`",
        f"- target: `{target}`",
        f"- threshold: `{threshold}` IU",
        f"- best_model: `{best_model_name}`",
        f"- valid_macro_f1: `{best_valid_macro_f1:.4f}`",
        f"- test_macro_f1: `{float(test_metrics.get('macro_f1', 0.0)):.4f}`",
        f"- feature_count: `{len(feature_bundle.feature_columns)}`",
        f"- sample_count: `{len(data)}`",
        "",
        "Main outputs:",
        "- `layer1_gn_action_metrics.csv`",
        "- `layer1_gn_action_predictions.csv`",
        "- `layer1_gn_action_confusion_matrix.png`",
        "- `layer1_gn_action_shap_top_features.png`",
        "- `layer1_gn_action_feature_importance.png`",
    ]
    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return Layer1ActionTrainingResult(
        run_id=run_id,
        output_dir=output_dir,
        target=target,
        best_model=best_model_name,
        best_valid_macro_f1=best_valid_macro_f1,
        best_test_macro_f1=float(test_metrics.get("macro_f1", np.nan)),
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
