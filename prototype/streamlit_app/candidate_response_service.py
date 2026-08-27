from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import shap
try:
    from .temporal_candidate_response_service import available as temporal_candidate_response_available, predict as predict_temporal_candidate_response
except Exception:
    try:
        from temporal_candidate_response_service import available as temporal_candidate_response_available, predict as predict_temporal_candidate_response
    except Exception:
        temporal_candidate_response_available = None
        predict_temporal_candidate_response = None

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models" / "candidate_dose_response"
CURRENT_RUN_POINTER = MODEL_ROOT / "current_run.txt"
CURRENT_EFFICACY_RUN_POINTER = MODEL_ROOT / "current_efficacy_run.txt"
DEFAULT_EFFICACY_RUN_ID = "candidate_dose_response_v1_20260702_01"
TASKS = ("layer2_oocytes", "layer2_mii", "layer2_ohss")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_divide(numerator: Any, denominator: Any) -> float:
    numerator_value = _to_float(numerator, np.nan)
    denominator_value = _to_float(denominator, np.nan)
    if not np.isfinite(numerator_value) or not np.isfinite(denominator_value) or abs(denominator_value) < 1e-12:
        return float("nan")
    return float(numerator_value / denominator_value)


def _clip_probability(value: Any) -> float:
    return float(np.clip(_to_float(value, 0.0), 0.0, 1.0))


def _bound_count_prediction(value: Any, total_follicles: Any | None = None) -> float:
    prediction = max(0.0, _to_float(value, 0.0))
    follicle_cap = _to_float(total_follicles, np.nan)
    if np.isfinite(follicle_cap) and follicle_cap >= 0.0:
        prediction = min(prediction, follicle_cap)
    return float(prediction)


def current_run_id(pointer: Path = CURRENT_RUN_POINTER) -> str:
    if not pointer.exists():
        raise FileNotFoundError(f"Candidate dose-response run pointer not found: {pointer}")
    run_id = pointer.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"Candidate dose-response run pointer is empty: {pointer}")
    return run_id


def current_efficacy_run_id() -> str:
    if not CURRENT_EFFICACY_RUN_POINTER.exists():
        return DEFAULT_EFFICACY_RUN_ID
    run_id = CURRENT_EFFICACY_RUN_POINTER.read_text(encoding="utf-8").strip()
    return run_id or DEFAULT_EFFICACY_RUN_ID


@lru_cache(maxsize=4)
def candidate_response_prediction_baselines(run_id: str | None = None) -> dict[str, float]:
    """Return selected-model mean test predictions for outcome-card baselines."""
    selected_run = run_id or current_efficacy_run_id()
    summary_path = MODEL_ROOT / selected_run / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Candidate response run summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baselines: dict[str, float] = {}
    for row in summary.get("selected_models", []):
        task = str(row.get("task", ""))
        value = _to_float(row.get("test_mean_predicted"), np.nan)
        if task in TASKS and np.isfinite(value):
            baselines[task] = float(value)
    return baselines


@lru_cache(maxsize=4)
def load_candidate_response_bundle(run_id: str | None = None) -> dict[str, Any]:
    selected_run = run_id or current_run_id()
    path = MODEL_ROOT / selected_run / "bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Candidate dose-response bundle not found: {path}")
    bundle = dict(joblib.load(path))
    bundle["run_id"] = selected_run
    bundle["bundle_path"] = str(path)
    return bundle


def candidate_response_available() -> tuple[bool, str]:
    try:
        load_candidate_response_bundle()
        load_candidate_response_bundle(current_efficacy_run_id())
    except Exception as exc:
        return False, str(exc)
    return True, ""


def build_candidate_response_features(
    snapshot: pd.Series | Mapping[str, Any],
    *,
    fsh_dose: float,
    lh_dose: float,
    hmg_dose: float,
) -> pd.Series:
    """Create an observational candidate-plan feature row for outcome scoring.

    The original current/reference dose fields are kept as state variables. Candidate
    fields represent the plan being scored and are the features trained by the V1
    candidate-response model.
    """
    row = pd.Series(snapshot).copy()
    current_fsh = _to_float(row.get("current_fsh_daily_dose"), 0.0)
    current_lh = _to_float(row.get("current_lh_daily_dose"), 0.0)
    current_hmg = _to_float(row.get("current_hmg_daily_dose"), 0.0)
    current_total = _to_float(row.get("current_gn_dose"), current_fsh + current_lh + current_hmg)

    fsh = max(_to_float(fsh_dose), 0.0)
    lh = max(_to_float(lh_dose), 0.0)
    hmg = max(_to_float(hmg_dose), 0.0)
    lh_like = lh + hmg
    total = fsh + lh_like
    total_follicles = _to_float(row.get("total_follicle_count"), np.nan)
    mature_follicles = _to_float(row.get("mature_follicle_count"), np.nan)
    current_e2 = _to_float(row.get("current_e2"), np.nan)

    row["candidate_fsh_daily_dose"] = fsh
    row["candidate_lh_daily_dose"] = lh
    row["candidate_hmg_daily_dose"] = hmg
    row["candidate_lh_like_hmg_daily_dose"] = lh_like
    row["candidate_total_gn_dose"] = total
    row["candidate_fsh_delta"] = fsh - current_fsh
    row["candidate_lh_delta"] = lh - current_lh
    row["candidate_hmg_delta"] = hmg - current_hmg
    row["candidate_lh_like_delta"] = lh_like - (current_lh + current_hmg)
    row["candidate_total_gn_delta"] = total - current_total
    row["candidate_gn_per_total_follicle"] = _safe_divide(total, total_follicles)
    row["candidate_gn_per_mature_follicle"] = _safe_divide(total, mature_follicles)
    row["candidate_e2_per_gn"] = _safe_divide(current_e2, total)
    row["candidate_fsh_share_gn"] = _safe_divide(fsh, total)
    row["candidate_lh_like_share_gn"] = _safe_divide(lh_like, total)
    row["candidate_total_to_current_ratio"] = _safe_divide(total, current_total)
    return row


def _task_bundles(bundle_or_tasks: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if "tasks" in bundle_or_tasks and isinstance(bundle_or_tasks["tasks"], Mapping):
        return bundle_or_tasks["tasks"]
    return bundle_or_tasks  # tests pass task bundles directly


def _feature_names_from_bundle(bundle: Mapping[str, Any]) -> list[str]:
    preprocessor = bundle["preprocessor"]
    if hasattr(preprocessor, "feature_names_in_"):
        return [str(name) for name in list(preprocessor.feature_names_in_)]
    explicit = bundle.get("feature_columns") or bundle.get("feature_names")
    if explicit:
        return [str(name) for name in explicit]
    raise ValueError("Candidate response task bundle has no feature contract")


def _prepare_frame(row: pd.Series, bundle: Mapping[str, Any]) -> pd.DataFrame:
    feature_names = _feature_names_from_bundle(bundle)
    return row.to_frame().T.reindex(columns=feature_names)


def _calibrated_probability(raw: float, bundle: Mapping[str, Any]) -> float:
    value = float(raw)
    calibration = bundle.get("probability_calibration") or {}
    if calibration.get("method") == "isotonic":
        x = np.asarray(calibration.get("x_thresholds", []), dtype=float)
        y = np.asarray(calibration.get("y_thresholds", []), dtype=float)
        if len(x) >= 2 and len(x) == len(y):
            value = float(np.interp(value, x, y))
    elif calibration.get("method") == "sigmoid":
        coef = float(calibration.get("coef", 1.0))
        intercept = float(calibration.get("intercept", 0.0))
        value = float(1.0 / (1.0 + np.exp(-((coef * value) + intercept))))
    return _clip_probability(value)


def _predict_task(row: pd.Series, bundle: Mapping[str, Any]) -> float:
    frame = _prepare_frame(row, bundle)
    matrix = bundle["preprocessor"].transform(frame)
    estimator = bundle["estimator"]
    task_type = str(bundle.get("task_type", "regression"))
    if task_type == "classification":
        if hasattr(estimator, "predict_proba"):
            proba = np.asarray(estimator.predict_proba(matrix), dtype=float)
            raw = float(proba[0, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.reshape(-1)[0])
        else:
            raw = float(np.asarray(estimator.predict(matrix)).reshape(-1)[0])
        return _calibrated_probability(raw, bundle)
    prediction = estimator.predict(matrix)
    return float(np.asarray(prediction).reshape(-1)[0])


OUTCOME_EXPLANATION_SPECS: tuple[tuple[str, str, str, float], ...] = (
    ("total_follicle_count", "\u603b\u5375\u6ce1\u6570", "\u4e2a", 12.0),
    ("mature_follicle_count", "\u226514 mm \u5375\u6ce1\u6570", "\u4e2a", 4.0),
    ("current_e2", "\u8840\u6e05 E2", "pg/mL", 1200.0),
    ("current_lh", "\u8840\u6e05 LH", "IU/L", 4.0),
    ("current_p", "\u8840\u6e05 P", "ng/mL", 0.8),
    ("max_follicle_diameter", "\u6700\u5927\u5375\u6ce1\u76f4\u5f84", "mm", 16.0),
    ("mean_follicle_diameter", "\u5e73\u5747\u5375\u6ce1\u76f4\u5f84", "mm", 12.0),
    ("candidate_total_gn_dose", "\u5019\u9009\u603b Gn \u5242\u91cf", "IU/\u5929", 225.0),
    ("candidate_gn_per_total_follicle", "\u5019\u9009\u5355\u4f4d\u5375\u6ce1 Gn", "", 18.0),
    ("candidate_e2_per_gn", "\u5019\u9009\u5355\u4f4d Gn E2", "", 6.0),
    ("amh", "AMH", "ng/mL", 2.0),
    ("afc", "AFC", "\u4e2a", 12.0),
    ("age", "\u5e74\u9f84", "\u5c81", 32.0),
    ("bmi", "BMI", "kg/m\u00b2", 22.0),
    ("gn_day", "\u4fc3\u6392\u65e5", "\u5929", 8.0),
)


def _format_explanation_value(value: Any, unit: str) -> str:
    try:
        number = float(value)
    except Exception:
        return "--"
    if not np.isfinite(number):
        return "--"
    if abs(number - round(number)) < 1e-8:
        text = f"{number:.0f}"
    else:
        text = f"{number:.1f}"
    return f"{text} {unit}".strip()


def _reference_value_for_feature(feature: str, row: pd.Series, default: float) -> float:
    if feature == "candidate_total_gn_dose":
        return _to_float(row.get("current_gn_dose"), default)
    if feature == "candidate_gn_per_total_follicle":
        return _safe_divide(row.get("current_gn_dose"), row.get("total_follicle_count"))
    if feature == "candidate_e2_per_gn":
        return _safe_divide(row.get("current_e2"), row.get("current_gn_dose"))
    return default


def explain_candidate_response(
    snapshot: pd.Series | Mapping[str, Any],
    *,
    fsh_dose: float,
    lh_dose: float,
    hmg_dose: float,
    task: str,
    limit: int = 6,
    bundles: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return local perturbation explanations for candidate-response predictions.

    Each item is the prediction difference after replacing one model feature with a
    fixed clinical reference value. This is a local model attribution for UI review,
    not a causal effect estimate and not KernelSHAP.
    """
    active_bundle = dict(bundles or load_candidate_response_bundle(run_id))
    task_bundles = _task_bundles(active_bundle)
    if task not in task_bundles:
        raise FileNotFoundError(f"Missing candidate response task bundle: {task}")
    task_bundle = task_bundles[task]
    feature_names = set(_feature_names_from_bundle(task_bundle))
    row = build_candidate_response_features(snapshot, fsh_dose=fsh_dose, lh_dose=lh_dose, hmg_dose=hmg_dose)
    base_prediction = _predict_task(row, task_bundle)
    items: list[dict[str, Any]] = []
    for feature, label, unit, default_ref in OUTCOME_EXPLANATION_SPECS:
        if feature not in feature_names:
            continue
        current_value = _to_float(row.get(feature), np.nan)
        if not np.isfinite(current_value):
            continue
        reference_value = _reference_value_for_feature(feature, row, default_ref)
        if not np.isfinite(reference_value) or abs(reference_value - current_value) < 1e-9:
            continue
        perturbed = row.copy()
        perturbed[feature] = reference_value
        perturbed_prediction = _predict_task(perturbed, task_bundle)
        contribution = float(base_prediction - perturbed_prediction)
        if not np.isfinite(contribution) or abs(contribution) < 1e-9:
            continue
        items.append(
            {
                "feature": feature,
                "label": label,
                "value_label": _format_explanation_value(current_value, unit),
                "reference_label": _format_explanation_value(reference_value, unit),
                "mean_abs_shap": abs(contribution),
                "mean_shap": contribution,
                "direction": "\u589e\u52a0\u9884\u6d4b" if contribution > 0 else "\u964d\u4f4e\u9884\u6d4b",
                "source": f"candidate_response_local_perturbation_{task}",
            }
        )
    items.sort(key=lambda item: abs(float(item.get("mean_shap", 0.0))), reverse=True)
    return items[:limit]


def _positive_class_tree_shap(
    estimator: Any,
    matrix: np.ndarray,
    *,
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(estimator, data=background) if background is not None else shap.TreeExplainer(estimator)
    values = explainer.shap_values(matrix)
    expected = explainer.expected_value
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    expected_values = np.asarray(expected, dtype=float).reshape(-1)
    expected_value = float(expected_values[-1])
    return values, expected_value


def _transformed_source_names(bundle: Mapping[str, Any]) -> list[str]:
    feature_names = _feature_names_from_bundle(bundle)
    transformed = [str(name) for name in bundle["preprocessor"].get_feature_names_out()]
    ordered = sorted(feature_names, key=len, reverse=True)
    sources: list[str] = []
    for transformed_name in transformed:
        stripped = transformed_name.split("__", 1)[-1]
        source = next(
            (
                feature
                for feature in ordered
                if stripped == feature or stripped.startswith(f"{feature}_")
            ),
            stripped,
        )
        sources.append(source)
    return sources


def explain_candidate_response_shap(
    snapshot: pd.Series | Mapping[str, Any],
    *,
    fsh_dose: float,
    lh_dose: float,
    hmg_dose: float,
    task: str,
    limit: int = 128,
    bundles: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    attribution_features: set[str] | None = None,
) -> dict[str, Any]:
    """Return source-aggregated Tree SHAP values for the exact deployed task bundle.

    Classification SHAP values are computed on the estimator's positive-class
    probability and uniformly aligned through the deployed one-dimensional
    calibration map. This preserves direction and exactly decomposes the calibrated
    baseline-to-current probability difference before UI-only display filtering.
    """
    active_bundle = dict(bundles or load_candidate_response_bundle(run_id))
    task_bundles = _task_bundles(active_bundle)
    if task not in task_bundles:
        raise FileNotFoundError(f"Missing candidate response task bundle: {task}")
    task_bundle = task_bundles[task]
    row = build_candidate_response_features(
        snapshot,
        fsh_dose=fsh_dose,
        lh_dose=lh_dose,
        hmg_dose=hmg_dose,
    )
    frame = _prepare_frame(row, task_bundle)
    matrix = np.asarray(task_bundle["preprocessor"].transform(frame), dtype=float)
    background_matrix = None
    if attribution_features is not None:
        conditional_background = frame.copy()
        for feature in conditional_background.columns:
            if str(feature) in attribution_features:
                conditional_background.loc[:, feature] = np.nan
        background_matrix = np.asarray(
            task_bundle["preprocessor"].transform(conditional_background),
            dtype=float,
        )
    estimator = task_bundle["estimator"]
    values, raw_baseline = _positive_class_tree_shap(
        estimator,
        matrix,
        background=background_matrix,
    )
    raw_values = values[0]
    task_type = str(task_bundle.get("task_type", "regression"))
    if task_type == "classification":
        probabilities = np.asarray(estimator.predict_proba(matrix), dtype=float)
        raw_prediction = float(
            probabilities[0, 1]
            if probabilities.ndim == 2 and probabilities.shape[1] > 1
            else probabilities.reshape(-1)[0]
        )
        baseline_prediction = _calibrated_probability(raw_baseline, task_bundle)
        prediction = _calibrated_probability(raw_prediction, task_bundle)
        raw_sum = float(raw_values.sum())
        scale = (prediction - baseline_prediction) / raw_sum if abs(raw_sum) > 1e-12 else 0.0
        aligned_values = raw_values * scale
        scale_name = "calibrated_probability"
    else:
        raw_prediction = float(np.asarray(estimator.predict(matrix)).reshape(-1)[0])
        baseline_prediction = float(raw_baseline)
        prediction = raw_prediction
        aligned_values = raw_values
        scale_name = "prediction_value"

    source_names = _transformed_source_names(task_bundle)
    grouped: dict[str, float] = {}
    for source, value in zip(source_names, aligned_values):
        grouped[source] = grouped.get(source, 0.0) + float(value)
    source = f"candidate_response_tree_shap_{task}_{scale_name}"
    items = [
        {
            "feature": feature,
            "feature_value": row.get(feature, np.nan),
            "value_label": _format_explanation_value(row.get(feature, np.nan), ""),
            "mean_abs_shap": abs(value),
            "mean_shap": value,
            "direction": "increases_prediction" if value >= 0 else "decreases_prediction",
            "source": source,
        }
        for feature, value in grouped.items()
    ]
    items.sort(key=lambda item: abs(float(item["mean_shap"])), reverse=True)
    all_sum = float(sum(float(item["mean_shap"]) for item in items))
    return {
        "task": task,
        "run_id": str(active_bundle.get("run_id", run_id or "")),
        "model_name": str(task_bundle.get("selected_model_name", task_bundle.get("model_family", ""))),
        "method": source,
        "baseline_prediction": float(baseline_prediction),
        "prediction": float(prediction),
        "raw_baseline_prediction": float(raw_baseline),
        "raw_prediction": float(raw_prediction),
        "all_attribution_sum": all_sum,
        "reconstruction_error": float(prediction - baseline_prediction - all_sum),
        "all_feature_count": len(items),
        "attribution_scope": (
            "clinician_entered_features_only"
            if attribution_features is not None
            else "all_model_features"
        ),
        "attribution_baseline_definition": (
            "Non-entered model context and candidate dose are fixed at the current scenario; "
            "clinician-entered fields use fitted train-reference imputation values."
            if attribution_features is not None
            else "Model Tree SHAP expected value over all model features."
        ),
        "items": items[: max(1, int(limit))],
    }


def score_candidate_response(
    snapshot: pd.Series | Mapping[str, Any],
    *,
    fsh_dose: float,
    lh_dose: float,
    hmg_dose: float,
    bundles: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    bound_counts: bool = True,
    history_snapshots: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    # The efficacy and safety contracts are intentionally independent:
    # Candidate-response V2 predicts oocytes/MII; the promoted strict moderate-to-severe OHSS
    # classifier remains the safety-warning model.
    if bundles is not None:
        efficacy_bundle = safety_bundle = dict(bundles)
        source = "candidate_dose_response_v1"
    else:
        efficacy_bundle = load_candidate_response_bundle(current_efficacy_run_id())
        safety_bundle = load_candidate_response_bundle(run_id)
        source = "candidate_response_v2_xgb_plus_modsev_ohss"
    efficacy_tasks = _task_bundles(efficacy_bundle)
    safety_tasks = _task_bundles(safety_bundle)
    missing = [task for task in ("layer2_oocytes", "layer2_mii") if task not in efficacy_tasks]
    if "layer2_ohss" not in safety_tasks:
        missing.append("layer2_ohss")
    if missing:
        raise FileNotFoundError(f"Missing candidate response task bundle(s): {missing}")
    row = build_candidate_response_features(
        snapshot,
        fsh_dose=fsh_dose,
        lh_dose=lh_dose,
        hmg_dose=hmg_dose,
    )
    total_follicles = row.get("total_follicle_count")
    raw_oocytes = _predict_task(row, efficacy_tasks["layer2_oocytes"])
    raw_mii = _predict_task(row, efficacy_tasks["layer2_mii"])
    if bound_counts:
        oocytes = _bound_count_prediction(raw_oocytes, total_follicles)
        mii = _bound_count_prediction(raw_mii, total_follicles)
    else:
        oocytes = _bound_count_prediction(raw_oocytes, None)
        mii = _bound_count_prediction(raw_mii, None)
    mii = min(mii, oocytes) if np.isfinite(oocytes) else mii
    ohss = _predict_task(row, safety_tasks["layer2_ohss"])
    efficacy_run = str(efficacy_bundle.get("run_id", current_efficacy_run_id() if bundles is None else ""))
    safety_run = str(safety_bundle.get("run_id", run_id or ""))
    return {
        "source": source,
        "run_id": safety_run,
        "efficacy_run_id": efficacy_run,
        "ohss_run_id": safety_run,
        "model_name": source if bundles is None else str(efficacy_bundle.get("model_name", source)),
        "efficacy_model_name": str(efficacy_bundle.get("model_name", "candidate_response_v2_xgb")),
        "ohss_model_name": str(safety_bundle.get("model_name", "strict_modsev_ohss")),
        "oocytes": oocytes,
        "mii": mii,
        "oocytes_raw": float(max(0.0, _to_float(raw_oocytes, 0.0))),
        "mii_raw": float(max(0.0, _to_float(raw_mii, 0.0))),
        "count_bound_applied": bool(bound_counts),
        "ohss_risk": ohss,
        "ohss_risk_raw": ohss,
    }
