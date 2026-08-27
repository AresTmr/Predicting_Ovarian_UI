from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import json
import joblib
import numpy as np
import pandas as pd

from models.layer1_strategy.action_model import ACTION_LABELS, ID_TO_ACTION, augment_layer1_action_features, create_gn_action_labels
from models.layer1_strategy.knn_retrieval import fit_knn_retriever, get_similar_cases, summarize_similar_action_statistics

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION_TARGET = "combined_gn_action"
DEFAULT_ACTION_THRESHOLD = 37.5
DEFAULT_K = 50
CURRENT_LAYER1_ACTION_RUN_POINTER = REPO_ROOT / "models" / "artifacts" / "current_layer1_action_run.txt"
CURRENT_LAYER1_SPLIT_ACTION_REGISTRY = REPO_ROOT / "models" / "artifacts" / "current_layer1_split_action_runs.json"
SPLIT_ACTION_TARGETS = ("fsh_action", "lh_action", "hmg_action")
SPLIT_ACTION_CN = {"fsh_action": "FSH", "lh_action": "LH", "hmg_action": "HMG"}
ACTION_CN = {"increase": "加量", "maintain": "维持", "decrease": "减量"}
ACTION_CLASS = {"increase": "up", "maintain": "keep", "decrease": "down"}


def _repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def find_latest_layer1_action_dir() -> Path:
    if CURRENT_LAYER1_ACTION_RUN_POINTER.exists():
        run_id = CURRENT_LAYER1_ACTION_RUN_POINTER.read_text(encoding="utf-8").strip()
        if run_id:
            pointer_dir = REPO_ROOT / "models" / "artifacts" / run_id / "layer1_action"
            if (pointer_dir / "layer1_gn_action_best_bundle.joblib").exists():
                return pointer_dir
    candidates = [
        path
        for path in (REPO_ROOT / "models" / "artifacts").glob("phase8_layer1_action_*/layer1_action")
        if (path / "layer1_gn_action_best_bundle.joblib").exists()
    ]
    candidates = sorted(candidates, key=lambda path: (path.stat().st_mtime, path.as_posix()), reverse=True)
    if not candidates:
        raise FileNotFoundError("No Layer1 action artifact directory with bundle found under models/artifacts")
    return candidates[0]


@lru_cache(maxsize=4)
def load_layer1_action_bundle(action_dir: str | None = None) -> dict[str, Any]:
    directory = Path(action_dir) if action_dir else find_latest_layer1_action_dir()
    bundle_path = directory / "layer1_gn_action_best_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Layer1 action bundle not found: {bundle_path}")
    return joblib.load(bundle_path)


@lru_cache(maxsize=2)
def load_layer1_split_action_registry(registry_path: str | None = None) -> dict[str, Any]:
    path = _repo_path(registry_path) if registry_path else CURRENT_LAYER1_SPLIT_ACTION_REGISTRY
    if not path.exists():
        raise FileNotFoundError(f"Layer1 split action registry not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "targets" not in payload or not isinstance(payload["targets"], Mapping):
        raise ValueError(f"Invalid Layer1 split action registry: {path}")
    return payload


def find_layer1_split_action_dir(target: str) -> Path:
    registry = load_layer1_split_action_registry()
    target_payload = registry.get("targets", {}).get(target)
    if not isinstance(target_payload, Mapping):
        raise KeyError(f"Target {target} not found in Layer1 split action registry")
    output_dir = target_payload.get("output_dir")
    if not output_dir:
        raise ValueError(f"Target {target} has no output_dir in split action registry")
    directory = _repo_path(str(output_dir))
    if not (directory / "layer1_gn_action_best_bundle.joblib").exists():
        raise FileNotFoundError(f"Layer1 split action bundle not found for {target}: {directory}")
    return directory


@lru_cache(maxsize=8)
def load_layer1_split_action_bundle(target: str) -> dict[str, Any]:
    directory = find_layer1_split_action_dir(target)
    return joblib.load(directory / "layer1_gn_action_best_bundle.joblib")


@lru_cache(maxsize=4)
def load_layer1_history_frame(
    input_path: str = "data/processed/layer1_strategy_dataset.csv",
    split_manifest_path: str = "data/splits/split_manifest_v1.csv",
    target: str = DEFAULT_ACTION_TARGET,
    threshold: float = DEFAULT_ACTION_THRESHOLD,
) -> pd.DataFrame:
    frame = create_gn_action_labels(augment_layer1_action_features(pd.read_csv(_repo_path(input_path))), threshold=threshold)
    if "has_next_visit" in frame.columns:
        frame = frame[frame["has_next_visit"].astype(bool)].copy()
    if "strategy_eligible_flag" in frame.columns:
        frame = frame[frame["strategy_eligible_flag"].astype(bool)].copy()
    frame = frame[frame[target].isin(ACTION_LABELS)].copy()
    manifest_path = _repo_path(split_manifest_path)
    if manifest_path.exists() and "cycle_uid" in frame.columns:
        manifest = pd.read_csv(manifest_path)
        if {"cycle_uid", "split"}.issubset(manifest.columns):
            split_map = manifest.drop_duplicates("cycle_uid").set_index("cycle_uid")["split"]
            frame["split"] = frame["cycle_uid"].map(split_map).fillna("train")
            frame = frame[frame["split"] == "train"].copy()
    if frame.empty:
        raise ValueError("Layer1 KNN history frame is empty")
    return frame.reset_index(drop=True)


@lru_cache(maxsize=4)
def load_layer1_knn_retriever(
    input_path: str = "data/processed/layer1_strategy_dataset.csv",
    split_manifest_path: str = "data/splits/split_manifest_v1.csv",
    target: str = DEFAULT_ACTION_TARGET,
    threshold: float = DEFAULT_ACTION_THRESHOLD,
    k: int = DEFAULT_K,
):
    history = load_layer1_history_frame(input_path, split_manifest_path, target, threshold)
    return fit_knn_retriever(history, k=k)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    return int(round(_to_float(value, float(default))))


def patient_form_to_snapshot(form: Mapping[str, Any]) -> pd.Series:
    # The UI predicts the latest/today Gn doses. These dose-valued model features
    # must therefore come from previous-known/reference dosing, not today's target.
    fsh_dose = _to_float(form.get("reference_fsh", form.get("previous_fsh_daily_dose", form.get("current_fsh"))))
    lh_dose = _to_float(form.get("reference_lh", form.get("previous_lh_daily_dose", form.get("current_lh"))))
    hmg_dose = _to_float(form.get("reference_hmg", form.get("previous_hmg_daily_dose", form.get("current_hmg"))))
    lh_like_dose = lh_dose + hmg_dose
    current_total = fsh_dose + lh_like_dose
    new_bin_keys = ("f_10_12", "f_13_15", "f_16_18", "f_gt18")
    legacy_bin_keys = ("f_10_13", "f_14_17", "f_ge18")
    has_new_bins = any(key in form for key in new_bin_keys)
    has_legacy_bins = (not has_new_bins) and any(key in form for key in legacy_bin_keys)
    f_lt10 = _to_int(form.get("f_lt10"))
    if has_new_bins:
        f_10_12 = _to_int(form.get("f_10_12"))
        f_13_15 = _to_int(form.get("f_13_15"))
        f_16_18 = _to_int(form.get("f_16_18"))
        f_gt18 = _to_int(form.get("f_gt18"))
    elif has_legacy_bins:
        f_10_12 = _to_int(form.get("f_10_13"))
        legacy_14_17 = _to_int(form.get("f_14_17"))
        f_13_15 = 0
        f_16_18 = legacy_14_17
        f_gt18 = _to_int(form.get("f_ge18"))
    else:
        f_10_12 = f_13_15 = f_16_18 = f_gt18 = 0
    binned_total_follicles = f_lt10 + f_10_12 + f_13_15 + f_16_18 + f_gt18
    explicit_total = form.get("total_follicles", form.get("total_follicle_count"))
    if explicit_total not in (None, ""):
        total_follicles = _to_int(explicit_total, binned_total_follicles)
    else:
        total_follicles = binned_total_follicles
    mature_follicles = f_16_18 + f_gt18
    left_follicles = _to_int(form.get("left_follicles", form.get("left_follicle_count")), 0)
    right_follicles = _to_int(form.get("right_follicles", form.get("right_follicle_count")), 0)
    serum_fsh = _to_float(form.get("serum_fsh", form.get("current_fsh_value", form.get("current_serum_fsh"))), np.nan)
    patient_id = str(form.get("patient_id", "ui_patient"))
    visit = _to_int(form.get("visit"), 1)
    stim_day = _to_int(form.get("stim_day"), 1)
    snapshot = {
        "art_id": patient_id,
        "cycle_uid": str(form.get("cycle_uid", f"{patient_id}__ui")),
        "visit_uid": str(form.get("visit_uid", f"{patient_id}__ui_m{visit}")),
        "canonical_visit_key": str(form.get("visit_uid", f"{patient_id}__ui_m{visit}")),
        "monitoring_order": visit,
        "Day": stim_day,
        "age": _to_float(form.get("age")),
        "bmi": _to_float(form.get("bmi")),
        "infertility_duration": _to_float(form.get("years", form.get("infertility_duration"))),
        "diagnosis_primary_secondary": form.get("diagnosis", form.get("infertility", "未知")),
        "afc": _to_float(form.get("afc")),
        "amh": _to_float(form.get("amh")),
        "initial_gn_dose": _to_float(form.get("initial_gn", form.get("initial_gn_dose")), current_total),
        "basal_fsh": _to_float(form.get("basal_fsh")),
        "basal_lh": _to_float(form.get("basal_lh")),
        "basal_e2": _to_float(form.get("basal_e2")),
        "basal_p": _to_float(form.get("basal_p")),
        "male_factor_infertility_flag": 0 if str(form.get("male_factor_infertility", "否")) in {"否", "0", "False"} else 1,
        "male_age": _to_float(form.get("male_age"), np.nan),
        "sperm_source_group": form.get("sperm_source_group", "Unknown"),
        "fertilization_method": form.get("fertilization_method", "Unknown"),
        "current_e2": _to_float(form.get("e2", form.get("current_e2"))),
        "current_lh": _to_float(form.get("lh_value", form.get("current_lh_value"))),
        "current_p": _to_float(form.get("p", form.get("current_p"))),
        "current_fsh": serum_fsh,
        "current_endometrium": _to_float(form.get("current_endometrium"), np.nan),
        "current_fsh_daily_dose": fsh_dose,
        "current_lh_daily_dose": lh_dose,
        "current_hmg_daily_dose": hmg_dose,
        "current_lh_like_hmg_daily_dose": lh_like_dose,
        "previous_fsh_daily_dose": fsh_dose,
        "previous_lh_daily_dose": lh_dose,
        "previous_hmg_daily_dose": hmg_dose,
        "previous_lh_like_hmg_daily_dose": lh_like_dose,
        "current_gn_dose": current_total,
        "cumulative_fsh_dose": fsh_dose * stim_day,
        "cumulative_lh_dose": lh_dose * stim_day,
        "cumulative_hmg_dose": hmg_dose * stim_day,
        "cumulative_lh_like_hmg_dose": lh_like_dose * stim_day,
        "cumulative_gn_dose": current_total * stim_day,
        "gn_day": stim_day,
        "cycle_day": _to_float(form.get("cycle_day"), np.nan),
        "previous_gn_dose": _to_float(form.get("previous_gn_dose"), current_total),
        "delta_e2": _to_float(form.get("delta_e2"), np.nan),
        "delta_lh": _to_float(form.get("delta_lh"), np.nan),
        "delta_p": _to_float(form.get("delta_p"), np.nan),
        "days_since_previous_visit": _to_float(form.get("days_since_previous_visit"), np.nan),
        "visits_seen": visit,
        "follicle_count_lt_10": f_lt10,
        "follicle_count_10_12": f_10_12,
        "follicle_count_13_15": f_13_15,
        "follicle_count_16_18": f_16_18,
        "follicle_count_gt_18": f_gt18,
        "total_follicle_count": total_follicles,
        "left_follicle_count": left_follicles,
        "right_follicle_count": right_follicles,
        "mature_follicle_count": mature_follicles,
        "growing_follicle_count": f_10_12 + f_13_15 + f_16_18 + f_gt18,
        "medium_plus_follicle_count": f_13_15 + f_16_18 + f_gt18,
        "max_follicle_diameter": _to_float(form.get("max_f")),
        "mean_follicle_diameter": _to_float(form.get("mean_f", form.get("mean_follicle_diameter")), np.nan),
        "follicle_maturity_index": mature_follicles / total_follicles if total_follicles else 0.0,
        "mature_follicle_share": mature_follicles / total_follicles if total_follicles else 0.0,
        "large_follicle_share": f_gt18 / total_follicles if total_follicles else 0.0,
    }
    return pd.Series(snapshot)


def _prepare_feature_frame(snapshot: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    frame = snapshot.to_frame().T
    for column in feature_names:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[feature_names]


def _predict_from_bundle(snapshot: pd.Series, bundle: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    estimator = bundle["estimator"]
    preprocessor = bundle["preprocessor"]
    feature_names = list(bundle.get("feature_names", []))
    labels = list(bundle.get("action_labels", ACTION_LABELS))
    features = _prepare_feature_frame(snapshot, feature_names)
    matrix = preprocessor.transform(features)
    pred_raw = np.asarray(estimator.predict(matrix)).reshape(-1)[0]
    if isinstance(pred_raw, str):
        recommended = pred_raw
    else:
        recommended = ID_TO_ACTION.get(int(pred_raw), "maintain")
    if hasattr(estimator, "predict_proba"):
        raw_proba = np.asarray(estimator.predict_proba(matrix), dtype=float).reshape(1, -1)[0]
    else:
        raw_proba = np.zeros(len(labels), dtype=float)
        raw_proba[labels.index(recommended) if recommended in labels else 1] = 1.0
    probabilities = {label: float(raw_proba[idx]) if idx < len(raw_proba) else 0.0 for idx, label in enumerate(labels)}
    decision_weights = bundle.get("decision_weights") or bundle.get("probability_class_weights")
    if isinstance(decision_weights, Mapping):
        adjusted = {label: probabilities.get(label, 0.0) * float(decision_weights.get(label, 1.0)) for label in labels}
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {label: value / total for label, value in adjusted.items()}
        recommended = max(adjusted, key=adjusted.get) if adjusted else recommended
        probabilities = adjusted
    return recommended, {label: probabilities.get(label, 0.0) for label in ACTION_LABELS}


def predict_layer1_action_context(
    snapshot: pd.Series | Mapping[str, Any],
    *,
    bundle: Mapping[str, Any] | None = None,
    history_frame: pd.DataFrame | None = None,
    k: int = DEFAULT_K,
    target: str = DEFAULT_ACTION_TARGET,
    action_dir: str | None = None,
) -> dict[str, Any]:
    snapshot_series = pd.Series(snapshot)
    snapshot_series = augment_layer1_action_features(snapshot_series.to_frame().T).iloc[0]
    bundle = dict(bundle or load_layer1_action_bundle(action_dir))
    recommended, probabilities = _predict_from_bundle(snapshot_series, bundle)
    if history_frame is None:
        history_frame = load_layer1_history_frame(target=target)
        retriever = load_layer1_knn_retriever(target=target, k=k)
    else:
        retriever = fit_knn_retriever(history_frame, k=k)
    similar = get_similar_cases(
        retriever,
        snapshot_series,
        k=k,
        exclude_patient_id=snapshot_series.get("art_id"),
        exclude_cycle_id=snapshot_series.get("cycle_uid"),
    )
    stats = summarize_similar_action_statistics(
        snapshot_series,
        similar,
        model_recommended_action=recommended,
        model_probabilities=probabilities,
        k=k,
        action_col=target,
    )
    return {
        "recommended_action": recommended,
        "recommended_action_cn": ACTION_CN.get(recommended, recommended),
        "recommended_action_class": ACTION_CLASS.get(recommended, "keep"),
        "confidence": float(probabilities.get(recommended, 0.0)),
        "probabilities": probabilities,
        "similar_action_stats": stats,
        "similar_cases": similar,
        "knn_k": int(k),
    }


def predict_layer1_split_action_contexts(
    snapshot: pd.Series | Mapping[str, Any],
    *,
    targets: tuple[str, ...] = SPLIT_ACTION_TARGETS,
    k: int = DEFAULT_K,
) -> dict[str, dict[str, Any]]:
    """Predict FSH/LH/HMG adjustment actions with their own bundles and KNN evidence."""
    snapshot_series = pd.Series(snapshot)
    contexts: dict[str, dict[str, Any]] = {}
    for target in targets:
        try:
            bundle = load_layer1_split_action_bundle(target)
            threshold = float(bundle.get("threshold", DEFAULT_ACTION_THRESHOLD))
            history = load_layer1_history_frame(target=target, threshold=threshold)
            context = predict_layer1_action_context(
                snapshot_series,
                bundle=bundle,
                history_frame=history,
                k=k,
                target=target,
            )
            context["target"] = target
            context["drug_cn"] = SPLIT_ACTION_CN.get(target, target)
            context["threshold"] = threshold
            contexts[target] = context
        except Exception as exc:
            contexts[target] = {
                "target": target,
                "drug_cn": SPLIT_ACTION_CN.get(target, target),
                "error": str(exc),
                "recommended_action": "maintain",
                "recommended_action_cn": ACTION_CN["maintain"],
                "recommended_action_class": ACTION_CLASS["maintain"],
                "confidence": 0.0,
                "probabilities": {label: 0.0 for label in ACTION_LABELS},
                "similar_action_stats": pd.DataFrame(),
                "similar_cases": pd.DataFrame(),
                "knn_k": int(k),
            }
    return contexts


def split_action_summary_text(contexts: Mapping[str, Mapping[str, Any]] | None) -> str:
    if not contexts:
        return "FSH/LH/HMG 拆分 action bundle 未读取，当前使用前端候选剂量规则。"
    parts = []
    for target in SPLIT_ACTION_TARGETS:
        ctx = contexts.get(target, {}) if isinstance(contexts, Mapping) else {}
        drug = SPLIT_ACTION_CN.get(target, target)
        action = str(ctx.get("recommended_action", "maintain"))
        probs = ctx.get("probabilities", {}) if isinstance(ctx.get("probabilities"), Mapping) else {}
        parts.append(
            f"{drug}: {ACTION_CN.get(action, action)} "
            f"(加 {float(probs.get('increase', 0.0)):.2f}/维 {float(probs.get('maintain', 0.0)):.2f}/减 {float(probs.get('decrease', 0.0)):.2f})"
        )
    return "；".join(parts)


def dose_delta_from_action(action: str, *, drug: str) -> float:
    if action == "increase":
        return 75.0 if drug in {"fsh_action", "hmg_action"} else 37.5
    if action == "decrease":
        return -75.0 if drug in {"fsh_action", "hmg_action"} else -37.5
    return 0.0


def evidence_for_action(context: Mapping[str, Any] | None, action: str) -> dict[str, Any]:
    fallback = {
        "selection": 0.0,
        "ovarian": np.nan,
        "mii": np.nan,
        "ohss_free": np.nan,
        "text": "当前未读取到实时相似病例统计，需检查 Layer1 action artifacts 和 KNN 历史库。",
    }
    if not context:
        return fallback
    stats = context.get("similar_action_stats")
    if not isinstance(stats, pd.DataFrame) or stats.empty or "action" not in stats.columns:
        return fallback
    rows = stats[stats["action"] == action]
    if rows.empty:
        return fallback
    row = rows.iloc[0]
    selection = float(row.get("selection_rate", 0.0) or 0.0)
    mii = row.get("mii_success_rate", np.nan)
    ohss_free = row.get("ohss_free_rate", np.nan)
    if action == context.get("recommended_action"):
        text = f"在 KNN 相似历史病例中，该动作选择率为 {selection:.0%}，可作为当前模型推荐的历史参考依据。"
    else:
        text = f"该候选动作在 KNN 相似历史病例中的选择率为 {selection:.0%}，用于对比参考。"
    return {
        "selection": selection,
        "ovarian": float(row.get("ovarian_response_success_rate", np.nan)),
        "mii": float(mii) if not pd.isna(mii) else np.nan,
        "ohss_free": float(ohss_free) if not pd.isna(ohss_free) else np.nan,
        "text": text,
    }
