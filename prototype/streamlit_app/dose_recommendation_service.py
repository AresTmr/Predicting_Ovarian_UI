from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch

try:
    from .ui_real_data_sources import UI_INPUT_ATTRIBUTION_FEATURES
    from .dose_probability_calibration import (
        apply_temperature_to_logits,
        target_calibration,
    )
except Exception:
    from ui_real_data_sources import UI_INPUT_ATTRIBUTION_FEATURES
    from dose_probability_calibration import (
        apply_temperature_to_logits,
        target_calibration,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "results" / "result1_deep_temporal_gn_dose_phase870_gru_addgate_ui_reduced"
CONTRACT_PATH = MODEL_ROOT / "logs" / "ui_reduced_feature_contract.json"
HOLDOUT_MODEL_ROOT = REPO_ROOT / "outputs" / "paper_audit_20260715" / "checkpoints"
FSH3_HOLDOUT_CHECKPOINT = REPO_ROOT / "outputs" / "paper_audit_fsh3_20260723" / "checkpoints" / "parsimonious_clinical_gru_addgate_fsh3_holdout.pt"
SPLIT_MANIFEST_PATH = REPO_ROOT / "data" / "splits" / "split_manifest_v1.csv"
DEFAULT_DEPLOYMENT_MODE = "holdout"
PHASE866_PATH = REPO_ROOT / "scripts" / "experiment" / "phase866_daily_expanded_model_comparison" / "run_phase866_daily_expanded_model_comparison.py"

TARGETS: dict[str, dict[str, Any]] = {
    "fsh": {
        "name": "FSH",
        "label_col": "daily_fsh_dose_bin_3class",
        "labels": ["low_or_none", "moderate", "high"],
        "display": {"low_or_none": "0-80", "moderate": "80-160", "high": ">160"},
        "dose": {"low_or_none": 40.0, "moderate": 120.0, "high": 200.0},
    },
    "lh": {
        "name": "LH",
        "label_col": "daily_lh_dose_bin_grouped",
        "labels": ["dose_0", "dose_75", "dose_ge150"],
        "display": {"dose_0": "0", "dose_75": "75", "dose_ge150": ">75"},
        "dose": {"dose_0": 0.0, "dose_75": 75.0, "dose_ge150": 150.0},
    },
    "hmg": {
        "name": "HMG",
        "label_col": "daily_hmg_dose_bin_grouped",
        "labels": ["dose_0", "dose_75", "dose_150", "dose_ge225"],
        "display": {"dose_0": "0", "dose_75": "75", "dose_150": "150", "dose_ge225": ">150"},
        "dose": {"dose_0": 0.0, "dose_75": 75.0, "dose_150": 150.0, "dose_ge225": 225.0},
    },
}

STATIC_ALIASES = {
    "age": ("age",),
    "bmi": ("bmi",),
    "amh": ("amh",),
    "afc": ("afc",),
    "basal_fsh": ("basal_fsh",),
    "basal_lh": ("basal_lh",),
    "basal_e2": ("basal_e2",),
    "basal_p": ("basal_p",),
    "infertility_duration": ("infertility_duration", "years"),
}

DYNAMIC_ALIASES = {
    "current_e2": ("current_e2", "e2"),
    "current_lh": ("current_lh", "lh_value", "current_lh_value"),
    "current_p": ("current_p", "p"),
    "current_fsh": ("current_fsh_value", "serum_fsh", "current_serum_fsh"),
    "current_endometrium": ("current_endometrium", "endometrium_thickness"),
    "total_follicle_count": ("total_follicle_count", "total_follicles"),
    "left_follicle_count": ("left_follicle_count", "left_follicles"),
    "right_follicle_count": ("right_follicle_count", "right_follicles"),
    "max_follicle_diameter": ("max_follicle_diameter", "max_f"),
    "mean_follicle_diameter": ("mean_follicle_diameter", "mean_f"),
    "follicle_count_lt_10": ("follicle_count_lt_10", "f_lt10"),
    "follicle_count_10_12": ("follicle_count_10_12", "f_10_12"),
    "follicle_count_13_15": ("follicle_count_13_15", "f_13_15"),
    "follicle_count_16_18": ("follicle_count_16_18", "f_16_18"),
    "follicle_count_gt_18": ("follicle_count_gt_18", "f_gt18"),
    "previous_fsh_daily_dose": ("previous_fsh_daily_dose", "current_fsh"),
    "previous_lh_daily_dose": ("previous_lh_daily_dose", "current_lh"),
    "previous_hmg_daily_dose": ("previous_hmg_daily_dose", "current_hmg"),
}

PREVIOUS_DOSE_FROM_PRIOR_RECORD = {
    "previous_fsh_daily_dose": ("current_fsh", "previous_fsh_daily_dose"),
    "previous_lh_daily_dose": ("current_lh", "previous_lh_daily_dose"),
    "previous_hmg_daily_dose": ("current_hmg", "previous_hmg_daily_dose"),
}


def _to_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _first_numeric(source: Mapping[str, Any], aliases: Sequence[str]) -> tuple[float, bool]:
    for key in aliases:
        if key in source and source.get(key) not in (None, ""):
            value = _to_float(source.get(key), np.nan)
            if np.isfinite(value):
                return value, True
    return float("nan"), False


def _fsh3_label(dose: Any) -> str | None:
    value = _to_float(dose, np.nan)
    if not np.isfinite(value):
        return None
    if value < 80:
        return "low_or_none"
    if value <= 160:
        return "moderate"
    return "high"


def _lh3_label(dose: Any) -> str | None:
    value = _to_float(dose, np.nan)
    if not np.isfinite(value):
        return None
    if value == 0:
        return "dose_0"
    if value == 75:
        return "dose_75"
    if value >= 150:
        return "dose_ge150"
    return "dose_75" if value < 150 else "dose_ge150"


def _hmg4_label(dose: Any) -> str | None:
    value = _to_float(dose, np.nan)
    if not np.isfinite(value):
        return None
    if value == 0:
        return "dose_0"
    if value == 75:
        return "dose_75"
    if value == 150:
        return "dose_150"
    if value >= 225:
        return "dose_ge225"
    if value < 75:
        return "dose_75"
    if value < 150:
        return "dose_150"
    return "dose_ge225"


@lru_cache(maxsize=1)
def load_ui_reduced_contract() -> dict[str, Any]:
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"UI-reduced feature contract not found: {CONTRACT_PATH}")
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _phase866_module() -> Any:
    spec = importlib.util.spec_from_file_location("phase866_ui_reduced_inference", PHASE866_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Phase 8.66 runner from {PHASE866_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _training_panel() -> pd.DataFrame:
    phase866 = _phase866_module()
    monitoring = pd.read_csv(REPO_ROOT / "data" / "processed" / "layer1_strategy_dataset.csv", low_memory=False)
    baseline = pd.read_csv(REPO_ROOT / "data" / "processed" / "baseline_cycle_dataset.csv", low_memory=False)
    panel = phase866.phase864.build_daily_no_sw_panel(monitoring, baseline, min_day=1, max_day=20).copy()
    panel["daily_fsh_dose_bin_3class"] = panel["daily_fsh_daily_dose"].map(_fsh3_label)
    panel["daily_lh_dose_bin_grouped"] = panel["daily_lh_daily_dose"].map(_lh3_label)
    panel["daily_hmg_dose_bin_grouped"] = panel["daily_hmg_daily_dose"].map(_hmg4_label)
    return panel


def _checkpoint_path(target: str, fold: int) -> Path:
    return MODEL_ROOT / f"train_{target}" / "models" / f"gru_addgate_fold{fold}.pt"


@lru_cache(maxsize=16)
def _checkpoint(target: str, fold: int) -> dict[str, Any]:
    path = _checkpoint_path(target, fold)
    if not path.exists():
        raise FileNotFoundError(f"UI-reduced GRU(AddGate) checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _holdout_checkpoint_path(target: str) -> Path:
    if target == "fsh":
        return FSH3_HOLDOUT_CHECKPOINT
    return HOLDOUT_MODEL_ROOT / f"gru_addgate_{target}_holdout.pt"


@lru_cache(maxsize=3)
def _holdout_checkpoint(target: str) -> dict[str, Any]:
    path = _holdout_checkpoint_path(target)
    if not path.exists():
        raise FileNotFoundError(f"UI-reduced GRU(AddGate) holdout checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


@lru_cache(maxsize=1)
def _split_manifest() -> pd.DataFrame:
    if not SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Frozen split manifest not found: {SPLIT_MANIFEST_PATH}")
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH, usecols=["cycle_uid", "split"])
    manifest["cycle_uid"] = manifest["cycle_uid"].astype(str)
    manifest["split"] = manifest["split"].astype(str).str.lower()
    if manifest["cycle_uid"].duplicated().any():
        raise ValueError("Frozen split manifest contains duplicate cycle_uid values")
    return manifest


def _target_training_raw(target: str) -> Any:
    phase866 = _phase866_module()
    spec = TARGETS[target]
    labels = spec["labels"]
    panel = _training_panel()
    checkpoint = _holdout_checkpoint(target)
    static_features = list(checkpoint["static_features"])
    dynamic_features = list(checkpoint["dynamic_features"])
    frame = panel[panel[spec["label_col"]].isin(labels)].copy()
    frame["ctfe_truth"] = frame[spec["label_col"]].astype(str)
    frame["ctfe_truth_id"] = frame["ctfe_truth"].map({label: i for i, label in enumerate(labels)}).astype(int)
    frame["source_visit_uid"] = frame.get("visit_uid", pd.Series([pd.NA] * len(frame), index=frame.index)).astype(str)
    frame["Day"] = pd.to_numeric(frame["Day"], errors="coerce").astype(int)
    frame["evaluation_day"] = frame["Day"]
    frame["monitoring_order"] = frame["Day"]
    frame["gn_day"] = frame["Day"].astype(float)
    frame["visit_uid"] = frame["cycle_uid"].astype(str) + f"__daily_{target}_day" + frame["Day"].astype(str)
    frame["baseline_only_feature_row"] = frame["baseline_only_feature_row"].astype(bool).astype(int) if "baseline_only_feature_row" in frame.columns else 0
    return phase866.phase864.build_daily_sequence_arrays(
        frame.sort_values(["cycle_uid", "Day", "visit_uid"], kind="mergesort").reset_index(drop=True),
        static_features=static_features,
        dynamic_features=dynamic_features,
        max_history_days=16,
    )


@lru_cache(maxsize=3)
def _cached_target_training_raw(target: str) -> Any:
    return _target_training_raw(target)


def _derive_days_since_previous(records: Sequence[Mapping[str, Any]], idx: int) -> float:
    record = records[idx]
    value = _to_float(record.get("days_since_previous_visit"), np.nan)
    if np.isfinite(value):
        return value
    if idx <= 0:
        return 0.0
    current = _to_float(record.get("stim_day"), np.nan)
    previous = _to_float(records[idx - 1].get("stim_day"), np.nan)
    if np.isfinite(current) and np.isfinite(previous):
        return max(0.0, current - previous)
    return float("nan")


def _query_rows(patient: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, target: str) -> tuple[pd.DataFrame, list[str]]:
    contract = load_ui_reduced_contract()
    static_features = list(contract.get("static_features") or _checkpoint(target, 1)["static_features"])
    dynamic_features = list(contract.get("dynamic_features") or _checkpoint(target, 1)["dynamic_features"])
    warnings: list[str] = []
    patient_id = str(patient.get("patient_id") or "ui_patient")
    cycle_uid = str(patient.get("cycle_uid") or f"{patient_id}__ui_reduced")
    active_records = [dict(r) for r in records if isinstance(r, Mapping)] or [dict(patient)]
    rows: list[dict[str, Any]] = []

    static_values: dict[str, Any] = {}
    for feature in static_features:
        if feature == "male_factor_infertility_flag":
            raw = str(patient.get("male_factor_infertility", patient.get("male_factor_infertility_flag", "?"))).strip().lower()
            static_values[feature] = 0.0 if raw in {"?", "0", "false", "no", "none", ""} else 1.0
            continue
        value, found = _first_numeric(patient, STATIC_ALIASES.get(feature, (feature,)))
        static_values[feature] = value
        if not found:
            warnings.append(f"missing static feature: {feature}")

    for idx, record in enumerate(active_records):
        row: dict[str, Any] = {
            **static_values,
            "art_id": patient_id,
            "cycle_uid": cycle_uid,
            "visit_uid": f"{cycle_uid}__ui_record_{idx + 1}",
            "source_visit_uid": f"{cycle_uid}__ui_record_{idx + 1}",
            "ctfe_truth": TARGETS[target]["labels"][0],
            "ctfe_truth_id": 0,
            TARGETS[target]["label_col"]: TARGETS[target]["labels"][0],
            "baseline_only_feature_row": 0,
        }
        stim_day = _to_float(record.get("stim_day", patient.get("stim_day")), np.nan)
        row["Day"] = stim_day
        row["gn_day"] = stim_day
        row["monitoring_order"] = float(idx + 1)
        row["days_since_previous_visit"] = _derive_days_since_previous(active_records, idx)
        row["cycle_day"] = _to_float(record.get("cycle_day"), stim_day)

        for feature in dynamic_features:
            if feature in row:
                continue
            if feature == "growing_follicle_count":
                row[feature] = sum(
                    _first_numeric(record, DYNAMIC_ALIASES[key])[0]
                    for key in ("follicle_count_10_12", "follicle_count_13_15", "follicle_count_16_18", "follicle_count_gt_18")
                )
                continue
            value, found = _first_numeric(record, DYNAMIC_ALIASES.get(feature, (feature,)))
            if not found and feature in PREVIOUS_DOSE_FROM_PRIOR_RECORD and idx > 0:
                value, found = _first_numeric(
                    active_records[idx - 1],
                    PREVIOUS_DOSE_FROM_PRIOR_RECORD[feature],
                )
            row[feature] = value
            if not found:
                warnings.append(f"missing dynamic feature in monitoring record {idx + 1}: {feature}")

        bins = [
            row.get("follicle_count_lt_10"),
            row.get("follicle_count_10_12"),
            row.get("follicle_count_13_15"),
            row.get("follicle_count_16_18"),
            row.get("follicle_count_gt_18"),
        ]
        if all(np.isfinite(_to_float(v, np.nan)) for v in bins) and np.isfinite(_to_float(row.get("total_follicle_count"), np.nan)):
            bin_sum = sum(float(v) for v in bins)
            if abs(float(row["total_follicle_count"]) - bin_sum) > 1e-6:
                warnings.append(f"monitoring record {idx + 1}: total follicle count differs from follicle-bin sum")
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["Day"] = pd.to_numeric(frame["Day"], errors="coerce").fillna(1).astype(int)
    return frame, sorted(set(warnings))


def _combine_raw(train_raw: Any, query_raw: Any) -> Any:
    return SimpleNamespace(
        static=np.vstack([train_raw.static, query_raw.static]),
        dynamic=np.concatenate([train_raw.dynamic, query_raw.dynamic], axis=0),
        mask=np.vstack([train_raw.mask, query_raw.mask]),
        y=np.concatenate([train_raw.y, query_raw.y]),
        row_index=pd.concat([train_raw.row_index, query_raw.row_index], ignore_index=True),
        static_features=train_raw.static_features,
        dynamic_features=train_raw.dynamic_features,
    )


def _predict_softmax(
    model: torch.nn.Module,
    data: Any,
    row_index: int,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        static_x = torch.tensor(data.static[[row_index]], dtype=torch.float32)
        dynamic_x = torch.tensor(data.dynamic[[row_index]], dtype=torch.float32)
        mask = torch.tensor(data.mask[[row_index]], dtype=torch.float32)
        logits, _embedding = model(static_x, dynamic_x, mask)
        raw = apply_temperature_to_logits(logits, temperature).cpu().numpy().astype(float)
    raw = raw - raw.max(axis=1, keepdims=True)
    proba = np.exp(raw)
    return (proba / proba.sum(axis=1, keepdims=True))[0]


def _selected_class_local_attribution(
    model: torch.nn.Module,
    data: Any,
    row_index: int,
    class_index: int,
    background_indices: Sequence[int] | None = None,
    attribution_features: set[str] | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Return selected-class probability EIG from a conditional input baseline.

    When ``attribution_features`` is provided, every other model feature is fixed
    at the current patient's value in all background rows. The resulting baseline
    and contributions therefore vary only clinician-entered fields and do not need
    a hidden residual or an unnamed internal-feature total for completeness.
    """
    model.eval()
    query_static = torch.tensor(data.static[[row_index]], dtype=torch.float32)
    query_dynamic = torch.tensor(data.dynamic[[row_index]], dtype=torch.float32)
    query_mask = torch.tensor(data.mask[[row_index]], dtype=torch.float32)
    visit_count = float(np.asarray(data.mask[row_index], dtype=float).sum())
    if background_indices is None:
        candidate_indices = np.arange(max(0, int(row_index)), dtype=int)
    else:
        candidate_indices = np.asarray(background_indices, dtype=int)
        candidate_indices = candidate_indices[
            (candidate_indices >= 0) & (candidate_indices < len(data.y)) & (candidate_indices != int(row_index))
        ]
    if candidate_indices.size:
        counts = np.asarray(data.mask[candidate_indices], dtype=float).sum(axis=1)
        exact = candidate_indices[np.isclose(counts, visit_count)]
        pool = exact if exact.size else candidate_indices[np.argsort(np.abs(counts - visit_count))]
    else:
        pool = np.asarray([row_index], dtype=int)
    background_size = min(12, int(pool.size))
    selected_positions = np.linspace(0, max(int(pool.size) - 1, 0), background_size, dtype=int)
    background_indices = pool[selected_positions]
    background_static = torch.tensor(data.static[background_indices], dtype=torch.float32)
    background_dynamic = torch.tensor(data.dynamic[background_indices], dtype=torch.float32)
    background_mask = query_mask.repeat(background_size, 1)
    background_dynamic = background_dynamic * background_mask.unsqueeze(-1)
    if attribution_features is not None:
        allowed = set(attribution_features)
        background_static = background_static.clone()
        background_dynamic = background_dynamic.clone()
        for idx, feature in enumerate(data.static_features):
            if str(feature) not in allowed:
                background_static[:, idx] = query_static[0, idx]
        for idx, feature in enumerate(data.dynamic_features):
            if str(feature) not in allowed:
                background_dynamic[:, :, idx] = query_dynamic[0, :, idx]

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(16)
    alphas = torch.tensor((nodes + 1.0) / 2.0, dtype=torch.float32)
    weights = torch.tensor(quadrature_weights / 2.0, dtype=torch.float32)
    static_delta = query_static - background_static
    dynamic_delta = (query_dynamic - background_dynamic) * background_mask.unsqueeze(-1)
    static_path = (
        background_static[:, None, :]
        + alphas[None, :, None] * static_delta[:, None, :]
    ).reshape(-1, query_static.shape[-1]).requires_grad_(True)
    dynamic_path = (
        background_dynamic[:, None, :, :]
        + alphas[None, :, None, None] * dynamic_delta[:, None, :, :]
    ).reshape(-1, query_dynamic.shape[1], query_dynamic.shape[2]).requires_grad_(True)
    path_mask = background_mask[:, None, :].repeat(1, len(alphas), 1).reshape(-1, query_mask.shape[1])
    logits, _embedding = model(static_path, dynamic_path, path_mask)
    selected_probability = torch.softmax(
        apply_temperature_to_logits(logits, temperature), dim=1
    )[:, int(class_index)]
    static_grad, dynamic_grad = torch.autograd.grad(
        selected_probability.sum(),
        (static_path, dynamic_path),
        retain_graph=False,
        create_graph=False,
    )
    static_grad = static_grad.reshape(background_size, len(alphas), -1)
    dynamic_grad = dynamic_grad.reshape(
        background_size,
        len(alphas),
        query_dynamic.shape[1],
        query_dynamic.shape[2],
    )
    integrated_static_grad = (static_grad * weights[None, :, None]).sum(dim=1)
    integrated_dynamic_grad = (dynamic_grad * weights[None, :, None, None]).sum(dim=1)
    static_values = (static_delta * integrated_static_grad).mean(dim=0).detach().cpu().numpy()
    dynamic_values = (
        dynamic_delta * integrated_dynamic_grad * background_mask.unsqueeze(-1)
    ).mean(dim=0).sum(dim=0).detach().cpu().numpy()
    attribution = {
        str(feature): float(static_values[idx])
        for idx, feature in enumerate(data.static_features)
    }
    for idx, feature in enumerate(data.dynamic_features):
        attribution[str(feature)] = attribution.get(str(feature), 0.0) + float(dynamic_values[idx])
    with torch.no_grad():
        baseline_logits, _embedding = model(background_static, background_dynamic, background_mask)
        baseline_probability = float(
            torch.softmax(
                apply_temperature_to_logits(baseline_logits, temperature), dim=1
            )[:, int(class_index)].mean().cpu().item()
        )
        current_logits, _embedding = model(query_static, query_dynamic, query_mask)
        current_probability = float(
            torch.softmax(
                apply_temperature_to_logits(current_logits, temperature), dim=1
            )[0, int(class_index)].cpu().item()
        )
    attribution_sum = float(sum(attribution.values()))
    return {
        "attributions": attribution,
        "baseline_probability": baseline_probability,
        "current_probability": current_probability,
        "attribution_sum": attribution_sum,
        "completeness_residual": current_probability - baseline_probability - attribution_sum,
        "background_size": background_size,
        "attribution_scope": "clinician_entered_features_only" if attribution_features is not None else "all_model_features",
    }


def _predict_target_oof(patient: Mapping[str, Any], records: Sequence[Mapping[str, Any]], target: str) -> tuple[dict[str, Any], list[str]]:
    if target == "fsh":
        raise ValueError("FSH three-class OOF bundle is unavailable; use holdout deployment")
    phase866 = _phase866_module()
    spec = TARGETS[target]
    labels = list(spec["labels"])
    phase866.LABELS = labels
    phase866.NUM_CLASSES = len(labels)
    train_raw = _cached_target_training_raw(target)
    query_frame, warnings = _query_rows(patient, records, target=target)
    query_raw = phase866.phase864.build_daily_sequence_arrays(
        query_frame.sort_values(["cycle_uid", "Day", "visit_uid"], kind="mergesort").reset_index(drop=True),
        static_features=list(_checkpoint(target, 1)["static_features"]),
        dynamic_features=list(_checkpoint(target, 1)["dynamic_features"]),
        max_history_days=16,
    )
    raw = _combine_raw(train_raw, query_raw)
    groups = train_raw.row_index["cycle_uid"].astype(str).to_numpy()
    train_y = np.asarray(train_raw.y, dtype=int)
    query_index = len(raw.y) - 1
    fold_probs: list[np.ndarray] = []
    fold_models: list[tuple[torch.nn.Module, Any]] = []
    for fold, outer_train_idx, _heldout_idx in phase866.grouped_oof_splits(groups, train_y, 5):
        outer_train = np.zeros(len(train_y), dtype=bool)
        outer_train[outer_train_idx] = True
        inner_train_short, _inner_valid = phase866.phase864._inner_train_valid_masks(groups, outer_train, seed=20260701, fold=int(fold))
        fit_mask = np.concatenate([inner_train_short, np.zeros(len(query_raw.y), dtype=bool)])
        data = phase866.phase864.prepare_arrays(raw, fit_mask=fit_mask)
        ckpt = _checkpoint(target, int(fold))
        args_payload = dict(ckpt.get("args") or {})
        args_payload.setdefault("hidden_dim", 96)
        args_payload.setdefault("dropout", 0.15)
        args = SimpleNamespace(**args_payload)
        model = phase866.make_neural_model("gru_addgate", data, args)
        model.load_state_dict(ckpt["state_dict"])
        fold_probs.append(_predict_softmax(model, data, query_index))
        fold_models.append((model, data))
    probs = np.mean(np.vstack(fold_probs), axis=0)
    pred_idx = int(np.argmax(probs))
    label = labels[pred_idx]
    probabilities = {class_label: float(probs[i]) for i, class_label in enumerate(labels)}
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    fold_explanations = [
        _selected_class_local_attribution(
            model,
            data,
            query_index,
            pred_idx,
            attribution_features=set(UI_INPUT_ATTRIBUTION_FEATURES),
        )
        for model, data in fold_models
    ]
    fold_attributions = [row["attributions"] for row in fold_explanations]
    attribution_features = sorted(set().union(*(row.keys() for row in fold_attributions)))
    local_attributions = [
        {
            "feature": feature,
            "attribution": float(np.mean([row.get(feature, 0.0) for row in fold_attributions])),
        }
        for feature in attribution_features
    ]
    local_attributions.sort(key=lambda row: abs(float(row["attribution"])), reverse=True)
    attribution_baseline_probability = float(
        np.mean([row["baseline_probability"] for row in fold_explanations])
    )
    attribution_sum = float(sum(row["attribution"] for row in local_attributions))
    attribution_residual = float(probs[pred_idx] - attribution_baseline_probability - attribution_sum)
    return {
        "target": target,
        "label": label,
        "display": spec["display"].get(label, label),
        "dose": float(spec["dose"].get(label, 0.0)),
        "probability": float(probs[pred_idx]),
        "probabilities": probabilities,
        "top_labels": [{"label": item[0], "display": spec["display"].get(item[0], item[0]), "dose": float(spec["dose"].get(item[0], 0.0)), "probability": float(item[1])} for item in ranked],
        "local_attributions": local_attributions,
        "attribution_baseline_probability": attribution_baseline_probability,
        "attribution_sum": attribution_sum,
        "attribution_residual": attribution_residual,
        "attribution_background_size": int(round(np.mean([row["background_size"] for row in fold_explanations]))),
        "attribution_method": "current_patient_selected_probability_expected_integrated_gradients",
        "attribution_scope": "clinician_entered_features_only",
        "attribution_baseline_definition": "matched training background with non-entered model context fixed at the current patient value",
    }, warnings


def _predict_target_holdout(
    patient: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    target: str,
) -> tuple[dict[str, Any], list[str]]:
    """Run the validation-frozen UI-reduced checkpoint with train-only preprocessing."""
    phase866 = _phase866_module()
    spec = TARGETS[target]
    labels = list(spec["labels"])
    phase866.LABELS = labels
    phase866.NUM_CLASSES = len(labels)
    train_raw = _cached_target_training_raw(target)
    query_frame, warnings = _query_rows(patient, records, target=target)
    checkpoint = _holdout_checkpoint(target)
    query_raw = phase866.phase864.build_daily_sequence_arrays(
        query_frame.sort_values(["cycle_uid", "Day", "visit_uid"], kind="mergesort").reset_index(drop=True),
        static_features=list(checkpoint["static_features"]),
        dynamic_features=list(checkpoint["dynamic_features"]),
        max_history_days=16,
    )
    raw = _combine_raw(train_raw, query_raw)

    split_lookup = _split_manifest().set_index("cycle_uid")["split"]
    train_splits = train_raw.row_index["cycle_uid"].astype(str).map(split_lookup)
    if train_splits.isna().any():
        missing = int(train_splits.isna().sum())
        raise ValueError(f"{target}: {missing} training rows are absent from the frozen split manifest")
    fit_mask = np.concatenate(
        [train_splits.eq("train").to_numpy(dtype=bool), np.zeros(len(query_raw.y), dtype=bool)]
    )
    data = phase866.phase864.prepare_arrays(raw, fit_mask=fit_mask)
    hyperparameters = dict(checkpoint.get("hyperparameters") or {})
    args = SimpleNamespace(
        hidden_dim=int(hyperparameters.get("hidden_dim", 96)),
        dropout=float(hyperparameters.get("dropout", 0.15)),
    )
    model = phase866.make_neural_model("gru_addgate", data, args)
    model.load_state_dict(checkpoint["state_dict"])

    query_index = len(raw.y) - 1
    calibration = target_calibration(target, labels, _holdout_checkpoint_path(target))
    temperature = float(calibration["deployed_temperature"])
    probs = _predict_softmax(model, data, query_index, temperature=temperature)
    pred_idx = int(np.argmax(probs))
    label = labels[pred_idx]
    probabilities = {class_label: float(probs[i]) for i, class_label in enumerate(labels)}
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    explanation = _selected_class_local_attribution(
        model,
        data,
        query_index,
        pred_idx,
        background_indices=np.flatnonzero(fit_mask),
        attribution_features=set(UI_INPUT_ATTRIBUTION_FEATURES),
        temperature=temperature,
    )
    local_attributions = [
        {"feature": feature, "attribution": float(value)}
        for feature, value in explanation["attributions"].items()
    ]
    local_attributions.sort(key=lambda row: abs(float(row["attribution"])), reverse=True)
    attribution_sum = float(sum(row["attribution"] for row in local_attributions))
    return {
        "target": target,
        "label": label,
        "display": spec["display"].get(label, label),
        "dose": float(spec["dose"].get(label, 0.0)),
        "probability": float(probs[pred_idx]),
        "probabilities": probabilities,
        "top_labels": [
            {
                "label": item[0],
                "display": spec["display"].get(item[0], item[0]),
                "dose": float(spec["dose"].get(item[0], 0.0)),
                "probability": float(item[1]),
            }
            for item in ranked
        ],
        "local_attributions": local_attributions,
        "attribution_baseline_probability": float(explanation["baseline_probability"]),
        "attribution_sum": attribution_sum,
        "attribution_residual": float(probs[pred_idx] - explanation["baseline_probability"] - attribution_sum),
        "attribution_background_size": int(explanation["background_size"]),
        "attribution_method": "holdout_train_background_selected_probability_expected_integrated_gradients",
        "attribution_scope": "clinician_entered_features_only",
        "attribution_baseline_definition": "matched train-only background with non-entered model context fixed at the current patient value",
        "checkpoint": str(_holdout_checkpoint_path(target)),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "best_validation_weighted_f1": float(checkpoint.get("best_validation_weighted_f1", np.nan)),
        "probability_calibrated": bool(calibration["calibration_applied"]),
        "calibration_method": str(calibration["method"]),
        "calibration_temperature": temperature,
        "calibration_status": str(calibration["calibration_status"]),
        "calibration_artifact": str(
            Path("models")
            / "layer1_strategy"
            / "calibration"
            / "ui_reduced_holdout"
            / f"{target}_temperature.json"
        ),
    }, warnings


def _predict_target(
    patient: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    target: str,
    *,
    deployment_mode: str = DEFAULT_DEPLOYMENT_MODE,
) -> tuple[dict[str, Any], list[str]]:
    if deployment_mode == "holdout":
        return _predict_target_holdout(patient, records, target)
    if deployment_mode == "oof":
        return _predict_target_oof(patient, records, target)
    raise ValueError(f"Unsupported dose-model deployment mode: {deployment_mode}")


def predict_ui_reduced_dose_context(
    patient: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | None = None,
    *,
    deployment_mode: str = DEFAULT_DEPLOYMENT_MODE,
) -> dict[str, Any]:
    """Predict next recorded absolute dose categories with the selected deployment protocol."""
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"UI-reduced GRU(AddGate) model root not found: {MODEL_ROOT}")
    active_records = list(records or [])
    predictions: dict[str, Any] = {}
    warnings: list[str] = []
    for target in ("fsh", "lh", "hmg"):
        prediction, target_warnings = _predict_target(
            patient,
            active_records,
            target,
            deployment_mode=deployment_mode,
        )
        predictions[target] = prediction
        warnings.extend(target_warnings)
    is_holdout = deployment_mode == "holdout"
    calibration_summary = {
        target: {
            "applied": bool(predictions[target].get("probability_calibrated", False)),
            "method": str(predictions[target].get("calibration_method") or "none"),
            "temperature": float(predictions[target].get("calibration_temperature", 1.0)),
            "status": str(predictions[target].get("calibration_status") or "not_applicable"),
        }
        for target in ("fsh", "lh", "hmg")
    }
    return {
        "source": "phase870_ui_reduced_gru_addgate_holdout_fsh3" if is_holdout else "phase870_ui_reduced_gru_addgate_oof_ensemble",
        "model": "GRU(AddGate)",
        "deployment_mode": deployment_mode,
        "inference_protocol": (
            "train-fit validation-selected single checkpoint; independent test evaluated once"
            if is_holdout
            else "cycle-grouped five-fold OOF checkpoint ensemble"
        ),
        "model_root": str(HOLDOUT_MODEL_ROOT if is_holdout else MODEL_ROOT),
        "target_model_paths": {
            target: str(_holdout_checkpoint_path(target) if is_holdout else _checkpoint_path(target, 1))
            for target in ("fsh", "lh", "hmg")
        },
        "contract_path": str(CONTRACT_PATH),
        "task": "next-recorded absolute dose category prediction",
        "probability_calibration": calibration_summary,
        "predictions": predictions,
        "warnings": sorted(set(warnings)),
    }


def ui_reduced_model_available(deployment_mode: str = DEFAULT_DEPLOYMENT_MODE) -> tuple[bool, str]:
    try:
        load_ui_reduced_contract()
        for target in ("fsh", "lh", "hmg"):
            if deployment_mode == "holdout":
                _holdout_checkpoint(target)
                target_calibration(
                    target,
                    TARGETS[target]["labels"],
                    _holdout_checkpoint_path(target),
                )
            elif deployment_mode == "oof":
                _checkpoint(target, 1)
            else:
                raise ValueError(f"Unsupported dose-model deployment mode: {deployment_mode}")
    except Exception as exc:
        return False, str(exc)
    return True, ""
