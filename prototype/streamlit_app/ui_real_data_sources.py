from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SHAP_TABLE_DIR = REPO_ROOT / "paper" / "tables" / "paper_claim_validation" / "phase867_gru_addgate_shap"
SHAP_TARGET_FILES = {
    "fsh": "gru_addgate_fsh_gradient_shap_summary.csv",
    "lh": "gru_addgate_lh_gradient_shap_summary.csv",
    "hmg": "gru_addgate_hmg_gradient_shap_summary.csv",
}
SHAP_LOCAL_VALUE_FILES = {
    "fsh": "gru_addgate_fsh_gradient_shap_values_long.csv",
    "lh": "gru_addgate_lh_gradient_shap_values_long.csv",
    "hmg": "gru_addgate_hmg_gradient_shap_values_long.csv",
}
DOSE_BACKGROUND_PROBABILITY_PATH = REPO_ROOT / "paper" / "tables" / "paper_claim_validation" / "phase870_ui_variable_validation" / "full_vs_ui_reduced_oof_probabilities.csv"
DOSE_PROBABILITY_COLUMNS = {
    "fsh": {
        "stop": "prob_stop",
        "low_dose": "prob_low_dose",
        "medium_low": "prob_medium_low",
        "medium_high": "prob_medium_high",
        "high_dose": "prob_high_dose",
    },
    "lh": {"dose_0": "prob_dose_0", "dose_75": "prob_dose_75", "dose_ge150": "prob_dose_ge150"},
    "hmg": {"dose_0": "prob_dose_0", "dose_75": "prob_dose_75", "dose_150": "prob_dose_150", "dose_ge225": "prob_dose_ge225"},
}
TARGET_LABELS = {"fsh": "FSH", "lh": "LH", "hmg": "HMG"}
TECHNICAL_DETAIL_LIMIT = 4
FEATURE_LABELS_CN = {
    "Day": "促排天数",
    "gn_day": "促排天数",
    "evaluation_day": "评估天数",
    "cycle_day": "周期天数",
    "monitoring_order": "监测序次",
    "visits_seen": "已监测次数",
    "days_since_previous_visit": "距上次监测天数",
    "previous_fsh_daily_dose": "既往 FSH 剂量",
    "previous_lh_daily_dose": "既往 LH 剂量",
    "previous_hmg_daily_dose": "既往 HMG 剂量",
    "previous_lh_like_hmg_daily_dose": "既往 LH-like 剂量",
    "previous_gn_dose": "既往总 Gn 剂量",
    "current_e2": "E2",
    "current_lh": "血清 LH",
    "current_p": "P",
    "current_fsh": "血清 FSH",
    "current_endometrium": "内膜厚度",
    "delta_e2": "E2 变化",
    "delta_lh": "LH 变化",
    "delta_p": "P 变化",
    "delta_fsh": "FSH 变化",
    "delta_endometrium": "内膜变化",
    "total_follicle_count": "总卵泡数",
    "left_follicle_count": "左侧卵泡数",
    "right_follicle_count": "右侧卵泡数",
    "mean_follicle_diameter": "平均卵泡直径",
    "max_follicle_diameter": "最大卵泡直径",
    "mature_follicle_count": "成熟卵泡数",
    "mature_follicle_share": "成熟卵泡占比",
    "large_follicle_share": "大卵泡占比",
    "dominant_follicle_count": "优势卵泡数",
    "follicle_count_lt_10": "<10mm 卵泡数",
    "follicle_count_10_12": "10-12mm 卵泡数",
    "follicle_count_13_15": "13-15mm 卵泡数",
    "follicle_count_16_18": "16-18mm 卵泡数",
    "follicle_count_gt_18": "≥18mm 卵泡数",
    "growing_follicle_count": "生长卵泡数",
    "medium_plus_follicle_count": "中大卵泡数",
    "ohss_follicle_load_score": "卵泡负荷评分",
    "follicle_size_weighted_count": "卵泡直径加权数",
    "follicle_maturity_index": "卵泡成熟指数",
    "mid_follicle_share": "中等卵泡占比",
    "large_to_mature_follicle_ratio": "大/成熟卵泡比",
    "e2_per_mature_follicle": "单成熟卵泡 E2",
    "e2_per_weighted_follicle": "加权卵泡 E2",
    "p_lh_ratio": "P/LH 比值",
    "delta_p_per_day": "每日 P 变化",
    "delta_lh_per_day": "每日 LH 变化",
    "age": "年龄",
    "bmi": "BMI",
    "infertility_duration": "不孕年限",
    "amh": "AMH",
    "afc": "AFC",
    "initial_gn_dose": "起始 Gn 剂量",
    "basal_fsh": "基础 FSH",
    "basal_lh": "基础 LH",
    "basal_e2": "基础 E2",
    "basal_p": "基础 P",
    "male_age": "男方年龄",
    "male_factor_infertility_flag": "男方因素",
    "treatment_count": "治疗次数",
    "fresh_treatment_count": "既往鲜胚治疗次数",
    "baseline_only_feature_row": "仅基础信息行",
}
CLINICAL_DISPLAY_FEATURES = {
    "previous_fsh_daily_dose",
    "previous_lh_daily_dose",
    "previous_hmg_daily_dose",
    "previous_lh_like_hmg_daily_dose",
    "previous_gn_dose",
    "current_e2",
    "current_lh",
    "current_p",
    "current_fsh",
    "current_endometrium",
    "total_follicle_count",
    "max_follicle_diameter",
    "mean_follicle_diameter",
    "follicle_count_lt_10",
    "follicle_count_10_12",
    "follicle_count_13_15",
    "follicle_count_16_18",
    "follicle_count_gt_18",
    "mature_follicle_count",
    "dominant_follicle_count",
    "amh",
    "afc",
    "initial_gn_dose",
    "basal_fsh",
    "basal_lh",
    "basal_e2",
    "basal_p",
    "age",
    "bmi",
    "infertility_duration",
    "Day",
    "gn_day",
    "evaluation_day",
    "monitoring_order",
    "visits_seen",
}
UI_INPUT_ATTRIBUTION_FEATURES = {
    "age",
    "bmi",
    "infertility_duration",
    "amh",
    "afc",
    "basal_fsh",
    "basal_lh",
    "basal_e2",
    "basal_p",
    "Day",
    "gn_day",
    "days_since_previous_visit",
    "current_e2",
    "current_lh",
    "current_p",
    "current_fsh",
    "current_endometrium",
    "total_follicle_count",
    "left_follicle_count",
    "right_follicle_count",
    "max_follicle_diameter",
    "mean_follicle_diameter",
    "follicle_count_lt_10",
    "follicle_count_10_12",
    "follicle_count_13_15",
    "follicle_count_16_18",
    "follicle_count_gt_18",
    "previous_fsh_daily_dose",
    "previous_lh_daily_dose",
    "previous_hmg_daily_dose",
}
TARGET_CLINICAL_DISPLAY_FEATURES = {
    "fsh": {
        "previous_fsh_daily_dose",
        "previous_gn_dose",
        "current_e2",
        "current_fsh",
        "current_lh",
        "current_p",
        "current_endometrium",
        "total_follicle_count",
        "max_follicle_diameter",
        "mean_follicle_diameter",
        "follicle_count_lt_10",
        "follicle_count_10_12",
        "follicle_count_13_15",
        "follicle_count_16_18",
        "follicle_count_gt_18",
        "amh",
        "afc",
        "basal_fsh",
        "basal_lh",
        "age",
        "bmi",
        "Day",
        "monitoring_order",
    },
    "lh": {
        "previous_lh_daily_dose",
        "previous_lh_like_hmg_daily_dose",
        "current_lh",
        "current_e2",
        "current_p",
        "current_fsh",
        "current_endometrium",
        "total_follicle_count",
        "max_follicle_diameter",
        "mean_follicle_diameter",
        "follicle_count_13_15",
        "follicle_count_16_18",
        "follicle_count_gt_18",
        "amh",
        "afc",
        "basal_lh",
        "basal_fsh",
        "age",
        "bmi",
        "Day",
        "monitoring_order",
    },
    "hmg": {
        "previous_hmg_daily_dose",
        "previous_lh_like_hmg_daily_dose",
        "previous_gn_dose",
        "current_e2",
        "current_lh",
        "current_p",
        "current_endometrium",
        "total_follicle_count",
        "max_follicle_diameter",
        "mean_follicle_diameter",
        "follicle_count_lt_10",
        "follicle_count_10_12",
        "follicle_count_13_15",
        "follicle_count_16_18",
        "follicle_count_gt_18",
        "amh",
        "afc",
        "basal_lh",
        "basal_fsh",
        "age",
        "bmi",
        "Day",
        "monitoring_order",
    },
}


def _repo_path(repo_root: Path | str | None) -> Path:
    return Path(repo_root) if repo_root is not None else REPO_ROOT


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _fmt_num(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except Exception:
        return "--"
    if pd.isna(number):
        return "--"
    return f"{number:.0f}" if abs(number - round(number)) < 1e-8 else f"{number:.{digits}f}"


def _fmt_percent(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "--"
    if pd.isna(number):
        return "--"
    return f"{number * 100:.0f}%" if abs(number) <= 1.5 else f"{number:.1f}%"


def _display_feature_value(feature: str, value: Any) -> str:
    name = str(feature)
    if name in {"current_e2", "basal_e2", "delta_e2"}:
        return f"{_fmt_num(value)} pg/mL"
    if name in {"current_lh", "basal_lh", "current_fsh", "basal_fsh", "delta_lh", "delta_fsh"}:
        return f"{_fmt_num(value, 2)} IU/L"
    if name in {"current_p", "basal_p", "delta_p", "delta_p_per_day"}:
        return f"{_fmt_num(value, 2)} ng/mL"
    if name in {"current_endometrium", "delta_endometrium", "max_follicle_diameter", "mean_follicle_diameter"}:
        return f"{_fmt_num(value, 1)} mm"
    if name in {
        "previous_fsh_daily_dose",
        "previous_lh_daily_dose",
        "previous_hmg_daily_dose",
        "previous_lh_like_hmg_daily_dose",
        "previous_gn_dose",
    }:
        return f"{_fmt_num(value)} IU/天"
    if name in {
        "total_follicle_count",
        "left_follicle_count",
        "right_follicle_count",
        "mature_follicle_count",
        "dominant_follicle_count",
        "follicle_count_lt_10",
        "follicle_count_10_12",
        "follicle_count_13_15",
        "follicle_count_16_18",
        "follicle_count_gt_18",
        "growing_follicle_count",
        "medium_plus_follicle_count",
        "afc",
    }:
        return f"{_fmt_num(value)} 个"
    if name in {
        "mature_follicle_share",
        "large_follicle_share",
        "mid_follicle_share",
        "large_to_mature_follicle_ratio",
    }:
        return _fmt_percent(value)
    if name in {"age", "male_age", "infertility_duration", "Day", "gn_day", "evaluation_day", "cycle_day"}:
        unit = "岁" if name in {"age", "male_age"} else "年" if name == "infertility_duration" else "天"
        return f"{_fmt_num(value)} {unit}"
    if name in {"monitoring_order", "visits_seen", "treatment_count", "fresh_treatment_count"}:
        return f"{_fmt_num(value)} 次"
    if name == "bmi":
        return f"{_fmt_num(value, 1)} kg/m²"
    if name == "amh":
        return f"{_fmt_num(value, 2)} ng/mL"
    if name == "male_factor_infertility_flag":
        return "是" if _safe_float(value, 0.0) >= 0.5 else "否"
    return _fmt_num(value, 2)


def _display_feature_name(feature: str, feature_name: str | None) -> str:
    return FEATURE_LABELS_CN.get(str(feature), str(feature_name or feature))


def _direction(mean_shap: float) -> tuple[str, str, str]:
    if mean_shap > 0:
        return "平均正向贡献", "w", "cw"
    if mean_shap < 0:
        return "平均负向贡献", "t", "ct"
    return "平均贡献接近 0", "", "cp"


def _feature_group(feature: str) -> str:
    name = str(feature).lower()
    if "follicle" in name:
        return "Follicle features"
    if "previous" in name or "dose" in name or "gn_" in name:
        return "Medication history"
    if name in {"amh", "afc"} or name.startswith("basal_"):
        return "Baseline ovarian reserve"
    if name in {"age", "bmi"} or "infertility" in name:
        return "Baseline demographics"
    return "Dynamic monitoring indicators"


def _relative_source(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


@lru_cache(maxsize=4)
def _load_phase870_dose_probability_baselines(repo_root_text: str) -> dict[str, dict[str, float]]:
    root = Path(repo_root_text)
    path = root / DOSE_BACKGROUND_PROBABILITY_PATH.relative_to(REPO_ROOT)
    if not path.exists():
        return {}
    probability_columns = sorted({column for columns in DOSE_PROBABILITY_COLUMNS.values() for column in columns.values()})
    frame = pd.read_csv(path, usecols=["target", "model_key", *probability_columns])
    frame = frame[frame["model_key"].astype(str).eq("ui_reduced")]
    baselines: dict[str, dict[str, float]] = {}
    for target, columns in DOSE_PROBABILITY_COLUMNS.items():
        subset = frame[frame["target"].astype(str).eq(target)]
        if subset.empty:
            continue
        target_values: dict[str, float] = {}
        for label, column in columns.items():
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            if not values.empty:
                target_values[label] = float(values.mean())
        if target_values:
            baselines[target] = target_values
    return baselines


def load_phase870_dose_probability_baselines(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Return class-specific mean OOF probabilities for the UI-reduced dose model."""
    root = _repo_path(repo_root)
    path = root / DOSE_BACKGROUND_PROBABILITY_PATH.relative_to(REPO_ROOT)
    baselines = _load_phase870_dose_probability_baselines(root.as_posix())
    return {
        "is_real": bool(baselines),
        "baselines": {target: dict(values) for target, values in baselines.items()},
        "source_path": _relative_source(path, root),
        "model_key": "ui_reduced",
        "definition": "mean OOF predicted probability by target dose class",
    }


def load_phase867_dose_shap_summary(repo_root: Path | str | None = None, limit: int = 4) -> dict[str, Any]:
    """Load real Phase 8.67 GRU(AddGate) Gradient SHAP summaries for FSH/LH/HMG."""
    root = _repo_path(repo_root)
    table_dir = root / "paper" / "tables" / "paper_claim_validation" / "phase867_gru_addgate_shap"
    targets: dict[str, Any] = {}
    group_totals: dict[str, float] = {}
    source_paths: list[str] = []
    for target, filename in SHAP_TARGET_FILES.items():
        path = table_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"feature", "mean_abs_shap", "mean_shap"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["mean_abs_shap"] = pd.to_numeric(frame["mean_abs_shap"], errors="coerce").fillna(0.0)
        frame["mean_shap"] = pd.to_numeric(frame["mean_shap"], errors="coerce").fillna(0.0)
        frame = frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        max_abs = max(float(frame["mean_abs_shap"].max()), 1e-12)
        items: list[dict[str, Any]] = []
        for _, row in frame.head(max(1, int(limit))).iterrows():
            mean_abs = float(row["mean_abs_shap"])
            mean_shap = float(row["mean_shap"])
            direction, fill, chip = _direction(mean_shap)
            width = int(round(30 + 58 * mean_abs / max_abs))
            width = max(24, min(88, width))
            feature = str(row.get("feature", ""))
            items.append(
                {
                    "feature": feature,
                    "label": _display_feature_name(feature, str(row.get("feature_name", feature))),
                    "direction": direction,
                    "width": width,
                    "fill": fill,
                    "chip": chip,
                    "mean_abs_shap": mean_abs,
                    "mean_shap": mean_shap,
                }
            )
        for _, row in frame.iterrows():
            group = _feature_group(str(row.get("feature", "")))
            group_totals[group] = group_totals.get(group, 0.0) + float(row["mean_abs_shap"])
        source_paths.append(_relative_source(path, root))
        sample_count = _safe_int(frame["sample_count"].max(), 0) if "sample_count" in frame else 0
        targets[target] = {
            "drug": TARGET_LABELS[target],
            "source_path": _relative_source(path, root),
            "sample_count": sample_count,
            "items": items,
        }
    if not targets:
        return {"is_real": False, "targets": {}, "groups": [], "source_paths": []}
    max_group = max(group_totals.values()) if group_totals else 1.0
    groups = [
        {"label": label, "width": int(round(24 + 60 * value / max_group)), "value": float(value)}
        for label, value in sorted(group_totals.items(), key=lambda item: item[1], reverse=True)[:5]
    ]
    return {"is_real": True, "targets": targets, "groups": groups, "source_paths": source_paths}


def _first_present(patient: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = patient.get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return value
    return default


def _ratio(num: float, den: float, default: float = 0.0) -> float:
    return default if abs(den) < 1e-12 else float(num) / float(den)


def _patient_raw_feature_values(patient: Mapping[str, Any]) -> dict[str, float]:
    """Map the current Streamlit patient snapshot to Phase 8.67 feature names."""
    age = _safe_float(_first_present(patient, "age"), 0.0)
    bmi = _safe_float(_first_present(patient, "bmi"), 0.0)
    years = _safe_float(_first_present(patient, "years", "infertility_duration"), 0.0)
    amh = _safe_float(_first_present(patient, "amh"), 0.0)
    afc = _safe_float(_first_present(patient, "afc"), 0.0)
    initial_gn = _safe_float(_first_present(patient, "initial_gn", "initial_gn_dose"), 0.0)
    day = _safe_float(_first_present(patient, "stim_day", "gn_day", "Day", "evaluation_day"), 0.0)
    visit = _safe_float(_first_present(patient, "visit", "monitoring_order", "visits_seen"), 0.0)
    e2 = _safe_float(_first_present(patient, "e2", "current_e2"), 0.0)
    lh = _safe_float(_first_present(patient, "lh_value", "current_lh"), 0.0)
    p_value = _safe_float(_first_present(patient, "p", "current_p"), 0.0)
    serum_fsh = _safe_float(_first_present(patient, "serum_fsh", "current_fsh"), 0.0)
    endometrium = _safe_float(_first_present(patient, "current_endometrium"), 0.0)
    f_lt10 = _safe_float(_first_present(patient, "f_lt10", "follicle_count_lt_10"), 0.0)
    f_10_12 = _safe_float(_first_present(patient, "f_10_12", "follicle_count_10_12"), 0.0)
    f_13_15 = _safe_float(_first_present(patient, "f_13_15", "follicle_count_13_15"), 0.0)
    f_16_18 = _safe_float(_first_present(patient, "f_16_18", "follicle_count_16_18"), 0.0)
    f_gt18 = _safe_float(_first_present(patient, "f_gt18", "follicle_count_gt_18"), 0.0)
    binned_total = f_lt10 + f_10_12 + f_13_15 + f_16_18 + f_gt18
    explicit_total = _first_present(patient, "total_follicles", "total_follicle_count")
    total = _safe_float(explicit_total, binned_total) if explicit_total is not None else binned_total
    max_f = _safe_float(_first_present(patient, "max_f", "max_follicle_diameter"), 0.0)
    mean_f = _safe_float(_first_present(patient, "mean_f", "mean_follicle_diameter"), 0.0)
    prev_fsh = _safe_float(_first_present(patient, "reference_fsh", "previous_fsh_daily_dose", "current_fsh"), 0.0)
    prev_lh = _safe_float(_first_present(patient, "reference_lh", "previous_lh_daily_dose", "current_lh"), 0.0)
    prev_hmg = _safe_float(_first_present(patient, "reference_hmg", "previous_hmg_daily_dose", "current_hmg"), 0.0)
    previous_lh_like = prev_lh + prev_hmg
    previous_gn = prev_fsh + prev_lh + prev_hmg
    mature = f_16_18 + f_gt18
    large = f_13_15 + f_16_18 + f_gt18
    growing = f_10_12 + f_13_15 + f_16_18 + f_gt18
    medium_plus = large
    weighted = f_lt10 * 8.0 + f_10_12 * 11.0 + f_13_15 * 14.0 + f_16_18 * 17.0 + f_gt18 * 19.0
    male_factor = str(_first_present(patient, "male_factor_infertility", default="否")).strip().lower()
    male_flag = 0.0 if male_factor in {"", "0", "false", "no", "否", "无", "none", "nan"} else 1.0
    values = {
        "age": age,
        "bmi": bmi,
        "infertility_duration": years,
        "amh": amh,
        "afc": afc,
        "initial_gn_dose": initial_gn,
        "basal_fsh": _safe_float(_first_present(patient, "basal_fsh"), 0.0),
        "basal_lh": _safe_float(_first_present(patient, "basal_lh"), 0.0),
        "basal_e2": _safe_float(_first_present(patient, "basal_e2"), 0.0),
        "basal_p": _safe_float(_first_present(patient, "basal_p"), 0.0),
        "male_age": _safe_float(_first_present(patient, "male_age"), 0.0),
        "male_factor_infertility_flag": male_flag,
        "Day": day,
        "treatment_count": _safe_float(_first_present(patient, "treatment_count"), 1.0),
        "fresh_treatment_count": _safe_float(_first_present(patient, "fresh_treatment_count", "treatment_count"), 1.0),
        "monitoring_order": visit,
        "cycle_day": day,
        "gn_day": day,
        "current_e2": e2,
        "current_lh": lh,
        "current_p": p_value,
        "current_fsh": serum_fsh,
        "current_endometrium": endometrium,
        "delta_e2": _safe_float(_first_present(patient, "delta_e2"), 0.0),
        "delta_lh": _safe_float(_first_present(patient, "delta_lh"), 0.0),
        "delta_p": _safe_float(_first_present(patient, "delta_p"), 0.0),
        "delta_fsh": _safe_float(_first_present(patient, "delta_fsh"), 0.0),
        "delta_endometrium": _safe_float(_first_present(patient, "delta_endometrium"), 0.0),
        "days_since_previous_visit": _safe_float(_first_present(patient, "days_since_previous_visit"), 0.0),
        "visits_seen": visit,
        "total_follicle_count": total,
        "left_follicle_count": _safe_float(_first_present(patient, "left_follicle_count"), total / 2.0),
        "right_follicle_count": _safe_float(_first_present(patient, "right_follicle_count"), total / 2.0),
        "max_follicle_diameter": max_f,
        "mean_follicle_diameter": mean_f,
        "mature_follicle_count": mature,
        "mature_follicle_share": _ratio(mature, total),
        "large_follicle_share": _ratio(large, total),
        "dominant_follicle_count": mature,
        "follicle_count_lt_10": f_lt10,
        "follicle_count_10_12": f_10_12,
        "follicle_count_13_15": f_13_15,
        "follicle_count_16_18": f_16_18,
        "follicle_count_gt_18": f_gt18,
        "growing_follicle_count": growing,
        "medium_plus_follicle_count": medium_plus,
        "ohss_follicle_load_score": total + 2.0 * mature,
        "follicle_size_weighted_count": weighted,
        "follicle_maturity_index": _ratio(weighted, max(total, 1.0)),
        "mid_follicle_share": _ratio(f_10_12 + f_13_15, total),
        "large_to_mature_follicle_ratio": _ratio(large, mature, 0.0),
        "e2_per_mature_follicle": _ratio(e2, max(mature, 1.0)),
        "e2_per_weighted_follicle": _ratio(e2, max(weighted, 1.0)),
        "p_lh_ratio": _ratio(p_value, max(lh, 1e-6)),
        "delta_p_per_day": _safe_float(_first_present(patient, "delta_p_per_day"), 0.0),
        "delta_lh_per_day": _safe_float(_first_present(patient, "delta_lh_per_day"), 0.0),
        "previous_fsh_daily_dose": prev_fsh,
        "previous_lh_daily_dose": prev_lh,
        "previous_hmg_daily_dose": prev_hmg,
        "previous_lh_like_hmg_daily_dose": previous_lh_like,
        "previous_gn_dose": previous_gn,
        "evaluation_day": day,
        "baseline_only_feature_row": 0.0,
    }
    return {key: float(value) for key, value in values.items() if value is not None}


def build_current_patient_dose_attribution_items(
    attributions: list[Mapping[str, Any]],
    patient: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Convert entered-feature probability attributions to clinician-facing rows."""
    raw_values = _patient_raw_feature_values(patient)
    grouped: dict[str, dict[str, Any]] = {}
    for item in attributions:
        feature = str(item.get("feature", ""))
        if feature not in UI_INPUT_ATTRIBUTION_FEATURES:
            continue
        canonical_feature = "Day" if feature == "gn_day" else feature
        value = _safe_float(item.get("attribution"), 0.0) * 100.0
        if canonical_feature not in grouped:
            grouped[canonical_feature] = {
                "feature": canonical_feature,
                "label": _display_feature_name(canonical_feature, canonical_feature),
                "value_label": _local_value_label(canonical_feature, raw_values, raw_values.get(canonical_feature)),
                "mean_shap": 0.0,
                "source": "phase870_current_patient_conditional_expected_integrated_gradients",
                "clinical_display": True,
            }
        grouped[canonical_feature]["mean_shap"] += value
    rows = list(grouped.values())
    for row in rows:
        value = float(row["mean_shap"])
        row["direction"] = "局部正向" if value >= 0 else "局部负向"
        row["mean_abs_shap"] = abs(value)
    rows.sort(key=lambda row: abs(float(row["mean_shap"])), reverse=True)
    return rows


@lru_cache(maxsize=4)
def _reference_feature_stats(repo_root_text: str) -> dict[str, tuple[float, float]]:
    root = Path(repo_root_text)
    path = root / "data" / "processed" / "layer1_strategy_dataset.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, low_memory=False)
    derived = pd.DataFrame(index=frame.index)
    aliases = {
        "Day": "gn_day",
        "evaluation_day": "gn_day",
        "cycle_day": "gn_day",
        "fresh_treatment_count": "treatment_count",
    }
    for feature, column in aliases.items():
        if column in frame.columns:
            derived[feature] = pd.to_numeric(frame[column], errors="coerce")
    if {"follicle_count_16_18", "follicle_count_gt_18"}.issubset(frame.columns):
        mature = pd.to_numeric(frame["follicle_count_16_18"], errors="coerce").fillna(0) + pd.to_numeric(frame["follicle_count_gt_18"], errors="coerce").fillna(0)
        derived["dominant_follicle_count"] = mature
        derived["mature_follicle_count"] = mature
    if {"follicle_count_lt_10", "follicle_count_10_12", "follicle_count_13_15", "follicle_count_16_18", "follicle_count_gt_18"}.issubset(frame.columns):
        f_lt10 = pd.to_numeric(frame["follicle_count_lt_10"], errors="coerce").fillna(0)
        f_10_12 = pd.to_numeric(frame["follicle_count_10_12"], errors="coerce").fillna(0)
        f_13_15 = pd.to_numeric(frame["follicle_count_13_15"], errors="coerce").fillna(0)
        f_16_18 = pd.to_numeric(frame["follicle_count_16_18"], errors="coerce").fillna(0)
        f_gt18 = pd.to_numeric(frame["follicle_count_gt_18"], errors="coerce").fillna(0)
        weighted = f_lt10 * 8.0 + f_10_12 * 11.0 + f_13_15 * 14.0 + f_16_18 * 17.0 + f_gt18 * 19.0
        derived["follicle_size_weighted_count"] = weighted
        derived["growing_follicle_count"] = f_10_12 + f_13_15 + f_16_18 + f_gt18
        derived["medium_plus_follicle_count"] = f_13_15 + f_16_18 + f_gt18
    if {"total_follicle_count", "follicle_count_16_18", "follicle_count_gt_18"}.issubset(frame.columns):
        total = pd.to_numeric(frame["total_follicle_count"], errors="coerce").replace(0, pd.NA)
        mature = pd.to_numeric(frame["follicle_count_16_18"], errors="coerce").fillna(0) + pd.to_numeric(frame["follicle_count_gt_18"], errors="coerce").fillna(0)
        derived["ohss_follicle_load_score"] = pd.to_numeric(frame["total_follicle_count"], errors="coerce").fillna(0) + 2.0 * mature
        derived["mature_follicle_share"] = mature / total
    if {"current_e2", "follicle_count_16_18", "follicle_count_gt_18"}.issubset(frame.columns):
        mature = pd.to_numeric(frame["follicle_count_16_18"], errors="coerce").fillna(0) + pd.to_numeric(frame["follicle_count_gt_18"], errors="coerce").fillna(0)
        derived["e2_per_mature_follicle"] = pd.to_numeric(frame["current_e2"], errors="coerce") / mature.replace(0, pd.NA)
    if {"current_p", "current_lh"}.issubset(frame.columns):
        derived["p_lh_ratio"] = pd.to_numeric(frame["current_p"], errors="coerce") / pd.to_numeric(frame["current_lh"], errors="coerce").replace(0, pd.NA)
    stats: dict[str, tuple[float, float]] = {}
    for source in (frame, derived):
        for column in source.columns:
            series = pd.to_numeric(source[column], errors="coerce")
            if series.notna().sum() < 2:
                continue
            mean = float(series.mean())
            std = float(series.std())
            if pd.isna(mean) or pd.isna(std) or std <= 1e-12:
                continue
            stats[str(column)] = (mean, std)
    return stats


@lru_cache(maxsize=8)
def _load_local_shap_values(repo_root_text: str, target: str) -> pd.DataFrame:
    root = Path(repo_root_text)
    filename = SHAP_LOCAL_VALUE_FILES[target]
    path = root / "paper" / "tables" / "paper_claim_validation" / "phase867_gru_addgate_shap" / filename
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"sample_id", "feature", "feature_value", "shap_value"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    frame["feature_value"] = pd.to_numeric(frame["feature_value"], errors="coerce")
    frame["shap_value"] = pd.to_numeric(frame["shap_value"], errors="coerce").fillna(0.0)
    return frame


def _scaled_patient_values(raw_values: Mapping[str, float], stats: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    scaled: dict[str, float] = {}
    for feature, raw in raw_values.items():
        stat = stats.get(feature)
        if stat is None:
            continue
        mean, std = stat
        scaled[feature] = (float(raw) - mean) / std
    return scaled


def _local_value_label(feature: str, raw_values: Mapping[str, float], fallback: Any) -> str:
    if feature in raw_values:
        return _display_feature_value(feature, raw_values[feature])
    return f"匹配值 {_fmt_num(fallback)}"


def _rank_local_display_rows(sample_rows: pd.DataFrame, target: str, limit: int) -> pd.DataFrame:
    rows = sample_rows.copy()
    rows["is_clinical_display"] = rows["feature"].astype(str).isin(UI_INPUT_ATTRIBUTION_FEATURES)
    return rows[rows["is_clinical_display"]].sort_values("abs_shap", ascending=False).head(max(1, int(limit)))


def _rank_technical_detail_rows(sample_rows: pd.DataFrame, target: str, selected_features: set[str]) -> pd.DataFrame:
    clinical_features = CLINICAL_DISPLAY_FEATURES | TARGET_CLINICAL_DISPLAY_FEATURES.get(target, set())
    rows = sample_rows.copy()
    row_features = rows["feature"].astype(str)
    rows = rows[~row_features.isin(clinical_features) & ~row_features.isin(selected_features)]
    return rows.sort_values("abs_shap", ascending=False).head(TECHNICAL_DETAIL_LIMIT)


def _local_item_from_row(
    row: Mapping[str, Any],
    raw_values: Mapping[str, float],
    max_abs: float,
    sample_id: str,
    *,
    clinical_display: bool,
    source: str,
) -> dict[str, Any]:
    feature = str(row.get("feature", ""))
    shap_value = float(row.get("shap_value", 0.0))
    abs_value = abs(shap_value)
    direction, fill, chip = _direction(shap_value)
    width = int(round(30 + 58 * abs_value / max(max_abs, 1e-12)))
    return {
        "feature": feature,
        "label": _display_feature_name(feature, str(row.get("feature_name", feature))),
        "value_label": _local_value_label(feature, raw_values, row.get("feature_value")),
        "direction": direction.replace("平均", "局部"),
        "width": max(24, min(88, width)),
        "fill": fill,
        "chip": chip,
        "mean_abs_shap": abs_value,
        "mean_shap": shap_value,
        "source": source,
        "sample_id": sample_id,
        "clinical_display": clinical_display,
    }


def load_phase867_local_dose_shap_for_patient(
    patient: Mapping[str, Any],
    repo_root: Path | str | None = None,
    limit: int = 10,
    min_match_features: int = 3,
) -> dict[str, Any]:
    """Return true per-sample Phase 8.67 local attribution rows matched to the current patient.

    The Phase 8.67 artifacts store real held-out OOF per-sample Gradient SHAP-style
    attributions. Streamlit selects the nearest available OOF sample in standardized
    feature space and renders that sample's actual ``shap_value`` rows.
    """
    root = _repo_path(repo_root)
    root_text = root.as_posix()
    raw_values = _patient_raw_feature_values(patient)
    stats = _reference_feature_stats(root_text)
    scaled_values = _scaled_patient_values(raw_values, stats)
    targets: dict[str, Any] = {}
    source_paths: list[str] = []
    matched_any = False
    for target, filename in SHAP_LOCAL_VALUE_FILES.items():
        frame = _load_local_shap_values(root_text, target)
        path = root / "paper" / "tables" / "paper_claim_validation" / "phase867_gru_addgate_shap" / filename
        if frame.empty:
            continue
        match_features = [feature for feature in scaled_values if feature in set(frame["feature"].unique())]
        subset = frame[frame["feature"].isin(match_features)]
        if subset.empty:
            continue
        pivot = subset.pivot_table(index="sample_id", columns="feature", values="feature_value", aggfunc="mean")
        common = [feature for feature in match_features if feature in pivot.columns]
        if not common:
            continue
        current = pd.Series({feature: scaled_values[feature] for feature in common})
        diff = pivot[common].sub(current, axis="columns").pow(2)
        valid_counts = diff.notna().sum(axis=1)
        min_features = min(max(1, int(min_match_features)), len(common))
        distances = diff.mean(axis=1).pow(0.5)
        distances = distances[valid_counts >= min_features].dropna()
        if distances.empty:
            continue
        sample_id = str(distances.sort_values().index[0])
        sample_distance = float(distances.loc[sample_id])
        sample_rows = frame[frame["sample_id"].astype(str) == sample_id].copy()
        sample_rows["abs_shap"] = sample_rows["shap_value"].abs()
        sample_rows = sample_rows.sort_values("abs_shap", ascending=False)
        max_abs = max(float(sample_rows["abs_shap"].max()), 1e-12)
        display_rows = _rank_local_display_rows(sample_rows, target, limit)
        selected_features = set(display_rows["feature"].astype(str))
        technical_rows = _rank_technical_detail_rows(sample_rows, target, selected_features)
        remaining_rows = sample_rows[
            sample_rows["feature"].astype(str).isin(UI_INPUT_ATTRIBUTION_FEATURES)
            & ~sample_rows["feature"].astype(str).isin(selected_features)
        ].sort_values("abs_shap", ascending=False)
        clinical_features = UI_INPUT_ATTRIBUTION_FEATURES
        items: list[dict[str, Any]] = []
        for _, row in display_rows.iterrows():
            items.append(
                _local_item_from_row(
                    row,
                    raw_values,
                    max_abs,
                    sample_id,
                    clinical_display=bool(row.get("is_clinical_display", False)),
                    source="phase867_oof_local_attribution",
                )
            )
        technical_items = [
            _local_item_from_row(
                row,
                raw_values,
                max_abs,
                sample_id,
                clinical_display=False,
                source="phase867_oof_local_attribution_technical_detail",
            )
            for _, row in technical_rows.iterrows()
        ]
        remaining_items = [
            _local_item_from_row(
                row,
                raw_values,
                max_abs,
                sample_id,
                clinical_display=str(row.get("feature", "")) in clinical_features,
                source="phase867_oof_local_attribution_remaining",
            )
            for _, row in remaining_rows.iterrows()
        ]
        source_paths.append(_relative_source(path, root))
        matched_any = True
        targets[target] = {
            "drug": TARGET_LABELS[target],
            "source_path": _relative_source(path, root),
            "sample_id": sample_id,
            "distance": sample_distance,
            "match_feature_count": len(common),
            "sample_count": int(frame["sample_id"].nunique()),
            "clinical_display_count": int(sum(bool(item.get("clinical_display")) for item in items)),
            "items": items,
            "remaining_items": remaining_items,
            "technical_items": technical_items,
            "all_shap_sum": float(sample_rows["shap_value"].sum()),
            "all_shap_abs_sum": float(sample_rows["shap_value"].abs().sum()),
            "all_feature_count": int(len(sample_rows)),
        }
    return {
        "is_real": matched_any,
        "is_local": matched_any,
        "method": "phase867_nearest_oof_local_attribution",
        "targets": targets,
        "source_paths": source_paths,
        "raw_feature_count": len(raw_values),
        "scaled_feature_count": len(scaled_values),
    }


def _dose_category(drug: str, dose: float) -> str:
    if drug == "fsh":
        if dose <= 0:
            return "stop"
        if dose <= 80:
            return "low_dose"
        if dose <= 160:
            return "medium_low"
        if dose <= 240:
            return "medium_high"
        return "high_dose"
    if dose <= 0:
        return "dose_0"
    if dose < 150:
        return "dose_75"
    if drug == "lh":
        return "dose_150_plus"
    if dose < 225:
        return "dose_150"
    return "dose_225_plus"


def _similarity_from_distance(distance: Any) -> float:
    distance_value = max(_safe_float(distance, 0.0), 0.0)
    return round(1.0 / (1.0 + distance_value), 3)


def _frame_from_context(context_or_frame: Any) -> pd.DataFrame:
    if isinstance(context_or_frame, pd.DataFrame):
        return context_or_frame.copy()
    if isinstance(context_or_frame, Mapping):
        frame = context_or_frame.get("similar_cases")
        if isinstance(frame, pd.DataFrame):
            return frame.copy()
    return pd.DataFrame()


@lru_cache(maxsize=1)
def _successful_cycle_label_lookup() -> dict[str, dict[str, Any]]:
    path = REPO_ROOT / "data" / "processed" / "successful_cycle_labels_v1.csv"
    columns = [
        "cycle_uid",
        "follicles_ge14",
        "exact_evaluable",
        "kong_aligned_exact_success",
        "complication_text",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    return {str(row["cycle_uid"]): row.to_dict() for _, row in frame.iterrows()}


def _strict_modsev_ohss(value: Any) -> bool | None:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return any(token in text for token in ("moderate", "severe", "中度", "重度", "中重度"))


def build_knn_reference_rows(context_or_frame: Any, limit: int = 4) -> list[dict[str, Any]]:
    """Build anonymized KNN support rows with audited Kong exact outcomes."""
    frame = _frame_from_context(context_or_frame)
    if frame.empty:
        return []
    if "distance" in frame.columns:
        frame = frame.sort_values("distance", ascending=True)
    label_lookup = _successful_cycle_label_lookup()
    prepared: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        current_fsh = _safe_float(row.get("current_fsh_daily_dose"), 0.0)
        current_lh = _safe_float(row.get("current_lh_daily_dose"), 0.0)
        current_hmg = _safe_float(row.get("current_hmg_daily_dose"), 0.0)
        next_fsh = _safe_float(row.get("next_fsh_daily_dose"), current_fsh)
        next_lh = _safe_float(row.get("next_lh_daily_dose"), current_lh)
        next_hmg = _safe_float(row.get("next_hmg_daily_dose"), current_hmg)
        oocytes = row.get("target_oocytes")
        mii = row.get("target_mii")
        label_row = label_lookup.get(str(row.get("cycle_uid")), {})
        exact_evaluable = bool(_safe_int(label_row.get("exact_evaluable"), 0))
        exact_value = label_row.get("kong_aligned_exact_success")
        success = bool(_safe_int(exact_value, 0)) if exact_evaluable and not pd.isna(exact_value) else None
        follicles_ge14 = label_row.get("follicles_ge14")
        modsev = _strict_modsev_ohss(label_row.get("complication_text"))
        day = _fmt_num(row.get("gn_day", row.get("Day")))
        visit = _fmt_num(row.get("monitoring_order", row.get("visits_seen")))
        categories = (_dose_category("fsh", next_fsh), _dose_category("lh", next_lh), _dose_category("hmg", next_hmg))
        trigger_label = "达标" if success is True else "未达标" if success is False else "不可评估"
        trigger_class = "ct" if success is True else "cw" if success is False else "cm"
        ohss_label = "发生" if modsev is True else "未发生" if modsev is False else "--"
        ohss_class = "cd" if modsev is True else "ct" if modsev is False else "cm"
        prepared.append(
            {
                "categories": categories,
                "success_bool": success,
                "case_payload": {
                    "similarity_score": _similarity_from_distance(row.get("distance", 0.0)),
                    "similarity": f"{_similarity_from_distance(row.get('distance', 0.0)) * 100:.0f}%",
                    "distance": _safe_float(row.get("distance"), 0.0),
                    "day_visit": f"Day {day} / Visit {visit}",
                    "fsh_category": categories[0],
                    "lh_category": categories[1],
                    "hmg_category": categories[2],
                    "fsh_dose": _fmt_num(next_fsh),
                    "lh_dose": _fmt_num(next_lh),
                    "hmg_dose": _fmt_num(next_hmg),
                    "oocytes": _fmt_num(oocytes),
                    "mii": _fmt_num(mii),
                    "follicles_ge14": _fmt_num(follicles_ge14),
                    "observed_outcome": f"获卵 {_fmt_num(oocytes)}",
                    "trigger_success": trigger_label,
                    "trigger_class": trigger_class,
                    "ohss_label": ohss_label,
                    "ohss_class": ohss_class,
                    "action": str(row.get("combined_gn_action", row.get("gn_action", "--"))),
                    "trajectory": (
                        f"Day {day}: current {_fmt_num(current_fsh)}/{_fmt_num(current_lh)}/{_fmt_num(current_hmg)} "
                        f"-> {_fmt_num(next_fsh)}/{_fmt_num(next_lh)}/{_fmt_num(next_hmg)}"
                    ),
                },
            }
        )
    total = max(len(prepared), 1)
    selection_counts: dict[tuple[str, str, str], int] = {}
    success_counts: dict[tuple[str, str, str], int] = {}
    success_denominators: dict[tuple[str, str, str], int] = {}
    for item in prepared:
        key = item["categories"]
        selection_counts[key] = selection_counts.get(key, 0) + 1
        if item["success_bool"] is not None:
            success_denominators[key] = success_denominators.get(key, 0) + 1
            success_counts[key] = success_counts.get(key, 0) + int(item["success_bool"])
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(prepared[: max(1, int(limit))], start=1):
        key = item["categories"]
        selection_rate_value = selection_counts.get(key, 0) / total
        denominator = success_denominators.get(key, 0)
        success_rate_value = success_counts.get(key, 0) / denominator if denominator else float("nan")
        payload = dict(item["case_payload"])
        payload.update(
            {
                "case": f"Case {idx}",
                "selection_rate_value": round(selection_rate_value, 4),
                "success_rate_value": round(success_rate_value, 4) if not pd.isna(success_rate_value) else None,
                "selection_rate": f"{selection_rate_value * 100:.0f}%",
                "success_rate": f"{success_rate_value * 100:.0f}%" if not pd.isna(success_rate_value) else "--",
                "success_class": "ct" if not pd.isna(success_rate_value) and success_rate_value >= 0.5 else "cw" if not pd.isna(success_rate_value) else "cm",
            }
        )
        rows.append(payload)
    return rows


def _drug_dose_group(drug: str, dose: float) -> tuple[str, str]:
    value = max(0.0, _safe_float(dose, 0.0))
    if drug == "fsh":
        if value < 80:
            return "low_or_none", "0-80"
        if value <= 160:
            return "moderate", "80-160"
        return "high", ">160"
    if drug == "lh":
        if value <= 0:
            return "dose_0", "0"
        if value < 150:
            return "dose_75", "75"
        return "dose_ge150", "≥150"
    if value <= 0:
        return "dose_0", "0"
    if value < 150:
        return "dose_75", "75"
    if value < 225:
        return "dose_150", "150"
    return "dose_ge225", "≥225"


def build_knn_drug_summary(context_or_frame: Any, limit: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Aggregate similar cases into drug-specific dose evidence ranked by selection and success."""
    frame = _frame_from_context(context_or_frame)
    empty = {drug: [] for drug in ("fsh", "lh", "hmg")}
    if frame.empty:
        return empty
    if "distance" in frame.columns:
        frame = frame.sort_values("distance", ascending=True)
    label_lookup = _successful_cycle_label_lookup()
    summaries: dict[str, list[dict[str, Any]]] = {}
    for drug in ("fsh", "lh", "hmg"):
        dose_universe = {
            "fsh": [("low_or_none", "0-80"), ("moderate", "80-160"), ("high", ">160")],
            "lh": [("dose_0", "0"), ("dose_75", "75"), ("dose_ge150", "≥150")],
            "hmg": [("dose_0", "0"), ("dose_75", "75"), ("dose_150", "150"), ("dose_ge225", "≥225")],
        }[drug]
        groups: dict[str, dict[str, Any]] = {
            key: {"dose": display, "case_count": 0, "success_count": 0, "success_denominator": 0}
            for key, display in dose_universe
        }
        for _, row in frame.iterrows():
            current = _safe_float(row.get(f"current_{drug}_daily_dose"), 0.0)
            dose = _safe_float(row.get(f"next_{drug}_daily_dose"), current)
            key, display = _drug_dose_group(drug, dose)
            group = groups.setdefault(
                key,
                {"dose": display, "case_count": 0, "success_count": 0, "success_denominator": 0},
            )
            group["case_count"] += 1
            label_row = label_lookup.get(str(row.get("cycle_uid")), {})
            exact_evaluable = bool(_safe_int(label_row.get("exact_evaluable"), 0))
            exact_value = label_row.get("kong_aligned_exact_success")
            if exact_evaluable and not pd.isna(exact_value):
                group["success_denominator"] += 1
                group["success_count"] += int(bool(_safe_int(exact_value, 0)))
        total = max(len(frame), 1)
        ranked: list[dict[str, Any]] = []
        for group in groups.values():
            selection_rate = group["case_count"] / total
            denominator = group["success_denominator"]
            success_rate = group["success_count"] / denominator if denominator else None
            composite = (selection_rate + success_rate) / 2.0 if success_rate is not None else -1.0
            ranked.append(
                {
                    "dose": group["dose"],
                    "case_count": int(group["case_count"]),
                    "selection_rate_value": round(selection_rate, 4),
                    "success_rate_value": round(success_rate, 4) if success_rate is not None else None,
                    "selection_rate": f"{selection_rate * 100:.0f}%",
                    "success_rate": f"{success_rate * 100:.0f}%" if success_rate is not None else "--",
                    "composite_score": round(composite, 4),
                }
            )
        ranked.sort(
            key=lambda item: (
                -float(item["composite_score"]),
                -float(item["success_rate_value"] if item["success_rate_value"] is not None else -1.0),
                -float(item["selection_rate_value"]),
                -int(item["case_count"]),
            )
        )
        rows = ranked[: max(1, int(limit))]
        for index, row in enumerate(rows, start=1):
            row["case"] = f"Case {index}"
        summaries[drug] = rows
    return summaries
