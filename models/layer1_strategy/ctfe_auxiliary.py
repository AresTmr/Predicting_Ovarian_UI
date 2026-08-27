from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CTFE_DOSE_LABELS = ["stop", "decrease", "low", "medium", "high"]
CTFE_DOSE_TO_ID = {label: idx for idx, label in enumerate(CTFE_DOSE_LABELS)}
CTFE_ID_TO_DOSE = {idx: label for label, idx in CTFE_DOSE_TO_ID.items()}
DEFAULT_MAX_VISITS = 6
DEFAULT_SEED = 20260424
CURRENT_CTFE_POINTER = Path("models/artifacts/current_layer1_ctfe_auxiliary_run.txt")

CTFE_STATIC_FEATURES = [
    "age",
    "bmi",
    "infertility_duration",
    "amh",
    "afc",
    "initial_gn_dose",
    "basal_fsh",
    "basal_lh",
    "basal_e2",
    "basal_p",
    "male_age",
    "male_factor_infertility_flag",
]

CTFE_DYNAMIC_FEATURES = [
    "gn_day",
    "cycle_day",
    "current_e2",
    "current_lh",
    "current_p",
    "current_fsh",
    "current_endometrium",
    "current_fsh_daily_dose",
    "current_lh_daily_dose",
    "current_hmg_daily_dose",
    "current_lh_like_hmg_daily_dose",
    "current_gn_dose",
    "cumulative_fsh_dose",
    "cumulative_lh_dose",
    "cumulative_hmg_dose",
    "cumulative_lh_like_hmg_dose",
    "cumulative_gn_dose",
    "delta_e2",
    "delta_lh",
    "delta_p",
    "delta_fsh",
    "delta_endometrium",
    "days_since_previous_visit",
    "visits_seen",
    "total_follicle_count",
    "mature_follicle_count",
    "max_follicle_diameter",
    "mean_follicle_diameter",
    "follicle_count_lt_10",
    "follicle_count_10_12",
    "follicle_count_13_15",
    "follicle_count_16_18",
    "follicle_count_gt_18",
    "growing_follicle_count",
    "medium_plus_follicle_count",
    "follicle_maturity_index",
    "mature_follicle_share",
    "large_follicle_share",
    "current_e2_per_gn",
    "gn_per_total_follicle",
    "gn_per_mature_follicle",
    "previous_fsh_daily_dose",
    "previous_lh_daily_dose",
    "previous_hmg_daily_dose",
    "previous_lh_like_hmg_daily_dose",
    "previous_gn_dose",
]

FORBIDDEN_FEATURE_FRAGMENTS = (
    "next_",
    "target_",
    "observed_",
    "embryo",
    "transfer",
    "clinical_pregnancy",
    "live_birth",
    "oocytes",
    "mii",
    "ohss_flag",
)


@dataclass
class CTFESequenceDataset:
    X: np.ndarray
    y: np.ndarray
    row_index: pd.DataFrame
    feature_names: list[str]
    sequence_lengths: np.ndarray
    split: np.ndarray
    static_features: list[str]
    dynamic_features: list[str]
    max_visits: int


def classify_fsh_dose_class(dose: float | int | None) -> str | pd.NA:
    if pd.isna(dose):
        return pd.NA
    value = float(dose)
    if value <= 0:
        return "stop"
    if value <= 80:
        return "decrease"
    if value <= 160:
        return "low"
    if value <= 240:
        return "medium"
    return "high"


def create_ctfe_fsh_labels(frame: pd.DataFrame, next_fsh_col: str = "next_fsh_daily_dose") -> pd.DataFrame:
    if next_fsh_col not in frame.columns:
        raise ValueError(f"Missing required CTFE label column: {next_fsh_col}")
    df = frame.copy()
    df["ctfe_next_fsh_dose_class"] = df[next_fsh_col].map(classify_fsh_dose_class)
    df["ctfe_next_fsh_dose_class_id"] = df["ctfe_next_fsh_dose_class"].map(CTFE_DOSE_TO_ID)
    return df


def _numeric_columns_available(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])]


def _safe_numeric(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return np.nan
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if not pd.isna(value) else np.nan


def _load_split(frame: pd.DataFrame, split_manifest_path: str | Path | None) -> pd.Series:
    if split_manifest_path is None:
        return pd.Series(["train"] * len(frame), index=frame.index)
    path = Path(split_manifest_path)
    if not path.exists() or "cycle_uid" not in frame.columns:
        return pd.Series(["train"] * len(frame), index=frame.index)
    manifest = pd.read_csv(path)
    if not {"cycle_uid", "split"}.issubset(manifest.columns):
        return pd.Series(["train"] * len(frame), index=frame.index)
    split_map = manifest.drop_duplicates("cycle_uid").set_index("cycle_uid")["split"]
    return frame["cycle_uid"].map(split_map).fillna("train")


def _assert_no_forbidden_features(feature_names: Sequence[str]) -> None:
    bad = [name for name in feature_names if any(fragment in name for fragment in FORBIDDEN_FEATURE_FRAGMENTS)]
    if bad:
        raise ValueError(f"CTFE feature list contains leakage-prone columns: {bad[:20]}")


def build_ctfe_sequence_dataset(
    frame: pd.DataFrame,
    *,
    split_manifest_path: str | Path | None = None,
    max_visits: int = DEFAULT_MAX_VISITS,
    eligible_only: bool = True,
) -> CTFESequenceDataset:
    df = create_ctfe_fsh_labels(frame)
    if eligible_only and "has_next_visit" in df.columns:
        df = df[df["has_next_visit"].astype(bool)].copy()
    if eligible_only and "strategy_eligible_flag" in df.columns:
        df = df[df["strategy_eligible_flag"].astype(bool)].copy()
    df = df[df["ctfe_next_fsh_dose_class"].isin(CTFE_DOSE_LABELS)].copy()
    if df.empty:
        raise ValueError("No CTFE-eligible rows after label creation and strategy filters.")
    if "cycle_uid" not in df.columns or "monitoring_order" not in df.columns:
        raise ValueError("CTFE sequence building requires cycle_uid and monitoring_order.")

    df["split"] = _load_split(df, split_manifest_path).values
    df = df.sort_values(["cycle_uid", "monitoring_order", "visit_uid" if "visit_uid" in df.columns else "monitoring_order"]).reset_index(drop=True)
    static_features = _numeric_columns_available(df, CTFE_STATIC_FEATURES)
    dynamic_features = _numeric_columns_available(df, CTFE_DYNAMIC_FEATURES)
    if not static_features:
        raise ValueError("No numeric static features available for CTFE.")
    if not dynamic_features:
        raise ValueError("No numeric dynamic features available for CTFE.")

    feature_names: list[str] = []
    feature_names.extend([f"static::{name}" for name in static_features])
    for visit_offset in range(max_visits):
        for name in dynamic_features:
            feature_names.append(f"visit{visit_offset + 1}::{name}")
            feature_names.append(f"visit{visit_offset + 1}::present_mask")
    for name in dynamic_features:
        feature_names.append(f"last::{name}")
    for name in ["current_e2", "total_follicle_count", "current_fsh_daily_dose", "current_gn_dose"]:
        if name in dynamic_features:
            feature_names.append(f"slope::{name}")
    feature_names.append("sequence_length")
    _assert_no_forbidden_features(feature_names)

    rows: list[np.ndarray] = []
    y: list[int] = []
    row_records: list[dict[str, Any]] = []
    lengths: list[int] = []

    grouped = {cycle: group.sort_values("monitoring_order").reset_index(drop=True) for cycle, group in df.groupby("cycle_uid", sort=False)}
    for _, query in df.iterrows():
        cycle = query["cycle_uid"]
        current_order = query["monitoring_order"]
        history = grouped[cycle][grouped[cycle]["monitoring_order"] <= current_order].tail(max_visits)
        seq_len = int(len(history))
        vector: list[float] = []
        vector.extend([_safe_numeric(query, name) for name in static_features])
        pad_count = max_visits - seq_len
        for _ in range(pad_count):
            vector.extend([np.nan] * len(dynamic_features))
            vector.append(0.0)
        for _, hist_row in history.iterrows():
            vector.extend([_safe_numeric(hist_row, name) for name in dynamic_features])
            vector.append(1.0)
        last_row = history.iloc[-1]
        vector.extend([_safe_numeric(last_row, name) for name in dynamic_features])
        first_row = history.iloc[0]
        denom = max(seq_len - 1, 1)
        for name in ["current_e2", "total_follicle_count", "current_fsh_daily_dose", "current_gn_dose"]:
            if name in dynamic_features:
                vector.append((_safe_numeric(last_row, name) - _safe_numeric(first_row, name)) / denom)
        vector.append(float(seq_len))
        rows.append(np.asarray(vector, dtype=float))
        y.append(int(query["ctfe_next_fsh_dose_class_id"]))
        row_records.append(
            {
                "visit_uid": query.get("visit_uid", ""),
                "cycle_uid": query.get("cycle_uid", ""),
                "art_id": query.get("art_id", ""),
                "monitoring_order": query.get("monitoring_order", np.nan),
                "split": query.get("split", "train"),
                "ctfe_next_fsh_dose_class": query.get("ctfe_next_fsh_dose_class"),
                "next_fsh_daily_dose": query.get("next_fsh_daily_dose", np.nan),
            }
        )
        lengths.append(seq_len)

    return CTFESequenceDataset(
        X=np.vstack(rows),
        y=np.asarray(y, dtype=int),
        row_index=pd.DataFrame(row_records),
        feature_names=feature_names,
        sequence_lengths=np.asarray(lengths, dtype=int),
        split=np.asarray(df["split"].tolist(), dtype=object),
        static_features=static_features,
        dynamic_features=dynamic_features,
        max_visits=int(max_visits),
    )


def fit_ctfe_encoder(X_train: np.ndarray, *, n_components: int = 24, seed: int = DEFAULT_SEED) -> Pipeline:
    n_components = int(max(2, min(n_components, X_train.shape[0] - 1, X_train.shape[1])))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
        ]
    ).fit(X_train)


def _build_classifier(seed: int):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=len(CTFE_DOSE_LABELS),
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, random_state=seed)


def _class_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CTFE_DOSE_LABELS)).astype(float)
    total = counts.sum()
    weights = np.ones_like(y, dtype=float)
    for class_id, count in enumerate(counts):
        if count > 0:
            weights[y == class_id] = total / (len(CTFE_DOSE_LABELS) * count)
    return weights


def _predict_proba_full(estimator: Any, X: np.ndarray) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(X), dtype=float)
    if raw.shape[1] == len(CTFE_DOSE_LABELS):
        return raw
    full = np.zeros((raw.shape[0], len(CTFE_DOSE_LABELS)), dtype=float)
    classes = getattr(estimator, "classes_", np.arange(raw.shape[1]))
    for idx, class_id in enumerate(classes):
        if int(class_id) < len(CTFE_DOSE_LABELS):
            full[:, int(class_id)] = raw[:, idx]
    row_sum = full.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return full / row_sum


def evaluate_ctfe_classifier(y_true: np.ndarray, proba: np.ndarray, split: str) -> dict[str, Any]:
    pred = proba.argmax(axis=1)
    metrics: dict[str, Any] = {
        "split": split,
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=list(range(len(CTFE_DOSE_LABELS))))),
    }
    report = classification_report(
        y_true,
        pred,
        labels=list(range(len(CTFE_DOSE_LABELS))),
        target_names=CTFE_DOSE_LABELS,
        output_dict=True,
        zero_division=0,
    )
    for label in CTFE_DOSE_LABELS:
        payload = report.get(label, {})
        metrics[f"precision_{label}"] = float(payload.get("precision", 0.0))
        metrics[f"recall_{label}"] = float(payload.get("recall", 0.0))
        metrics[f"f1_{label}"] = float(payload.get("f1-score", 0.0))
        metrics[f"support_{label}"] = int(payload.get("support", 0))
    return metrics


def make_prediction_frame(dataset: CTFESequenceDataset, proba: np.ndarray, split_mask: np.ndarray | None = None) -> pd.DataFrame:
    idx = dataset.row_index.copy()
    if split_mask is not None:
        idx = idx.loc[split_mask].reset_index(drop=True)
    pred_ids = proba.argmax(axis=1)
    idx["ctfe_prediction"] = [CTFE_ID_TO_DOSE[int(value)] for value in pred_ids]
    for class_id, label in enumerate(CTFE_DOSE_LABELS):
        idx[f"prob_{label}"] = proba[:, class_id]
    return idx


def fit_embedding_knn(embeddings: np.ndarray, row_index: pd.DataFrame, k: int = 50) -> dict[str, Any]:
    k_eff = int(max(1, min(k, len(row_index))))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(embeddings)
    return {"nearest_neighbors": nn, "embeddings": embeddings, "row_index": row_index.reset_index(drop=True), "k": k_eff}


def get_ctfe_similar_cases(retriever: Mapping[str, Any], query_embedding: np.ndarray, *, exclude_patient_id: Any = None, exclude_cycle_id: Any = None, k: int = 50) -> pd.DataFrame:
    nn = retriever["nearest_neighbors"]
    distances, indices = nn.kneighbors(query_embedding.reshape(1, -1), n_neighbors=min(int(k) + 10, len(retriever["row_index"])))
    rows = retriever["row_index"].iloc[indices[0]].copy()
    rows.insert(0, "distance", distances[0])
    if exclude_patient_id is not None and "art_id" in rows.columns:
        rows = rows[rows["art_id"].astype(str) != str(exclude_patient_id)]
    if exclude_cycle_id is not None and "cycle_uid" in rows.columns:
        rows = rows[rows["cycle_uid"].astype(str) != str(exclude_cycle_id)]
    return rows.head(k).reset_index(drop=True)


def train_ctfe_auxiliary_model(
    *,
    input_path: str | Path = "data/processed/layer1_strategy_dataset.csv",
    split_manifest_path: str | Path = "data/splits/split_manifest_v1.csv",
    output_root: str | Path = "models/artifacts",
    run_id: str | None = None,
    max_visits: int = DEFAULT_MAX_VISITS,
    seed: int = DEFAULT_SEED,
    embedding_components: int = 24,
) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    dataset = build_ctfe_sequence_dataset(frame, split_manifest_path=split_manifest_path, max_visits=max_visits)
    train_mask = dataset.split == "train"
    valid_mask = dataset.split == "valid"
    test_mask = dataset.split == "test"
    if train_mask.sum() < 10:
        raise ValueError("CTFE training requires at least 10 train rows.")
    if valid_mask.sum() == 0:
        valid_mask = train_mask
    if test_mask.sum() == 0:
        test_mask = valid_mask

    encoder = fit_ctfe_encoder(dataset.X[train_mask], n_components=embedding_components, seed=seed)
    Z = encoder.transform(dataset.X)
    estimator = _build_classifier(seed)
    sample_weight = _class_sample_weight(dataset.y[train_mask])
    fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
    try:
        from lightgbm import log_evaluation

        # Kong-style dose-class prediction is used as an auxiliary reference.
        # In this dataset, validation weighted-F1 was better without class reweighting/early stopping.
        estimator.fit(
            Z[train_mask],
            dataset.y[train_mask],
            eval_set=[(Z[train_mask], dataset.y[train_mask]), (Z[valid_mask], dataset.y[valid_mask])],
            eval_names=["train", "valid"],
            eval_metric="multi_logloss",
            callbacks=[log_evaluation(0)],
        )
    except TypeError:
        estimator.fit(Z[train_mask], dataset.y[train_mask], **fit_kwargs)
    except Exception:
        estimator.fit(Z[train_mask], dataset.y[train_mask], **fit_kwargs)

    proba_all = _predict_proba_full(estimator, Z)
    metrics = [
        evaluate_ctfe_classifier(dataset.y[train_mask], proba_all[train_mask], "train"),
        evaluate_ctfe_classifier(dataset.y[valid_mask], proba_all[valid_mask], "valid"),
        evaluate_ctfe_classifier(dataset.y[test_mask], proba_all[test_mask], "test"),
    ]
    run_id = run_id or f"phase8_layer1_ctfe_auxiliary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_root) / run_id / "layer1_ctfe_auxiliary"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics).to_csv(output_dir / "ctfe_dose_class_metrics.csv", index=False)
    evals_result = getattr(estimator, "evals_result_", None)
    if isinstance(evals_result, Mapping):
        curve_rows: list[dict[str, Any]] = []
        for split_name, metric_payload in evals_result.items():
            normalized_split = "train" if split_name in {"train", "valid_0"} else "valid" if split_name in {"valid", "valid_1"} else str(split_name)
            for metric_name, values in metric_payload.items():
                for step, value in enumerate(values):
                    curve_rows.append({"split": normalized_split, "metric": metric_name, "step": step, "value": float(value)})
        if curve_rows:
            pd.DataFrame(curve_rows).to_csv(output_dir / "ctfe_training_curve.csv", index=False)
    make_prediction_frame(dataset, proba_all).to_csv(output_dir / "ctfe_dose_class_predictions.csv", index=False)
    pd.DataFrame(
        {
            "dose_class": CTFE_DOSE_LABELS,
            "train_count": np.bincount(dataset.y[train_mask], minlength=len(CTFE_DOSE_LABELS)),
            "valid_count": np.bincount(dataset.y[valid_mask], minlength=len(CTFE_DOSE_LABELS)),
            "test_count": np.bincount(dataset.y[test_mask], minlength=len(CTFE_DOSE_LABELS)),
        }
    ).to_csv(output_dir / "ctfe_class_distribution.csv", index=False)
    cm = confusion_matrix(dataset.y[test_mask], proba_all[test_mask].argmax(axis=1), labels=list(range(len(CTFE_DOSE_LABELS))))
    pd.DataFrame(cm, index=CTFE_DOSE_LABELS, columns=CTFE_DOSE_LABELS).to_csv(output_dir / "ctfe_confusion_matrix.csv")

    train_index = dataset.row_index.loc[train_mask].copy().reset_index(drop=True)
    for class_id, label in enumerate(CTFE_DOSE_LABELS):
        train_index[f"prob_{label}"] = proba_all[train_mask, class_id]
    train_index["ctfe_prediction"] = [CTFE_ID_TO_DOSE[int(value)] for value in proba_all[train_mask].argmax(axis=1)]
    retriever = fit_embedding_knn(Z[train_mask], train_index, k=50)
    joblib.dump(
        {
            "encoder": encoder,
            "estimator": estimator,
            "feature_names": dataset.feature_names,
            "static_features": dataset.static_features,
            "dynamic_features": dataset.dynamic_features,
            "max_visits": dataset.max_visits,
            "dose_labels": CTFE_DOSE_LABELS,
            "run_id": run_id,
            "retriever": retriever,
        },
        output_dir / "ctfe_auxiliary_bundle.joblib",
    )

    summary = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "max_visits": max_visits,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "best_metric_name": "valid_weighted_f1",
        "valid_weighted_f1": float(pd.DataFrame(metrics).query("split == 'valid'")["weighted_f1"].iloc[0]),
        "test_weighted_f1": float(pd.DataFrame(metrics).query("split == 'test'")["weighted_f1"].iloc[0]),
        "model_note": "CTFE-inspired auxiliary branch: padded current/prior visit sequence + cross-temporal summary encoder + LightGBM/GBDT classifier. It does not replace Layer1 FSH/LH/HMG action bundles.",
    }
    (output_dir / "ctfe_auxiliary_summary.json").write_text(pd.Series(summary).to_json(force_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "ctfe_auxiliary_summary.md").write_text(
        "\n".join(
            [
                "# Layer1 Kong CTFE Auxiliary Model",
                "",
                "This run is an auxiliary reference branch and does not replace the current Layer1 action recommendation model.",
                "",
                f"- run_id: `{run_id}`",
                f"- max_visits: `{max_visits}`",
                f"- train/valid/test rows: `{summary['train_rows']}/{summary['valid_rows']}/{summary['test_rows']}`",
                f"- valid weighted-F1: `{summary['valid_weighted_f1']:.4f}`",
                f"- test weighted-F1: `{summary['test_weighted_f1']:.4f}`",
                "",
                "Dose classes follow Kong CTFE bins: stop=0, decrease=(0,80], low=(80,160], medium=(160,240], high=(240, inf).",
            ]
        ),
        encoding="utf-8",
    )
    CURRENT_CTFE_POINTER.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_CTFE_POINTER.write_text(run_id, encoding="utf-8")
    return summary
