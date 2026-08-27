from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.layer1_strategy.action_model import ACTION_LABELS

DEFAULT_K_SIMILAR = 50
DEFAULT_STAGE_DAY_TOLERANCE = 1
DEFAULT_KNN_FEATURES = [
    "age",
    "bmi",
    "amh",
    "afc",
    "basal_fsh",
    "basal_lh",
    "basal_e2",
    "infertility_duration",
    "diagnosis_primary_secondary",
    "male_factor_infertility_flag",
    "current_e2",
    "current_lh",
    "current_fsh",
    "current_p",
    "current_endometrium",
    "total_follicle_count",
    "mature_follicle_count",
    "follicle_count_lt_10",
    "follicle_count_10_12",
    "follicle_count_13_15",
    "follicle_count_16_18",
    "follicle_count_gt_18",
    "growing_follicle_count",
    "medium_plus_follicle_count",
    "max_follicle_diameter",
    "mean_follicle_diameter",
    "follicle_maturity_index",
    "delta_e2",
    "delta_follicle_count",
    "current_fsh_daily_dose",
    "current_lh_like_hmg_daily_dose",
    "cumulative_gn_dose",
    "gn_day",
    "previous_gn_dose",
]
KNN_FORBIDDEN_COLUMNS = {
    "next_gn_dose",
    "next_action",
    "next_monitoring_order",
    "next_monitoring_date",
    "next_visit_interval_days",
    "observed_gn_dose_delta",
    "observed_action_label",
    "gn_action",
    "fsh_action",
    "lh_like_action",
    "combined_gn_action",
    "target_oocytes",
    "target_mii",
    "target_ohss_flag",
}
KNN_FORBIDDEN_PREFIXES = ("next_", "target_", "observed_", "embryo", "transfer")


@dataclass(frozen=True)
class KnnRetriever:
    source_frame: pd.DataFrame
    feature_columns: list[str]
    preprocessor: ColumnTransformer
    estimator: NearestNeighbors
    matrix: np.ndarray
    metric: str


def filter_history_by_monitoring_stage(
    history_frame: pd.DataFrame,
    query: pd.Series | Mapping[str, Any],
    day_tolerance: int = DEFAULT_STAGE_DAY_TOLERANCE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep historical snapshots from the same visit stage as the query.

    The primary pool requires the same monitoring order and a stimulation day
    within the configured tolerance. If that pool is empty, only the day
    window is relaxed; the monitoring order is never relaxed.
    """
    if history_frame.empty:
        return history_frame.copy(), {
            "match_mode": "no_history",
            "monitoring_order": None,
            "stimulation_day": None,
            "day_tolerance": int(day_tolerance),
            "candidate_count": 0,
        }

    query_series = pd.Series(query)
    order = pd.to_numeric(pd.Series([query_series.get("monitoring_order")]), errors="coerce").iloc[0]
    day = pd.to_numeric(pd.Series([query_series.get("gn_day")]), errors="coerce").iloc[0]
    if pd.isna(order) or "monitoring_order" not in history_frame.columns:
        return history_frame.iloc[0:0].copy(), {
            "match_mode": "missing_monitoring_order",
            "monitoring_order": None if pd.isna(order) else int(order),
            "stimulation_day": None if pd.isna(day) else float(day),
            "day_tolerance": int(day_tolerance),
            "candidate_count": 0,
        }

    order_values = pd.to_numeric(history_frame["monitoring_order"], errors="coerce")
    stage_frame = history_frame.loc[order_values == order].copy()
    if stage_frame.empty:
        return stage_frame, {
            "match_mode": "no_same_monitoring_order",
            "monitoring_order": int(order),
            "stimulation_day": None if pd.isna(day) else float(day),
            "day_tolerance": int(day_tolerance),
            "candidate_count": 0,
        }

    if pd.isna(day) or "gn_day" not in stage_frame.columns:
        return stage_frame, {
            "match_mode": "same_monitoring_order_day_unavailable",
            "monitoring_order": int(order),
            "stimulation_day": None if pd.isna(day) else float(day),
            "day_tolerance": int(day_tolerance),
            "candidate_count": int(len(stage_frame)),
        }

    stage_days = pd.to_numeric(stage_frame["gn_day"], errors="coerce")
    within_window = (stage_days - float(day)).abs() <= max(int(day_tolerance), 0)
    day_frame = stage_frame.loc[within_window].copy()
    if not day_frame.empty:
        return day_frame, {
            "match_mode": "same_monitoring_order_and_day_window",
            "monitoring_order": int(order),
            "stimulation_day": float(day),
            "day_tolerance": int(day_tolerance),
            "candidate_count": int(len(day_frame)),
            "same_order_count": int(len(stage_frame)),
        }

    return stage_frame, {
        "match_mode": "same_monitoring_order_day_window_fallback",
        "monitoring_order": int(order),
        "stimulation_day": float(day),
        "day_tolerance": int(day_tolerance),
        "candidate_count": int(len(stage_frame)),
        "same_order_count": int(len(stage_frame)),
    }


def _allowed_knn_features(frame: pd.DataFrame, requested: Sequence[str] | None = None) -> list[str]:
    candidates = list(requested or DEFAULT_KNN_FEATURES)
    selected: list[str] = []
    for column in candidates:
        if column not in frame.columns:
            continue
        if column in KNN_FORBIDDEN_COLUMNS:
            continue
        lowered = column.lower()
        if any(lowered.startswith(prefix) for prefix in KNN_FORBIDDEN_PREFIXES):
            continue
        if "embryo" in lowered or "transfer" in lowered:
            continue
        selected.append(column)
    return selected


def _build_knn_preprocessor(frame: pd.DataFrame, features: Sequence[str]) -> ColumnTransformer:
    numeric = [column for column in features if is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def create_knn_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    preprocessor: ColumnTransformer | None = None,
) -> tuple[np.ndarray, list[str], ColumnTransformer]:
    features = _allowed_knn_features(frame, feature_columns)
    if not features:
        raise ValueError("No valid non-leakage KNN features are available")
    if preprocessor is None:
        preprocessor = _build_knn_preprocessor(frame, features)
        matrix = preprocessor.fit_transform(frame[features])
    else:
        matrix = preprocessor.transform(frame[features])
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=float)
    return matrix, features, preprocessor


def fit_knn_retriever(
    history_frame: pd.DataFrame,
    k: int = DEFAULT_K_SIMILAR,
    metric: str = "euclidean",
    feature_columns: Sequence[str] | None = None,
) -> KnnRetriever:
    if history_frame.empty:
        raise ValueError("KNN history frame is empty")
    matrix, features, preprocessor = create_knn_feature_matrix(history_frame, feature_columns=feature_columns)
    n_neighbors = min(len(history_frame), max(1, int(k)))
    estimator = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    estimator.fit(matrix)
    return KnnRetriever(
        source_frame=history_frame.reset_index(drop=True).copy(),
        feature_columns=features,
        preprocessor=preprocessor,
        estimator=estimator,
        matrix=matrix,
        metric=metric,
    )


def _query_matrix(retriever: KnnRetriever, query: pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(query, pd.Series):
        query_frame = query.to_frame().T
    else:
        query_frame = query.copy()
    for column in retriever.feature_columns:
        if column not in query_frame.columns:
            query_frame[column] = np.nan
    transformed = retriever.preprocessor.transform(query_frame[retriever.feature_columns])
    return transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed, dtype=float)


def get_similar_cases(
    retriever: KnnRetriever,
    query: pd.Series | pd.DataFrame,
    k: int = DEFAULT_K_SIMILAR,
    exclude_patient_id: Any | None = None,
    exclude_cycle_id: Any | None = None,
) -> pd.DataFrame:
    if isinstance(query, pd.DataFrame):
        if len(query) != 1:
            raise ValueError("get_similar_cases expects one query row")
        query_series = query.iloc[0]
    else:
        query_series = query
    exclude_patient_id = query_series.get("art_id", exclude_patient_id) if exclude_patient_id is None else exclude_patient_id
    exclude_cycle_id = query_series.get("cycle_uid", exclude_cycle_id) if exclude_cycle_id is None else exclude_cycle_id

    q = _query_matrix(retriever, query_series)
    n_neighbors = min(len(retriever.source_frame), max(int(k) * 5, int(k) + 20, 1))
    if n_neighbors < len(retriever.source_frame):
        distances, indices = retriever.estimator.kneighbors(q, n_neighbors=n_neighbors)
    else:
        estimator = NearestNeighbors(n_neighbors=len(retriever.source_frame), metric=retriever.metric)
        estimator.fit(retriever.matrix)
        distances, indices = estimator.kneighbors(q, n_neighbors=len(retriever.source_frame))

    rows = retriever.source_frame.iloc[indices[0]].copy().reset_index(drop=True)
    rows.insert(0, "distance", distances[0])
    if "art_id" in rows.columns and exclude_patient_id is not None:
        rows = rows[rows["art_id"] != exclude_patient_id]
    if "cycle_uid" in rows.columns and exclude_cycle_id is not None:
        rows = rows[rows["cycle_uid"] != exclude_cycle_id]
    return rows.sort_values("distance", ascending=True).head(int(k)).reset_index(drop=True)


def compute_selection_rate(
    similar_cases: pd.DataFrame,
    action_col: str = "combined_gn_action",
    labels: Sequence[str] = ACTION_LABELS,
) -> pd.DataFrame:
    denominator = max(len(similar_cases), 1)
    counts = similar_cases[action_col].value_counts().reindex(list(labels), fill_value=0) if action_col in similar_cases else pd.Series(0, index=list(labels))
    return pd.DataFrame(
        {
            "action": list(labels),
            "similar_case_count": [int(counts.get(label, 0)) for label in labels],
            "selection_rate": [float(counts.get(label, 0)) / denominator for label in labels],
        }
    )


def _rate(frame: pd.DataFrame, column: str, condition: pd.Series | None = None) -> float:
    if column not in frame.columns or frame.empty:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    if condition is not None:
        values = values[condition]
    values = values.dropna()
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def compute_success_rate(
    similar_cases: pd.DataFrame,
    action_col: str = "combined_gn_action",
    labels: Sequence[str] = ACTION_LABELS,
    normal_oocyte_min: float = 4,
    normal_oocyte_max: float = 20,
    mii_threshold: float = 4,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in labels:
        subset = similar_cases[similar_cases[action_col] == label].copy() if action_col in similar_cases else similar_cases.iloc[0:0].copy()
        support = len(subset)
        if support == 0:
            rows.append(
                {
                    "action": label,
                    "ovarian_response_success_rate": np.nan,
                    "mii_success_rate": np.nan,
                    "ohss_free_rate": np.nan,
                }
            )
            continue
        oocytes = pd.to_numeric(subset.get("target_oocytes"), errors="coerce") if "target_oocytes" in subset else pd.Series(dtype=float)
        mii = pd.to_numeric(subset.get("target_mii"), errors="coerce") if "target_mii" in subset else pd.Series(dtype=float)
        ohss = pd.to_numeric(subset.get("target_ohss_flag"), errors="coerce") if "target_ohss_flag" in subset else pd.Series(dtype=float)
        rows.append(
            {
                "action": label,
                "ovarian_response_success_rate": float(((oocytes >= normal_oocyte_min) & (oocytes <= normal_oocyte_max)).mean()) if len(oocytes.dropna()) else np.nan,
                "mii_success_rate": float((mii >= mii_threshold).mean()) if len(mii.dropna()) else np.nan,
                "ohss_free_rate": float((ohss <= 0).mean()) if len(ohss.dropna()) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_similar_action_statistics(
    query: pd.Series | Mapping[str, Any],
    similar_cases: pd.DataFrame,
    model_recommended_action: str,
    model_probabilities: Mapping[str, float] | None = None,
    k: int | None = None,
    action_col: str = "combined_gn_action",
    labels: Sequence[str] = ACTION_LABELS,
) -> pd.DataFrame:
    query_series = pd.Series(query)
    k = int(k or len(similar_cases))
    model_probabilities = dict(model_probabilities or {})
    selection = compute_selection_rate(similar_cases, action_col=action_col, labels=labels)
    success = compute_success_rate(similar_cases, action_col=action_col, labels=labels)
    summary = selection.merge(success, on="action", how="left")
    summary.insert(0, "knn_k", k)
    summary.insert(0, "model_probability_decrease", float(model_probabilities.get("decrease", np.nan)))
    summary.insert(0, "model_probability_maintain", float(model_probabilities.get("maintain", np.nan)))
    summary.insert(0, "model_probability_increase", float(model_probabilities.get("increase", np.nan)))
    summary.insert(0, "model_recommended_action", model_recommended_action)
    summary.insert(0, "snapshot_id", query_series.get("visit_uid", query_series.get("canonical_visit_key", pd.NA)))
    summary.insert(0, "cycle_id", query_series.get("cycle_uid", query_series.get("cycle_id", pd.NA)))
    summary.insert(0, "patient_id", query_series.get("art_id", query_series.get("patient_id", pd.NA)))
    return summary


def plot_similar_action_selection_rate(stats: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(stats["action"], stats["selection_rate"], color="#0ea5e9")
    ax.set_ylim(0, max(0.1, float(stats["selection_rate"].max()) * 1.25))
    ax.set_ylabel("Selection rate")
    ax.set_title("Similar-patient action selection rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_similar_action_success_rate(stats: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = stats.set_index("action")[metrics]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Similar-patient action success rate")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_similar_case_distance(similar_cases: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    distances = pd.to_numeric(similar_cases.get("distance"), errors="coerce").dropna().sort_values().reset_index(drop=True)
    ax.plot(np.arange(1, len(distances) + 1), distances, marker="o", color="#0ea5e9")
    ax.set_xlabel("Similar case rank")
    ax.set_ylabel("Distance")
    ax.set_title("Similar case distance distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_similar_patient_report(
    similar_cases: pd.DataFrame,
    output_path: str | Path,
    action_col: str = "combined_gn_action",
) -> pd.DataFrame:
    columns = [
        "art_id",
        "cycle_uid",
        "visit_uid",
        "distance",
        action_col,
        "target_oocytes",
        "target_mii",
        "target_ohss_flag",
    ]
    available = [column for column in columns if column in similar_cases.columns]
    report = similar_cases[available].copy()
    rename = {
        "art_id": "similar_patient_id",
        "cycle_uid": "similar_cycle_id",
        "visit_uid": "similar_snapshot_id",
        action_col: "actual_action",
        "target_oocytes": "oocytes",
        "target_mii": "MII",
        "target_ohss_flag": "OHSS",
    }
    report = report.rename(columns=rename)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def export_similar_action_report(
    query: pd.Series,
    similar_cases: pd.DataFrame,
    output_dir: str | Path,
    model_recommended_action: str,
    model_probabilities: Mapping[str, float] | None = None,
    k: int = DEFAULT_K_SIMILAR,
    action_col: str = "combined_gn_action",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize_similar_action_statistics(
        query,
        similar_cases,
        model_recommended_action=model_recommended_action,
        model_probabilities=model_probabilities,
        k=k,
        action_col=action_col,
    )
    stats.to_csv(output_dir / "layer1_knn_similar_action_stats.csv", index=False)
    export_similar_patient_report(similar_cases, output_dir / "layer1_knn_similar_patient_table.csv", action_col=action_col)
    plot_similar_action_selection_rate(stats, output_dir / "layer1_knn_selection_rate.png")
    plot_similar_action_success_rate(stats, output_dir / "layer1_knn_success_rate.png")
    plot_similar_case_distance(similar_cases, output_dir / "layer1_knn_similar_case_distance.png")
    return stats
