from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS, CTFE_ID_TO_DOSE

SAMPLING_MODES = ("none", "class_sqrt", "stage_class_sqrt")


def _training_labels(y: np.ndarray, train_mask: np.ndarray) -> pd.Series:
    ids = np.asarray(y)[np.asarray(train_mask, dtype=bool)]
    return pd.Series([CTFE_ID_TO_DOSE[int(value)] for value in ids], dtype="object")


def build_training_sample_weights(
    y: np.ndarray,
    row_index: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    sampling_mode: str = "none",
) -> tuple[np.ndarray | None, pd.DataFrame]:
    """Build sampler weights from training labels/stages only.

    Validation and test rows are excluded before counts are calculated. Labels
    are allowed here because sampling is a training-time operation, not an
    inference feature or candidate selection signal.
    """

    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown CTFE sampling_mode: {sampling_mode}")
    mask = np.asarray(train_mask, dtype=bool)
    if len(mask) != len(y) or len(row_index) != len(y):
        raise ValueError("Sampling inputs must share the same row count.")
    if not mask.any():
        raise ValueError("Sampling requires at least one training row.")
    labels = _training_labels(y, mask)
    train_rows = row_index.loc[mask].reset_index(drop=True)
    if sampling_mode == "stage_class_sqrt":
        if "ctfe_stage_group" not in train_rows.columns:
            raise ValueError("stage_class_sqrt sampling requires ctfe_stage_group.")
        strata = train_rows["ctfe_stage_group"].astype(str).fillna("unknown") + "|" + labels.astype(str)
    else:
        strata = labels.astype(str)
    counts = strata.value_counts(dropna=False)
    if sampling_mode == "none":
        per_row = np.ones(len(strata), dtype=float)
        sampler_weights = None
    else:
        per_row = strata.map(lambda value: 1.0 / np.sqrt(float(counts.loc[value]))).to_numpy(dtype=float)
        per_row = per_row / float(per_row.mean())
        sampler_weights = per_row
    distribution = (
        pd.DataFrame({"sampling_stratum": strata, "per_row_weight": per_row})
        .groupby("sampling_stratum", as_index=False)
        .agg(train_count=("sampling_stratum", "size"), per_row_weight=("per_row_weight", "first"))
    )
    distribution["sampling_mode"] = sampling_mode
    distribution["raw_share"] = distribution["train_count"] / float(distribution["train_count"].sum())
    weighted_count = distribution["train_count"] * distribution["per_row_weight"]
    distribution["expected_sampling_share"] = weighted_count / float(weighted_count.sum())
    return sampler_weights, distribution[
        ["sampling_mode", "sampling_stratum", "train_count", "raw_share", "per_row_weight", "expected_sampling_share"]
    ].sort_values("sampling_stratum").reset_index(drop=True)


def summarize_sampling_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"sampling_mode", "split", "seed", "macro_f1", "weighted_f1"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Sampling metrics missing columns: {sorted(missing)}")
    metric_columns = [column for column in ["accuracy", "macro_f1", "weighted_f1", "log_loss"] if column in metrics.columns]
    rows = []
    for (sampling_mode, split), group in metrics.groupby(["sampling_mode", "split"], sort=True):
        row: dict[str, object] = {"sampling_mode": sampling_mode, "split": split, "n_splits": int(group["seed"].nunique())}
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def select_validation_sampling_mode(
    metrics: pd.DataFrame,
    *,
    control_mode: str = "none",
    min_improved_seeds: int = 2,
    max_macro_drop: float = 0.01,
    min_weighted_gain: float = 0.0,
) -> dict[str, object]:
    """Choose a sampling strategy from validation results only."""

    valid = metrics[metrics["split"].astype(str).eq("valid")].copy()
    if valid.empty:
        raise ValueError("Sampling selection requires validation rows.")
    required = {"sampling_mode", "seed", "weighted_f1", "macro_f1"}
    missing = required.difference(valid.columns)
    if missing:
        raise ValueError(f"Sampling selection missing validation columns: {sorted(missing)}")
    summary = (
        valid.groupby("sampling_mode", as_index=False)
        .agg(
            weighted_f1_mean=("weighted_f1", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            n_splits=("seed", "nunique"),
        )
    )
    control_rows = summary[summary["sampling_mode"].astype(str).eq(control_mode)]
    if control_rows.empty:
        raise ValueError(f"Sampling selection requires control mode {control_mode!r}.")
    control = control_rows.iloc[0]
    control_seed = valid[valid["sampling_mode"].astype(str).eq(control_mode)].set_index("seed")["weighted_f1"]
    qualified: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        mode = str(row["sampling_mode"])
        if mode == control_mode:
            continue
        candidate_seed = valid[valid["sampling_mode"].astype(str).eq(mode)].set_index("seed")["weighted_f1"]
        paired = pd.concat([candidate_seed.rename("candidate"), control_seed.rename("control")], axis=1).dropna()
        improved_seeds = int((paired["candidate"] > paired["control"]).sum())
        if (
            float(row["weighted_f1_mean"]) > float(control["weighted_f1_mean"]) + float(min_weighted_gain)
            and float(row["macro_f1_mean"]) >= float(control["macro_f1_mean"]) - float(max_macro_drop)
            and improved_seeds >= int(min_improved_seeds)
        ):
            qualified.append({**row.to_dict(), "improved_seed_count": improved_seeds})
    if not qualified:
        return {
            "sampling_mode": control_mode,
            "accepted": False,
            "reason": "no_sampling_mode_improved_validation_with_guardrails",
            "selection_split": "valid",
            "test_used_for_selection": False,
            "control_valid_weighted_f1_mean": float(control["weighted_f1_mean"]),
            "control_valid_macro_f1_mean": float(control["macro_f1_mean"]),
        }
    best = sorted(qualified, key=lambda row: (float(row["weighted_f1_mean"]), float(row["macro_f1_mean"])), reverse=True)[0]
    return {
        **best,
        "accepted": True,
        "reason": "validation_sampling_gain_with_macro_and_seed_guardrails",
        "selection_split": "valid",
        "test_used_for_selection": False,
        "control_valid_weighted_f1_mean": float(control["weighted_f1_mean"]),
        "control_valid_macro_f1_mean": float(control["macro_f1_mean"]),
    }


def build_sampling_protocol_metadata(seeds: Iterable[int]) -> dict[str, object]:
    return {
        "seeds": [int(seed) for seed in seeds],
        "sampling_modes": list(SAMPLING_MODES),
        "sampling_scope": "train_only",
        "sampling_weight_rule": "inverse_sqrt_frequency",
        "selection_split": "valid",
        "test_used_for_selection": False,
        "updates_current_pointer": False,
        "split_grain": "cycle_uid",
    }