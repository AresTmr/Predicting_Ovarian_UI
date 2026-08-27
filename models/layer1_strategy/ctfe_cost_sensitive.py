from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS

CTFE_COST_PROFILES: dict[str, dict[str, float]] = {
    "none": {"stop": 1.0, "decrease": 1.0, "low": 1.0, "medium": 1.0, "high": 1.0},
    "mild_110_115": {"stop": 1.10, "decrease": 1.0, "low": 1.0, "medium": 1.0, "high": 1.15},
    "mild_115_125": {"stop": 1.15, "decrease": 1.0, "low": 1.0, "medium": 1.0, "high": 1.25},
    "mild_120_135": {"stop": 1.20, "decrease": 1.0, "low": 1.0, "medium": 1.0, "high": 1.35},
}
COST_PROFILE_NAMES = tuple(CTFE_COST_PROFILES)


def get_ctfe_cost_vector(profile_name: str) -> np.ndarray:
    """Return fixed per-class loss weights ordered by the CTFE dose labels."""

    if profile_name not in CTFE_COST_PROFILES:
        raise ValueError(f"Unknown CTFE class cost profile: {profile_name}")
    profile = CTFE_COST_PROFILES[profile_name]
    return np.asarray([profile[label] for label in CTFE_DOSE_LABELS], dtype=float)


def cost_profile_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"cost_profile": name, **weights} for name, weights in CTFE_COST_PROFILES.items()]
    )


def _with_minority_metric(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"cost_profile", "split", "seed", "weighted_f1", "macro_f1", "f1_stop", "f1_high"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Cost-sensitive metrics missing columns: {sorted(missing)}")
    enriched = metrics.copy()
    enriched["minority_mean_f1"] = (
        pd.to_numeric(enriched["f1_stop"], errors="coerce") + pd.to_numeric(enriched["f1_high"], errors="coerce")
    ) / 2.0
    return enriched


def summarize_cost_sensitive_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    enriched = _with_minority_metric(metrics)
    metric_columns = [
        column for column in ["accuracy", "macro_f1", "weighted_f1", "log_loss", "f1_stop", "f1_high", "minority_mean_f1"]
        if column in enriched.columns
    ]
    rows: list[dict[str, object]] = []
    for (profile, split), group in enriched.groupby(["cost_profile", "split"], sort=True):
        row: dict[str, object] = {"cost_profile": profile, "split": split, "n_splits": int(group["seed"].nunique())}
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def select_validation_cost_profile(
    metrics: pd.DataFrame,
    *,
    control_profile: str = "none",
    min_improved_seeds: int = 2,
    max_weighted_f1_drop: float = 0.003,
    max_log_loss_increase: float = 0.02,
) -> dict[str, object]:
    """Select mild loss costs using validation minority performance with safety floors."""

    valid = _with_minority_metric(metrics[metrics["split"].astype(str).eq("valid")].copy())
    if valid.empty:
        raise ValueError("Cost-sensitive selection requires validation rows.")
    grouped = (
        valid.groupby("cost_profile", as_index=False)
        .agg(
            weighted_f1_mean=("weighted_f1", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            minority_mean_f1_mean=("minority_mean_f1", "mean"),
            log_loss_mean=("log_loss", "mean"),
            n_splits=("seed", "nunique"),
        )
    )
    control_rows = grouped[grouped["cost_profile"].astype(str).eq(control_profile)]
    if control_rows.empty:
        raise ValueError(f"Cost-sensitive selection requires control profile {control_profile!r}.")
    control = control_rows.iloc[0]
    control_seed = valid[valid["cost_profile"].astype(str).eq(control_profile)].set_index("seed")["minority_mean_f1"]
    qualified: list[dict[str, object]] = []
    for _, row in grouped.iterrows():
        profile = str(row["cost_profile"])
        if profile == control_profile:
            continue
        candidate_seed = valid[valid["cost_profile"].astype(str).eq(profile)].set_index("seed")["minority_mean_f1"]
        paired = pd.concat([candidate_seed.rename("candidate"), control_seed.rename("control")], axis=1).dropna()
        improved_seeds = int((paired["candidate"] > paired["control"]).sum())
        if (
            float(row["minority_mean_f1_mean"]) > float(control["minority_mean_f1_mean"])
            and float(row["weighted_f1_mean"]) >= float(control["weighted_f1_mean"]) - float(max_weighted_f1_drop)
            and float(row["log_loss_mean"]) <= float(control["log_loss_mean"]) + float(max_log_loss_increase)
            and improved_seeds >= int(min_improved_seeds)
        ):
            qualified.append({**row.to_dict(), "improved_seed_count": improved_seeds})
    base = {
        "selection_split": "valid",
        "test_used_for_selection": False,
        "control_valid_weighted_f1_mean": float(control["weighted_f1_mean"]),
        "control_valid_macro_f1_mean": float(control["macro_f1_mean"]),
        "control_valid_minority_mean_f1_mean": float(control["minority_mean_f1_mean"]),
        "control_valid_log_loss_mean": float(control["log_loss_mean"]),
    }
    if not qualified:
        return {
            "cost_profile": control_profile,
            "accepted": False,
            "reason": "no_cost_profile_improved_validation_minority_with_guardrails",
            **base,
        }
    best = sorted(
        qualified,
        key=lambda row: (float(row["minority_mean_f1_mean"]), float(row["macro_f1_mean"]), float(row["weighted_f1_mean"])),
        reverse=True,
    )[0]
    return {
        **best,
        "cost_profile": str(best["cost_profile"]),
        "accepted": True,
        "reason": "validation_minority_gain_with_weighted_and_logloss_guardrails",
        **base,
    }


def build_cost_sensitive_protocol_metadata(seeds: Iterable[int]) -> dict[str, object]:
    return {
        "seeds": [int(seed) for seed in seeds],
        "cost_profiles": CTFE_COST_PROFILES,
        "training_change": "fixed_loss_weight_for_stop_high_only",
        "sampling_mode": "none",
        "selection_split": "valid",
        "test_used_for_selection": False,
        "updates_current_pointer": False,
        "split_grain": "cycle_uid",
    }
