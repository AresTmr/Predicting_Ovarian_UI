from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureBundle:
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]


def select_feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str]) -> FeatureBundle:
    resolved = [column for column in feature_columns if column in frame.columns]
    categorical = [column for column in resolved if not is_numeric_dtype(frame[column])]
    numeric = [column for column in resolved if column not in categorical]
    return FeatureBundle(feature_columns=resolved, categorical_columns=categorical, numeric_columns=numeric)


def build_preprocessor(bundle: FeatureBundle) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, bundle.numeric_columns),
            ("categorical", categorical_pipe, bundle.categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
