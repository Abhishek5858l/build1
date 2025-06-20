from __future__ import annotations

"""
Transformation utilities that plug into CustomConstrainedRidge (and friends).

* Spec-driven: each step dict has `method`, `columns`, optional `params`, `name`
* Alignment-safe: TransformationPipeline.steps stores (name, transformer, columns)
* Robust: accepts "params": [0,1] as shorthand for MinMaxScaler feature_range
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

__all__ = ["DataTransformer", "TransformationPipeline"]


# ---------------------------------------------------------------------------
# Low-level single-scaler wrapper
# ---------------------------------------------------------------------------


class DataTransformer:
    """Wrap one scikit-learn scaler and a subset of columns."""

    def __init__(self, method: str, params: Dict[str, Any] | List[Any] | None, columns: List[str]):
        # ---------- coerce params to dict ----------------------------------
        if params is None:
            params = {}
        elif isinstance(params, list):
            # interpret [low, high] as feature_range for MinMaxScaler
            if len(params) == 2 and all(isinstance(x, (int, float)) for x in params):
                params = {"feature_range": tuple(params)}
            else:
                raise ValueError("'params' list must have exactly two numeric values")
        elif not isinstance(params, dict):
            raise ValueError("'params' must be a dict or a two-element list")

        self.method = method.lower()
        self.params: Dict[str, Any] = params
        self.columns = columns

        # ---------- choose scaler ------------------------------------------
        if self.method == "standard":
            self.scaler = StandardScaler(**self.params)

        elif self.method == "minmax":
            # JSON might still contain list under params ➜ convert here too
            fr = self.params.get("feature_range")
            if isinstance(fr, list):
                self.params["feature_range"] = tuple(fr)
            self.scaler = MinMaxScaler(**self.params)

        elif self.method == "robust":
            self.scaler = RobustScaler(**self.params)

        else:
            raise ValueError(f"Unsupported transformation method '{method}'")

    # ---------- scikit-learn-style API -------------------------------------
    def fit(self, df: pd.DataFrame) -> "DataTransformer":
        self.scaler.fit(df[self.columns].values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler.transform(df[self.columns].values)
        df_out = df.copy()
        df_out[self.columns] = scaled
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler.fit_transform(df[self.columns].values)
        df_out = df.copy()
        df_out[self.columns] = scaled
        return df_out


# ---------------------------------------------------------------------------
# High-level ordered pipeline
# ---------------------------------------------------------------------------


class TransformationPipeline:
    """An ordered list of :class:`DataTransformer` steps."""

    def __init__(self, specs: List[Dict[str, Any]]):
        if not isinstance(specs, list):
            raise TypeError("'specs' must be a list of dicts")

        self.steps: List[Tuple[str, DataTransformer, List[str]]] = []

        for idx, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise ValueError("Each transformation spec must be a dict")

            name: str = spec.get("name", f"step_{idx}")
            method: str | None = spec.get("method")
            columns: List[str] | None = spec.get("columns")
            params = spec.get("params", {})

            if method is None:
                raise ValueError(f"Transformation spec '{name}' missing 'method'")
            if not columns:
                raise ValueError(f"Transformation spec '{name}' must include non-empty 'columns'")
            if not isinstance(columns, list):
                raise TypeError(f"'columns' in spec '{name}' must be a list of str")

            transformer = DataTransformer(method=method, params=params, columns=columns)
            self.steps.append((name, transformer, columns))

    # ---------- pipeline API ----------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for _, transformer, _ in self.steps:
            out = transformer.fit_transform(out)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for _, transformer, _ in self.steps:
            out = transformer.transform(out)
        return out

    # ---------- helpers ----------------------------------------------------
    def __repr__(self) -> str:
        names = ", ".join(name for name, _, _ in self.steps) or "<empty>"
        return f"TransformationPipeline([{names}])"

    def to_spec(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "method": transformer.method,
                "columns": columns,
                "params": transformer.params,
            }
            for name, transformer, columns in self.steps
        ]