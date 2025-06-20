from __future__ import annotations

"""
Stacked‑regression models enhanced with optional, spec‑driven data‑transformation
capabilities.

* Drop‑in replacement for the original `GeneralStackedRegression` and
  `GroupedStackedRegression` – no behavioural changes unless you pass a
  `transformation_specs` list.
* Uses the `TransformationPipeline` / `DataTransformer` utilities exactly as
  provided (unchanged – copied here for convenience). If you keep them in a
  separate module, just delete the duplicate definitions and adjust the import
  at the bottom of this header.

Typical usage
-------------
>>> pipeline_spec = [
...     {"method": "standard", "columns": ["x1", "x2"]},
...     {"method": "minmax",  "columns": ["age"], "params": [0, 1]}
... ]
>>> model = GeneralStackedRegression(
...     base_estimators=[RandomForestRegressor(), GradientBoostingRegressor()],
...     transformation_specs=pipeline_spec
... )
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# ---------------------------------------------------------------------------
# Transformation utilities (verbatim copy – no behavioural changes)
# ---------------------------------------------------------------------------

__all__ = [
    "DataTransformer",
    "TransformationPipeline",
    "GeneralStackedRegression",
    "GroupedStackedRegression",
]


class DataTransformer:
    """Wrap one scikit‑learn scaler and a subset of columns."""

    def __init__(
        self,
        method: str,
        params: Dict[str, Any] | List[Any] | None,
        columns: List[str],
    ):
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
            raise ValueError("'params' must be a dict or a two‑element list")

        self.method = method.lower()
        self.params: Dict[str, Any] = params
        self.columns = columns

        # ---------- choose scaler ------------------------------------------
        if self.method == "standard":
            self.scaler = StandardScaler(**self.params)
        elif self.method == "minmax":
            fr = self.params.get("feature_range")
            if isinstance(fr, list):
                self.params["feature_range"] = tuple(fr)
            self.scaler = MinMaxScaler(**self.params)
        elif self.method == "robust":
            self.scaler = RobustScaler(**self.params)
        else:
            raise ValueError(f"Unsupported transformation method '{method}'")

    # ---------- scikit‑learn‑style API -------------------------------------
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
                raise ValueError(
                    f"Transformation spec '{name}' must include non‑empty 'columns'"
                )
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
    def __repr__(self) -> str:  # pragma: no cover – human‑facing only
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


# ---------------------------------------------------------------------------
# Stacked‑regression models WITH transformation support
# ---------------------------------------------------------------------------

# Import here to avoid circular imports if StandardResults lives elsewhere
try:
    from .results import StandardResults  # type: ignore
except (ImportError, ModuleNotFoundError):
    StandardResults = None  # pragma: no cover – optional dependency


class GeneralStackedRegression(BaseEstimator, RegressorMixin):
    """General Stacked Regression with optional data transformations."""

    def __init__(
        self,
        base_estimators: List[BaseEstimator],
        meta_estimator: Optional[BaseEstimator] = None,
        cv_folds: int = 5,
        use_probas: bool = False,
        random_state: int = 42,
        transformation_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator or LinearRegression()
        self.cv_folds = cv_folds
        self.use_probas = use_probas
        self.random_state = random_state
        # ── Transformation pipeline ────────────────────────────────
        self.transformations = transformations
        self.transformation_pipeline = None
        if self.transformations:
            self.transformation_pipeline = TransformationPipeline(self.transformations)
        # NEW
        self.transformation_specs = transformation_specs
        self.transform_pipeline_: Optional[TransformationPipeline] = (
            TransformationPipeline(transformation_specs) if transformation_specs else None
        )

        self.fitted_base_estimators_: Optional[List[BaseEstimator]] = None
        self.fitted_meta_estimator_: Optional[BaseEstimator] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core helpers – pipeline‑aware wrappers around ndarray conversions
    # ------------------------------------------------------------------
    def _prepare_fit_X(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.transform_pipeline_ is not None:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X = self.transform_pipeline_.fit_transform(X)
        return np.asarray(X)

    def _prepare_pred_X(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.transform_pipeline_ is not None:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X = self.transform_pipeline_.transform(X)
        return np.asarray(X)

    # ------------------------------------------------------------------
    # scikit‑learn API
    # ------------------------------------------------------------------
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> "GeneralStackedRegression":
        X_arr = self._prepare_fit_X(X)
        y_arr = np.asarray(y)

        # Step 1: meta‑features via CV on *transformed* data
        meta_features = self._generate_meta_features(X_arr, y_arr)

        # Step 2: fit clones of base estimators on full data
        self.fitted_base_estimators_ = []
        for estimator in self.base_estimators:
            fitted_estimator = estimator.__class__(**estimator.get_params())
            fitted_estimator.fit(X_arr, y_arr)
            self.fitted_base_estimators_.append(fitted_estimator)

        # Step 3: fit meta‑estimator on meta‑features
        self.fitted_meta_estimator_ = self.meta_estimator.__class__(
            **self.meta_estimator.get_params()
        )
        self.fitted_meta_estimator_.fit(meta_features, y_arr)

        self.is_fitted = True
        return self

    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.base_estimators)))

        for i, estimator in enumerate(self.base_estimators):
            cv_predictions = cross_val_predict(estimator, X, y, cv=kfold, method="predict")
            meta_features[:, i] = cv_predictions
        return meta_features

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_arr = self._prepare_pred_X(X)
        base_predictions = np.zeros((X_arr.shape[0], len(self.fitted_base_estimators_)))

        for i, estimator in enumerate(self.fitted_base_estimators_):
            base_predictions[:, i] = estimator.predict(X_arr)

        final_predictions = self.fitted_meta_estimator_.predict(base_predictions)
        return final_predictions

    # ------------------------------------------------------------------
    # Convenience utilities (unchanged API)
    # ------------------------------------------------------------------
    def get_base_predictions(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        X_arr = self._prepare_pred_X(X)
        base_predictions = np.zeros((X_arr.shape[0], len(self.fitted_base_estimators_)))

        for i, estimator in enumerate(self.fitted_base_estimators_):
            base_predictions[:, i] = estimator.predict(X_arr)
        return base_predictions

    def get_meta_weights(self) -> Optional[np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        if hasattr(self.fitted_meta_estimator_, "coef_"):
            return self.fitted_meta_estimator_.coef_
        return None

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> float:
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    # NEW – unchanged signature: produces StandardResults if available
    def get_results(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating results")
        if StandardResults is None:
            raise ImportError("StandardResults could not be imported – is it installed?")

        train_pred = self.predict(X_train)
        test_pred = self.predict(X_test) if X_test is not None else None

        if not feature_names:
            feature_names = [
                f"base_model_{i}_{type(est).__name__}" for i, est in enumerate(self.base_estimators)
            ]

        pseudo_model = self._create_pseudo_model_for_results()
        return StandardResults.evaluate(
            model_name=self.__class__.__name__,
            model_instance=pseudo_model,
            X_train=X_train,
            y_train=y_train,
            y_train_pred=train_pred,
            X_test=X_test,
            y_test=y_test,
            y_test_pred=test_pred,
            feature_names=feature_names,
        )

    # helper for StandardResults
    def _create_pseudo_model_for_results(self):
        class PseudoStackedModel:
            def __init__(self, stacked_model: "GeneralStackedRegression") -> None:
                if hasattr(stacked_model.fitted_meta_estimator_, "coef_"):
                    self.coef_ = stacked_model.fitted_meta_estimator_.coef_
                else:
                    n_base = len(stacked_model.fitted_base_estimators_)
                    self.coef_ = np.ones(n_base) / n_base
                self.intercept_ = getattr(
                    stacked_model.fitted_meta_estimator_, "intercept_", 0.0
                )
                self.base_model_count = len(stacked_model.fitted_base_estimators_)
                self.meta_model_type = type(stacked_model.fitted_meta_estimator_).__name__
                self.stacking_method = "cross_validation"
                self.cv_folds = stacked_model.cv_folds

        return PseudoStackedModel(self)


class GroupedStackedRegression(GeneralStackedRegression):
    """Stacked Regression for grouped data with optional interactions and transformations."""

    def __init__(
        self,
        base_estimators: List[BaseEstimator],
        group_column: str,
        meta_estimator: Optional[BaseEstimator] = None,
        include_interactions: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
        transformation_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            base_estimators=base_estimators,
            meta_estimator=meta_estimator,
            cv_folds=cv_folds,
            use_probas=False,
            random_state=random_state,
            transformation_specs=transformation_specs,
        )
        self.group_column = group_column
        self.include_interactions = include_interactions
        self.groups_: Optional[List[Any]] = None
        self.feature_names_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # scikit‑learn API
    # ------------------------------------------------------------------
    def fit(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> "GroupedStackedRegression":
        if self.group_column not in X.columns:
            raise ValueError(f"'{self.group_column}' not found in input DataFrame")

        self.groups_ = sorted(X[self.group_column].unique())

        if self.include_interactions:
            X_processed = self._create_group_interactions(X)
        else:
            X_processed = X.drop(columns=[self.group_column])

        # Apply transformations *after* interaction creation so that scaling is
        # based on the final feature set.
        return super().fit(X_processed, y)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _create_group_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = X.copy()
        feature_cols = [col for col in X.columns if col != self.group_column]
        self.feature_names_ = feature_cols.copy()

        # group dummies
        for group in self.groups_:
            X_processed[f"group_{group}"] = (X[self.group_column] == group).astype(int)

        # interactions
        for group in self.groups_:
            for feature in feature_cols:
                interaction_name = f"{group}_X_{feature}"
                X_processed[interaction_name] = (
                    X_processed[f"group_{group}"] * X_processed[feature]
                )

        return X_processed.drop(columns=[self.group_column])
