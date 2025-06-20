import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from typing import List, Tuple, Dict, Any, Optional

# Import transformation utilities
from .Transformation import TransformationPipeline
# Import standardized evaluation
from .results import StandardResults


class BaseRegressionWrapper(BaseEstimator, RegressorMixin):
    """Base wrapper for sklearn regression models with DataFrame support
       and optional preprocessing (scaling/normalization)."""

    def __init__(
        self,
        formula: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        transformations: Optional[List[Tuple[str, str, Dict[str, Any], List[str]]]] = None
    ):
        self.formula = formula
        self.test_size = test_size
        self.random_state = random_state
        self.transformations = transformations
        self.is_fitted = False
        self.transformation_pipeline: Optional[TransformationPipeline] = None

        if self.transformations:
            self.transformation_pipeline = TransformationPipeline(self.transformations)

    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: Optional[Any] = None,
        x_variables: Optional[List[str]] = None
    ):
        """Prepare DataFrame, apply transformations, split, then build design matrices."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        df = X.copy()
        # Apply all transformations to full dataset to fit scaler parameters
        if self.transformation_pipeline:
            df = self.transformation_pipeline.fit_transform(df)

        self.data = df
        # Determine target column
        if isinstance(y, str):
            self.target_col = y
            if y not in df.columns:
                raise ValueError(f"Target column '{y}' not found in DataFrame")
        elif y is not None:
            self.target_col = 'target'
            df[self.target_col] = y
        else:
            self.target_col = df.columns[0]

        # Train/test split
        if self.test_size and self.test_size > 0:
            self.train_data, self.test_data = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state
            )
        else:
            self.train_data = df.copy()
            self.test_data = None

        # Transform test set only
        if self.transformation_pipeline and self.test_data is not None:
            self.test_data = self.transformation_pipeline.transform(self.test_data)

        # Build design matrices
        if self.formula:
            y_df, X_df = dmatrices(self.formula, self.train_data, return_type='dataframe')
        else:
            features = x_variables if x_variables else [c for c in self.train_data.columns if c != self.target_col]
            missing = [c for c in features if c not in self.train_data.columns]
            if missing:
                raise ValueError(f"Columns not found: {missing}")
            self.formula = f"{self.target_col} ~ " + " + ".join(features)
            y_df, X_df = dmatrices(self.formula, self.train_data, return_type='dataframe')
        self.feature_names = X_df.columns.tolist()
        self.y_train = y_df.values.flatten()
        self.X_train = X_df.values

    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Predict on provided DataFrame or on the test split if X is None."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        # Select data
        df_pred = self.test_data if X is None else X.copy()
        if df_pred is None:
            raise ValueError("No data available for prediction")
        if self.transformation_pipeline and X is not None:
            df_pred = self.transformation_pipeline.transform(df_pred)
        rhs = self.formula.split('~', 1)[1].strip()
        X_pred = dmatrix(rhs, df_pred, return_type='dataframe').values
        return self.model.predict(X_pred)

    def get_results(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute metrics and optionally unscale coefficients/intercept."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting results")
        # Base metrics on transformed data
        train_pred = self.model.predict(self.X_train)
        X_test_arr, test_actual, test_pred = None, None, None
        if self.test_data is not None:
            test_pred = self.predict(None)
            test_actual = self.test_data[self.target_col].values
            rhs = self.formula.split('~', 1)[1].strip()
            X_test_arr = dmatrix(rhs, self.test_data, return_type='dataframe').values
        res_df = StandardResults.evaluate(
            model_name=self.__class__.__name__,
            model_instance=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            y_train_pred=train_pred,
            X_test=X_test_arr,
            y_test=test_actual,
            y_test_pred=test_pred,
            feature_names=feature_names or self.feature_names
        )
        # Unscale coefficients/intercept if scalers were applied
        if self.transformation_pipeline:
            # Start with scaled coefs/intercept
            b = np.array(self.model.coef_, dtype=float)
            b0 = float(self.model.intercept_)
            # Iterate reverse through steps to undo scaling
            for name, transformer, cols in reversed(self.transformation_pipeline.steps):
                scaler = transformer.scaler
                # Determine scale and offset attributes
                if hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
                    scales = scaler.scale_
                    centers = scaler.mean_
                    for idx, col in enumerate(cols):
                        fi = self.feature_names.index(col)
                        b0 -= centers[idx] * b[fi] / scales[idx]
                        b[fi] /= scales[idx]
                elif hasattr(scaler, 'scale_') and hasattr(scaler, 'min_'):
                    scales = scaler.scale_
                    mins = scaler.min_
                    for idx, col in enumerate(cols):
                        fi = self.feature_names.index(col)
                        b0 -= mins[idx] * b[fi] / scales[idx]
                        b[fi] /= scales[idx]
                else:
                    raise ValueError(f"Unsupported transformer type for unscaling: {type(scaler)}")
            # Append original coefficients to result
            for i, fname in enumerate(self.feature_names):
                res_df[f"coef_orig_{i}"] = b[i]
                res_df[f"feature_name_orig_{i}"] = fname
            res_df["intercept_orig"] = b0
        return res_df

# Subclasses inherit updated logic
class EnhancedLinearRegression(BaseRegressionWrapper):
    def __init__(self, formula=None, fit_intercept=True,
                 test_size=0.2, random_state=42,
                 transformations=None):
        super().__init__(formula, test_size, random_state, transformations)
        self.fit_intercept = fit_intercept
    def fit(self, X, y=None, x_variables=None):
        self._prepare_data(X, y, x_variables)
        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        return self

class EnhancedRidgeRegression(BaseRegressionWrapper):
    def __init__(self, formula=None, alpha=1.0,
                 fit_intercept=True, test_size=0.2,
                 random_state=42, transformations=None):
        super().__init__(formula, test_size, random_state, transformations)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
    def fit(self, X, y=None, x_variables=None):
        self._prepare_data(X, y, x_variables)
        self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        return self

class EnhancedLassoRegression(BaseRegressionWrapper):
    def __init__(self, formula=None, alpha=1.0,
                 fit_intercept=True, max_iter=1000,
                 test_size=0.2, random_state=42,
                 transformations=None):
        super().__init__(formula, test_size, random_state, transformations)
        self.alpha = alpha; self.fit_intercept = fit_intercept; self.max_iter = max_iter
    def fit(self, X, y=None, x_variables=None):
        self._prepare_data(X, y, x_variables)
        self.model = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept,
                          max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        return self
    def summary(self):
        super().summary()
        if self.is_fitted:
            selected = [f for f, c in zip(self.feature_names, self.model.coef_) if abs(c) > 1e-6]
            print(f"\nSelected features ({len(selected)}/{len(self.feature_names)}): {selected}")

class EnhancedElasticNetRegression(BaseRegressionWrapper):
    def __init__(self, formula=None, alpha=1.0, l1_ratio=0.5,
                 fit_intercept=True, max_iter=1000,
                 test_size=0.2, random_state=42,
                 transformations=None):
        super().__init__(formula, test_size, random_state, transformations)
        self.alpha = alpha; self.l1_ratio = l1_ratio; self.fit_intercept = fit_intercept; self.max_iter = max_iter
    def fit(self, X, y=None, x_variables=None):
        self._prepare_data(X, y, x_variables)
        self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                               fit_intercept=self.fit_intercept,
                               max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        return self

class EnhancedBayesianRidge(BaseRegressionWrapper):
    def __init__(self, formula=None, alpha_1=1e-6, alpha_2=1e-6,
                 lambda_1=1e-6, lambda_2=1e-6,
                 fit_intercept=True, max_iter=300,
                 test_size=0.2, random_state=42,
                 transformations=None):
        super().__init__(formula, test_size, random_state, transformations)
        self.alpha_1=alpha_1; self.alpha_2=alpha_2; self.lambda_1=lambda_1; self.lambda_2=lambda_2; self.fit_intercept=fit_intercept; self.max_iter=max_iter
    def fit(self, X, y=None, x_variables=None):
        self._prepare_data(X, y, x_variables)
        self.model = BayesianRidge(alpha_1=self.alpha_1, alpha_2=self.alpha_2,
                                  lambda_1=self.lambda_1, lambda_2=self.lambda_2,
                                  fit_intercept=self.fit_intercept, n_iter=self.max_iter)
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        return self
