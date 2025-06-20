# models/constraint_models.py - Combined Constraint Models with Adam Optimizer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple, Dict, Any

from .Transformation import TransformationPipeline
from .results import StandardResults
from .results import StandardResults



class CustomConstrainedRidge(BaseEstimator, RegressorMixin):
    """Ridge regression with L2 penalty, linear constraints, and optional preprocessing."""

    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.001,
        max_iter: int = 100000,
        tolerance: float = 1e-6,
        test_size: float = 0.2,
        random_state: int = 42,
        fit_intercept: bool = True,
        verbose: bool = False,
        constraints: Optional[List[Dict[str, Any]]] = None,
        transformations: Optional[List[Dict[str, Any]]] = None,
    ):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.test_size = test_size
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.constraints = constraints
        self.transformations = transformations
        self.is_fitted = False

        # Build transformation pipeline if provided
        self.transformation_pipeline: Optional[TransformationPipeline] = None
        if self.transformations:
            self.transformation_pipeline = TransformationPipeline(self.transformations)

    def _prepare_data(
        self,
        X: pd.DataFrame,
        y_col: str,
        feature_cols: Optional[List[str]] = None
    ):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        df = X.copy()

        # Apply preprocessing
        if self.transformation_pipeline:
            df = self.transformation_pipeline.fit_transform(df)

        # Determine feature names
        self.feature_names_ = feature_cols or [c for c in df.columns if c != y_col]
        if y_col not in df.columns:
            raise ValueError(f"Target column '{y_col}' not found in DataFrame")

        X_vals = df[self.feature_names_].values
        y_vals = df[y_col].values

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_vals, y_vals, test_size=self.test_size, random_state=self.random_state
        )
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.n_samples_, self.n_features_ = X_train.shape

    def _parse_constraints(self):
        parsed = []
        if not self.constraints:
            self.parsed_constraints_ = parsed
            return
        for c in self.constraints:
            t = c['type']
            feats = c.get('features', 'all')
            if feats == 'all':
                idxs = list(range(self.n_features_))
            else:
                idxs = [self.feature_names_.index(f) for f in feats]
            parsed.append({'type': t, 'indices': idxs})
        self.parsed_constraints_ = parsed

    def _apply_constraints(self):
        for cons in getattr(self, 'parsed_constraints_', []):
            idxs = cons['indices']
            if cons['type'] == 'non_positive':
                self.coef_[idxs] = np.minimum(self.coef_[idxs], 0)
            elif cons['type'] == 'non_negative':
                self.coef_[idxs] = np.maximum(self.coef_[idxs], 0)

    def fit(
        self,
        X: pd.DataFrame,
        y: str,
        x_variables: Optional[List[str]] = None   # <— renamed to match your endpoint
    ) -> "CustomConstrainedRidge":
        # Prepare data and constraints, passing x_variables through
        self._prepare_data(X, y, feature_cols=x_variables)
        self._parse_constraints()

        # Initialize parameters
        self.coef_ = np.zeros(self.n_features_)
        self.intercept_ = 0.0
        self.cost_history_ = []

        # Gradient descent
        for i in range(self.max_iter):
            preds = self.X_train.dot(self.coef_) + self.intercept_
            resid = self.y_train - preds
            grad_w = (-2/self.n_samples_) * self.X_train.T.dot(resid) + 2*self.alpha*self.coef_
            grad_b = (-2/self.n_samples_) * resid.sum()

            self.coef_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_b

            # Apply constraints periodically
            if i % 50 == 0 or i == self.max_iter - 1:
                self._apply_constraints()

            # Compute loss and check convergence
            loss = (resid**2).mean() + self.alpha * np.sum(self.coef_**2)
            self.cost_history_.append(loss)
            if i > 0 and abs(self.cost_history_[-1] - self.cost_history_[-2]) < self.tolerance:
                break

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        df = X.copy()
        if self.transformation_pipeline:
            df = self.transformation_pipeline.transform(df)
        vals = df[self.feature_names_].values
        return vals.dot(self.coef_) + self.intercept_

    def get_results(self) -> pd.DataFrame:
        # Predictions
        train_pred = self.predict(pd.DataFrame(self.X_train, columns=self.feature_names_))
        X_test_arr, test_act, test_pred = None, None, None
        if hasattr(self, 'X_test'):
            X_test_arr = self.X_test
            test_act = self.y_test
            test_pred = self.predict(pd.DataFrame(self.X_test, columns=self.feature_names_))

        # Evaluate on transformed scale
        res = StandardResults.evaluate(
            model_name=self.__class__.__name__,
            model_instance=self,
            X_train=self.X_train,
            y_train=self.y_train,
            y_train_pred=train_pred,
            X_test=X_test_arr,
            y_test=test_act,
            y_test_pred=test_pred,
            feature_names=self.feature_names_
        )

        # Unscale coefficients/intercept if preprocessing was applied
        if self.transformation_pipeline:
            b = self.coef_.copy()
            b0 = float(self.intercept_)
            # Reverse each transformation step
            for _, transformer, cols in reversed(self.transformation_pipeline.steps):
                scaler = transformer.scaler
                if hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
                    scales = scaler.scale_
                    means = scaler.mean_
                    for idx, col in enumerate(cols):
                        fi = self.feature_names_.index(col)
                        b0 -= means[idx] * b[fi] / scales[idx]
                        b[fi] /= scales[idx]
                elif hasattr(scaler, 'scale_') and hasattr(scaler, 'min_'):
                    scales = scaler.scale_
                    mins = scaler.min_
                    for idx, col in enumerate(cols):
                        fi = self.feature_names_.index(col)
                        b0 -= mins[idx] * b[fi] / scales[idx]
                        b[fi] /= scales[idx]
                else:
                    raise ValueError(f"Unsupported transformer: {type(scaler)}")
            # Append original-scale values
            for i, fname in enumerate(self.feature_names_):
                res[f"coef_orig_{i}"] = b[i]
                res[f"feature_name_orig_{i}"] = fname
            res["intercept_orig"] = b0

        return res

    def add_constraints(self, constraint_specs: List[Dict[str, Any]]) -> None:
        self.constraints = constraint_specs or None

    def get_constraints_summary(self) -> List[Dict[str, Any]]:
        if not hasattr(self, 'parsed_constraints_') or not self.parsed_constraints_:
            return []
        summary = []
        for cons in self.parsed_constraints_:
            idxs = cons['indices']
            feats = [self.feature_names_[i] for i in idxs]
            vals = [self.coef_[i] for i in idxs]
            summary.append({'type': cons['type'], 'features': feats, 'coefficients': vals})
        return summary


            
            

# class CustomConstrainedLinearRegression(BaseEstimator, RegressorMixin):
#     """
#     Linear Regression with flexible constraint support and Adam optimizer.

#     This model minimizes the Mean Squared Error (MSE) using gradient descent
#     and allows for a variety of constraints on the coefficients.
#     """

#     def __init__(self, learning_rate=0.001, iterations=10000,
#                  tolerance=1e-7, optimizer='adam', beta1=0.9, beta2=0.999,
#                  epsilon=1e-8, verbose=False, constraints=None):
#         """
#         Parameters
#         ----------
#         learning_rate : float, default=0.001
#             The step size for gradient descent updates.
#         iterations : int, default=10000
#             The maximum number of iterations for the gradient descent algorithm.
#         tolerance : float, default=1e-7
#             The tolerance for convergence. Stops if the change in cost is less than this value.
#         optimizer : {'adam', 'gd'}, default='adam'
#             The optimization algorithm to use. 'gd' for standard gradient descent.
#         beta1 : float, default=0.9
#             Exponential decay rate for the first moment estimates (for Adam).
#         beta2 : float, default=0.999
#             Exponential decay rate for the second moment estimates (for Adam).
#         epsilon : float, default=1e-8
#             A small constant for numerical stability in the Adam optimizer.
#         verbose : bool, default=False
#             If True, prints progress and convergence information.
#         constraints : list of dict, optional
#             A list of dictionaries, where each dictionary defines a constraint.
#             Example: [{'type': 'non_negative', 'features': ['my_feature']}]
#         """
#         self.learning_rate = learning_rate
#         self.iterations = iterations
#         self.tolerance = tolerance
#         self.optimizer = optimizer
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon
#         self.verbose = verbose
#         self.constraints = constraints

#     def _parse_constraints(self, n_features, feature_names=None):
#         """Parse and validate constraint definitions."""
#         if self.constraints is None:
#             return []
        
#         parsed_constraints = []
#         for constraint in self.constraints:
#             if 'type' not in constraint:
#                 raise ValueError("Each constraint must have a 'type' field.")
            
#             indices = self._get_feature_indices(constraint, n_features, feature_names)
            
#             parsed = {'type': constraint['type'], 'indices': indices}
            
#             if constraint['type'] in ['box', 'exact']:
#                 parsed['value'] = constraint['value']
#             elif constraint['type'] == 'custom':
#                 parsed['function'] = constraint['function']
            
#             parsed_constraints.append(parsed)
#         return parsed_constraints

#     def _get_feature_indices(self, constraint, n_features, feature_names):
#         """Helper to convert feature names or 'all' to indices."""
#         features = constraint.get('features', 'all')
#         if features == 'all':
#             return list(range(n_features))
        
#         indices = []
#         for f in features:
#             if isinstance(f, str):
#                 if feature_names is None:
#                     raise ValueError("feature_names must be provided when using names in constraints.")
#                 try:
#                     indices.append(feature_names.index(f))
#                 except ValueError:
#                     raise ValueError(f"Feature '{f}' not found in feature_names.")
#             elif isinstance(f, int):
#                 if not 0 <= f < n_features:
#                     raise ValueError(f"Feature index {f} is out of bounds.")
#                 indices.append(f)
#         return indices

#     def _apply_constraints(self):
#         """Apply all parsed constraints to the model coefficients."""
#         for constraint in self.parsed_constraints_:
#             indices = constraint['indices']
#             for idx in indices:
#                 if constraint['type'] == 'non_positive':
#                     self.coef_[idx] = min(0, self.coef_[idx])
#                 elif constraint['type'] == 'non_negative':
#                     self.coef_[idx] = max(0, self.coef_[idx])
#                 elif constraint['type'] == 'box':
#                     self.coef_[idx] = np.clip(self.coef_[idx], constraint['value'][0], constraint['value'][1])
#                 elif constraint['type'] == 'exact':
#                     self.coef_[idx] = constraint['value']
#                 elif constraint['type'] == 'custom':
#                     self.coef_[indices] = constraint['function'](self.coef_[indices])
#                     break # Custom functions apply to all specified indices at once

#     def fit(self, X, y, feature_names=None):
#         """
#         Fit the linear model with constraints.
        
#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             Training data.
#         y : array-like of shape (n_samples,)
#             Target values.
#         feature_names : list of str, optional
#             Names of features for applying constraints by name.
#         """
#         X, y = check_X_y(X, y)
#         self.n_samples_, self.n_features_ = X.shape
#         self.feature_names_in_ = list(feature_names) if feature_names else None

#         self.parsed_constraints_ = self._parse_constraints(self.n_features_, self.feature_names_in_)
        
#         self.coef_ = np.zeros(self.n_features_)
#         self.intercept_ = 0.0
        
#         self._apply_constraints() # Apply initial exact constraints

#         if self.optimizer == 'adam':
#             self.m_w_ = np.zeros(self.n_features_)
#             self.v_w_ = np.zeros(self.n_features_)
#             self.m_b_ = 0.0
#             self.v_b_ = 0.0
#             self.t_ = 0

#         self.cost_history_ = []
#         for i in range(self.iterations):
#             y_pred = X @ self.coef_ + self.intercept_
#             cost = np.mean((y - y_pred) ** 2)
#             self.cost_history_.append(cost)

#             if i > 0 and abs(self.cost_history_[-2] - cost) < self.tolerance:
#                 if self.verbose:
#                     print(f"Converged at iteration {i + 1}")
#                 break

#             dw = (2 / self.n_samples_) * X.T @ (y_pred - y)
#             db = (2 / self.n_samples_) * np.sum(y_pred - y)

#             if self.optimizer == 'adam':
#                 self.t_ += 1
#                 self.m_w_ = self.beta1 * self.m_w_ + (1 - self.beta1) * dw
#                 self.v_w_ = self.beta2 * self.v_w_ + (1 - self.beta2) * (dw ** 2)
#                 self.m_b_ = self.beta1 * self.m_b_ + (1 - self.beta1) * db
#                 self.v_b_ = self.beta2 * self.v_b_ + (1 - self.beta2) * (db ** 2)
                
#                 t_safe = min(self.t_, 1000)
#                 m_w_hat = self.m_w_ / (1 - self.beta1 ** t_safe)
#                 v_w_hat = self.v_w_ / (1 - self.beta2 ** t_safe)
#                 m_b_hat = self.m_b_ / (1 - self.beta1 ** t_safe)
#                 v_b_hat = self.v_b_ / (1 - self.beta2 ** t_safe)
                
#                 self.coef_ -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
#                 self.intercept_ -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
#             else: # Standard Gradient Descent
#                 self.coef_ -= self.learning_rate * dw
#                 self.intercept_ -= self.learning_rate * db
            
#             self._apply_constraints()

#         self.is_fitted_ = True
#         return self

#     def predict(self, X):
#         """Make predictions using the fitted model."""
#         check_is_fitted(self)
#         X = check_array(X)
#         return X @ self.coef_ + self.intercept_
    
#     def get_results(self, X_train, y_train, X_test=None, y_test=None, feature_names=None):
#         """Get standardized results for constrained linear regression model"""
#         if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
#             raise ValueError("Model must be fitted before generating results")
        
#         # Make predictions
#         train_pred = self.predict(X_train)
#         test_pred = self.predict(X_test) if X_test is not None else None
        
#         # Use feature names if provided, otherwise use stored ones or generate
#         if not feature_names:
#             if hasattr(self, 'feature_names_in_') and self.feature_names_in_:
#                 feature_names = self.feature_names_in_
#             else:
#                 feature_names = [f"feature_{i}" for i in range(self.n_features_)]
        
#         # Enhance model with constraint information for StandardResults[1][6]
#         self._prepare_constraint_info_for_results()
        
#         # Get standardized evaluation
#         return StandardResults.evaluate(
#             model_name=self.__class__.__name__,
#             model_instance=self,
#             X_train=X_train,
#             y_train=y_train,
#             y_train_pred=train_pred,
#             X_test=X_test,
#             y_test=y_test,
#             y_test_pred=test_pred,
#             feature_names=feature_names
#         )
    
#     def _prepare_constraint_info_for_results(self):
#         """Prepare constraint information for StandardResults compatibility"""
#         # Initialize constraint summary
#         self.constraints = {
#             'positive': [],
#             'negative': [],
#             'box': [],
#             'exact': [],
#             'custom': []
#         }
        
#         positive_count = 0
#         negative_count = 0
        
#         if hasattr(self, 'parsed_constraints_') and self.parsed_constraints_:
#             # Process each constraint[2][3]
#             for constraint in self.parsed_constraints_:
#                 constraint_type = constraint['type']
#                 indices = constraint['indices']
                
#                 # Categorize constraints[2][3][5]
#                 if constraint_type == 'non_negative':
#                     self.constraints['positive'].extend(indices)
#                     positive_count += len(indices)
#                 elif constraint_type == 'non_positive':
#                     self.constraints['negative'].extend(indices)
#                     negative_count += len(indices)
#                 elif constraint_type == 'box':
#                     self.constraints['box'].extend(indices)
#                 elif constraint_type == 'exact':
#                     self.constraints['exact'].extend(indices)
#                 elif constraint_type == 'custom':
#                     self.constraints['custom'].extend(indices)
        
#         # Add constraint summary attributes for StandardResults
#         self.positive_constraints_count = positive_count
#         self.negative_constraints_count = negative_count
#         self.constraint_violations_ = self._count_constraint_violations()
#         self.business_rules_satisfied_ = (self.constraint_violations_ == 0)
        
#         # Add optimization info
#         self.optimizer_type_ = self.optimizer
#         self.final_cost_ = self.cost_history_[-1] if hasattr(self, 'cost_history_') else None
#         self.iterations_completed_ = len(self.cost_history_) if hasattr(self, 'cost_history_') else None
    
#     def _count_constraint_violations(self):
#         """Count how many constraints are currently violated"""
#         violations = 0
        
#         if not hasattr(self, 'parsed_constraints_'):
#             return violations
            
#         for constraint in self.parsed_constraints_:
#             constraint_type = constraint['type']
#             indices = constraint['indices']
            
#             for idx in indices:
#                 if constraint_type == 'non_negative' and self.coef_[idx] < 0:
#                     violations += 1
#                 elif constraint_type == 'non_positive' and self.coef_[idx] > 0:
#                     violations += 1
#                 elif constraint_type == 'box':
#                     lower, upper = constraint['value']
#                     if self.coef_[idx] < lower or self.coef_[idx] > upper:
#                         violations += 1
#                 elif constraint_type == 'exact':
#                     if abs(self.coef_[idx] - constraint['value']) > 1e-6:
#                         violations += 1
        
#         return violations
    
#     def get_optimization_summary(self):
#         """Get detailed optimization summary"""
#         if not hasattr(self, 'cost_history_'):
#             return {}
            
#         return {
#             'optimizer': self.optimizer,
#             'initial_cost': self.cost_history_[0] if self.cost_history_ else None,
#             'final_cost': self.cost_history_[-1] if self.cost_history_ else None,
#             'cost_reduction': (self.cost_history_[0] - self.cost_history_[-1]) if len(self.cost_history_) > 1 else 0,
#             'iterations_completed': len(self.cost_history_),
#             'convergence_achieved': len(self.cost_history_) < self.iterations,
#             'constraint_violations': self.constraint_violations_ if hasattr(self, 'constraint_violations_') else 0
#         }