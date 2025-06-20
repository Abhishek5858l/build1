import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression


# Add this import at the top of your file
from .results import StandardResults  # Import the standardized evaluation


class GeneralStackedRegression(BaseEstimator, RegressorMixin):
    """
    General Stacked Regression for any domain
    Independent of transformations and domain-specific features
    """
    def __init__(self, 
                 base_estimators: List[BaseEstimator],
                 meta_estimator: BaseEstimator = None,
                 cv_folds: int = 5,
                 use_probas: bool = False,
                 random_state: int = 42):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator or LinearRegression()
        self.cv_folds = cv_folds
        self.use_probas = use_probas
        self.random_state = random_state
        self.fitted_base_estimators_ = None
        self.fitted_meta_estimator_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GeneralStackedRegression':
        X = np.array(X)
        y = np.array(y)
        
        # Step 1: Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Step 2: Fit base estimators on full training data
        self.fitted_base_estimators_ = []
        for estimator in self.base_estimators:
            fitted_estimator = estimator.__class__(**estimator.get_params())
            fitted_estimator.fit(X, y)
            self.fitted_base_estimators_.append(fitted_estimator)
        
        # Step 3: Fit meta-estimator on meta-features
        self.fitted_meta_estimator_ = self.meta_estimator.__class__(**self.meta_estimator.get_params())
        self.fitted_meta_estimator_.fit(meta_features, y)
        
        self.is_fitted = True
        return self
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        # FIXED: X.shape[0] instead of X.shape
        meta_features = np.zeros((X.shape[0], len(self.base_estimators)))
        
        for i, estimator in enumerate(self.base_estimators):
            cv_predictions = cross_val_predict(
                estimator, X, y, cv=kfold, method='predict'
            )
            meta_features[:, i] = cv_predictions
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        # FIXED: X.shape[0] instead of X.shape
        base_predictions = np.zeros((X.shape[0], len(self.fitted_base_estimators_)))
        
        for i, estimator in enumerate(self.fitted_base_estimators_):
            base_predictions[:, i] = estimator.predict(X)
        
        final_predictions = self.fitted_meta_estimator_.predict(base_predictions)
        return final_predictions
    
    def get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = np.array(X)
        # FIXED: X.shape[0] instead of X.shape  
        base_predictions = np.zeros((X.shape[0], len(self.fitted_base_estimators_)))
        
        for i, estimator in enumerate(self.fitted_base_estimators_):
            base_predictions[:, i] = estimator.predict(X)
        
        return base_predictions
    
    def get_meta_weights(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.fitted_meta_estimator_, 'coef_'):
            return self.fitted_meta_estimator_.coef_
        else:
            return None
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    # NEW: Added StandardResults compatibility
    def get_results(self, X_train, y_train, X_test=None, y_test=None, feature_names=None):
        """Get standardized results for stacked regression model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating results")
        
        # Make predictions
        train_pred = self.predict(X_train)
        test_pred = self.predict(X_test) if X_test is not None else None
        
        # Use base model names as feature names if not provided
        if not feature_names:
            feature_names = [f"base_model_{i}_{type(est).__name__}" 
                           for i, est in enumerate(self.base_estimators)]
        
        # Create pseudo-model for StandardResults compatibility
        pseudo_model = self._create_pseudo_model_for_results()
        
        # Get standardized evaluation
        return StandardResults.evaluate(
            model_name=self.__class__.__name__,
            model_instance=pseudo_model,
            X_train=X_train,
            y_train=y_train,
            y_train_pred=train_pred,
            X_test=X_test,
            y_test=y_test,
            y_test_pred=test_pred,
            feature_names=feature_names
        )
    
    def _create_pseudo_model_for_results(self):
        """Create pseudo-model for StandardResults compatibility"""
        class PseudoStackedModel:
            def __init__(self, stacked_model):
                # Use meta-estimator coefficients as primary coefficients
                if hasattr(stacked_model.fitted_meta_estimator_, 'coef_'):
                    self.coef_ = stacked_model.fitted_meta_estimator_.coef_
                else:
                    # Fallback: equal weights for all base models
                    n_base = len(stacked_model.fitted_base_estimators_)
                    self.coef_ = np.ones(n_base) / n_base
                
                self.intercept_ = getattr(stacked_model.fitted_meta_estimator_, 'intercept_', 0.0)
                
                # Stacking-specific attributes
                self.base_model_count = len(stacked_model.fitted_base_estimators_)
                self.meta_model_type = type(stacked_model.fitted_meta_estimator_).__name__
                self.stacking_method = 'cross_validation'
                self.cv_folds = stacked_model.cv_folds
        
        return PseudoStackedModel(self)


class GroupedStackedRegression(GeneralStackedRegression):
    """Extension for grouped data (like regions, brands, etc.)"""
    
    def __init__(self, 
                 base_estimators: List[BaseEstimator],
                 group_column: str,
                 meta_estimator: BaseEstimator = None,
                 include_interactions: bool = True,
                 cv_folds: int = 5,
                 random_state: int = 42):
        super().__init__(base_estimators, meta_estimator, cv_folds, random_state=random_state)
        self.group_column = group_column
        self.include_interactions = include_interactions
        self.groups_ = None
        self.feature_names_ = None
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'GroupedStackedRegression':
        self.groups_ = sorted(X[self.group_column].unique())
        
        if self.include_interactions:
            X_processed = self._create_group_interactions(X)
        else:
            X_processed = X.drop(columns=[self.group_column])
        
        X_array = X_processed.values
        y_array = np.array(y)
        
        return super().fit(X_array, y_array)
    
    def _create_group_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = X.copy()
        feature_cols = [col for col in X.columns if col != self.group_column]
        self.feature_names_ = feature_cols.copy()
        
        # Create group dummies
        for group in self.groups_:
            X_processed[f"group_{group}"] = (X[self.group_column] == group).astype(int)
        
        # Create interactions
        for group in self.groups_:
            for feature in feature_cols:
                interaction_name = f"{group}_X_{feature}"
                X_processed[interaction_name] = (
                    X_processed[f"group_{group}"] * X_processed[feature]
                )
        
        X_processed = X_processed.drop(columns=[self.group_column])
        return X_processed
