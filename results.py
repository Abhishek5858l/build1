import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_percentage_error, mean_absolute_error,
    r2_score, mean_squared_error
)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from typing import Any, Optional, List, Dict

class StandardResults:
    """
    Standardized evaluation that ANY model can use
    No comparison - just consistent results format
    """
    
    @staticmethod
    def evaluate(model_name: str,
                model_instance: Any,
                X_train: np.ndarray,
                y_train: np.ndarray, 
                y_train_pred: np.ndarray,
                X_test: Optional[np.ndarray] = None,
                y_test: Optional[np.ndarray] = None,
                y_test_pred: Optional[np.ndarray] = None,
                feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Standard evaluation for ANY model type
        Returns consistent DataFrame format
        """
        
        n_features = X_train.shape[1] if hasattr(X_train, 'shape') else 0
        n_train = len(y_train)
        n_test = len(y_test) if y_test is not None else 0
        
        # Core metrics
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': model_name,
            'model_type': type(model_instance).__name__,
            'n_train_samples': n_train,
            'n_test_samples': n_test,
            'n_features': n_features,
            
            # Training metrics
            'mape_train': StandardResults._safe_mape(y_train, y_train_pred),
            'r2_train': r2_score(y_train, y_train_pred),
            'adj_r2_train': StandardResults._adjusted_r2(r2_score(y_train, y_train_pred), n_train, n_features),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae_train': np.mean(np.abs(y_train - y_train_pred)),
            
            # Test metrics (if available)
            'mape_test': StandardResults._safe_mape(y_test, y_test_pred) if y_test is not None else np.nan,
            'r2_test': r2_score(y_test, y_test_pred) if y_test is not None else np.nan,
            'adj_r2_test': StandardResults._adjusted_r2(r2_score(y_test, y_test_pred), n_test, n_features) if y_test is not None else np.nan,
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)) if y_test is not None else np.nan,
            'mae_test': np.mean(np.abs(y_test - y_test_pred)) if y_test is not None else np.nan,
        }
        
        # Information criteria
        aic, bic = StandardResults._calculate_aic_bic(y_train, y_train_pred, n_features)
        result.update({'aic': aic, 'bic': bic})
        
        # Linear model specifics (coefficients, p-values, etc.)
        result.update(StandardResults._extract_linear_stats(model_instance, feature_names))
        
        # Custom model features (constraints, mixed effects)
        result.update(StandardResults._extract_custom_stats(model_instance))
        
        return pd.DataFrame([result])

    @staticmethod
    def evaluate_with_cv(model_name: str,
                         model_instance: Any,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         y_train_pred: np.ndarray,
                         X_test: Optional[np.ndarray] = None,
                         y_test: Optional[np.ndarray] = None,
                         y_test_pred: Optional[np.ndarray] = None,
                         feature_names: Optional[List[str]] = None,
                         cv: int = 5) -> pd.DataFrame:
        """
        Same output as evaluate(), plus K-fold CV means & stds for four
        common metrics (R², RMSE, MAE, MAPE).
        """
        # 1 ── base, single-split metrics
        base_df = StandardResults.evaluate(
            model_name, model_instance,
            X_train, y_train, y_train_pred,
            X_test, y_test, y_test_pred,
            feature_names
        )

        # 2 ── scorers
        scoring = {
            'r2': make_scorer(r2_score),
            'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
            'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
            'neg_mean_absolute_percentage_error': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
        }

        # 3 ── run cross-validation
        cv_stats = {}
        for name, scorer in scoring.items():
            try:
                scores = cross_val_score(model_instance, X_train, y_train, cv=cv, scoring=scorer)
                cv_stats[f'cv_{name}_mean'] = scores.mean()
                cv_stats[f'cv_{name}_std'] = scores.std(ddof=0)
            except Exception:
                cv_stats[f'cv_{name}_mean'] = np.nan
                cv_stats[f'cv_{name}_std'] = np.nan

        # 4 ── flip sign for "neg_" errors so that bigger = worse
        for err in ('neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'):
            key = f'cv_{err}_mean'
            if key in cv_stats and not np.isnan(cv_stats[key]):
                cv_stats[key] = -cv_stats[key]

        # 5 ── merge and return
        for k, v in cv_stats.items():
            base_df[k] = v
        return base_df
    
    @staticmethod
    def _extract_linear_stats(model, feature_names):
        """Extract coefficients, p-values, t-stats, standard errors"""
        stats = {}
        
        # Coefficients
        if hasattr(model, 'coef_'):
            coeffs = np.atleast_1d(model.coef_)
            for i, coef in enumerate(coeffs[:20]):
                stats[f'coef_{i}'] = round(coef, 6)
                if feature_names and i < len(feature_names):
                    stats[f'feature_name_{i}'] = feature_names[i]
        
        # Intercept
        if hasattr(model, 'intercept_'):
            stats['intercept'] = round(float(model.intercept_), 6)
        
        # Standard errors
        if hasattr(model, 'std_errors_'):
            for i, se in enumerate(model.std_errors_[:20]):
                stats[f'se_{i}'] = round(se, 6)
        
        # T-statistics
        if hasattr(model, 't_stats_'):
            for i, tstat in enumerate(model.t_stats_[:20]):
                stats[f'tstat_{i}'] = round(tstat, 4)
        
        # P-values
        if hasattr(model, 'p_values_'):
            for i, pval in enumerate(model.p_values_[:20]):
                stats[f'pval_{i}'] = round(pval, 6)
        
        return stats
    
    @staticmethod
    def _extract_custom_stats(model):
        """Extract statistics specific to custom model types (constraints, mixed effects)."""
        stats: Dict[str, Any] = {}

        # ── Constraint model details ─────────────────────────────────────────
        if hasattr(model, "constraints"):
            constraints = getattr(model, "constraints", None)
            applied = bool(constraints)
            pos_cnt = neg_cnt = 0

            if isinstance(constraints, dict):
                # legacy format: {'positive': [...], 'negative': [...]}
                pos_cnt = len(constraints.get("positive", []))
                neg_cnt = len(constraints.get("negative", []))
            elif isinstance(constraints, list):
                # new spec list format: [{type: 'non_negative'|'non_positive', 'features': [...]}, ...]
                for c in constraints:
                    c_type = c.get("type") if isinstance(c, dict) else None
                    feats = c.get("features", []) if isinstance(c, dict) else []
                    if c_type == "non_negative":
                        pos_cnt += len(feats) or 1  # treat "all" as 1
                    elif c_type == "non_positive":
                        neg_cnt += len(feats) or 1
            else:
                # unknown structure – count as applied but no breakdown
                applied = True

            stats.update(
                {
                    "constraints_applied": applied,
                    "positive_constraints_count": pos_cnt,
                    "negative_constraints_count": neg_cnt,
                    "constraint_violations": getattr(model, "constraint_violations_", 0),
                    "business_rules_satisfied": getattr(model, "business_rules_satisfied_", True),
                }
            )

        # ── Mixed-effects model details ──────────────────────────────────────
        if hasattr(model, "random_effects_"):
            stats.update(
                {
                    "random_effects_groups": getattr(model, "n_groups", 0),
                    "fixed_effects_count": getattr(model, "n_fixed", 0),
                    "random_effects_count": getattr(model, "n_random_total", 0),
                    "marginal_r2": getattr(model, "marginal_r2_", np.nan),
                    "conditional_r2": getattr(model, "conditional_r2_", np.nan),
                }
            )

        return stats
    
    @staticmethod
    def _safe_mape(y_true, y_pred):
        """Safe MAPE calculation"""
        try:
            return mean_absolute_percentage_error(y_true, y_pred)
        except:
            return np.nan
    
    @staticmethod
    def _adjusted_r2(r2, n, p):
        """Calculate adjusted R-squared"""
        if r2 is None or n <= p + 1:
            return np.nan
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    @staticmethod
    def _calculate_aic_bic(y_true, y_pred, n_features):
        """Calculate AIC and BIC"""
        try:
            residuals = y_true - y_pred
            rss = np.sum(residuals**2)
            n = len(y_true)
            
            if n == 0 or rss <= 0:
                return np.nan, np.nan
            
            aic = n * np.log(rss / n) + 2 * n_features
            bic = n * np.log(rss / n) + np.log(n) * n_features
            
            return aic, bic
        except:
            return np.nan, np.nan