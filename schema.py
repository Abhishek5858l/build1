# from __future__ import annotations

# import enum
# from typing import Dict, List, Optional

# from pydantic import BaseModel, Field, root_validator, validator
# from enum import Enum
# from typing import List, Optional, Dict, Any
# from pydantic import BaseModel, Field
# from datetime import datetime
# from __future__ import annotations

# """Pydantic models & enums for the *constraint-training* API.

# This module must be imported by FastAPI endpoints (see `endpoints/constraint_training.py`).
# All definitions here are **framework-agnostic** – no FastAPI specifics.
# """

# import enum
# from typing import Dict, List, Optional

# from pydantic import BaseModel, Field, root_validator, validator

# class SimpleColumn(BaseModel):
#     """Simplified column information for X/Y selection"""
#     column_name: str = Field(..., description="Column name")
#     is_numeric: bool = Field(..., description="Suitable for Y variable (target)")

# class CombinationInfo(BaseModel):
#     """Information about available data combinations"""
#     combination: Dict[str, str] = Field(..., description="Filter combination values")
#     file_key: str = Field(..., description="File location")
#     record_count: int = Field(..., description="Number of records")

# class SimpleScopeInfo(BaseModel):
#     """Simplified scope information for variable selection"""
#     scope_id: str
#     scope_name: str
#     scope_type: str
#     description: Optional[str]
#     total_records: int
#     available_columns: List[SimpleColumn] = Field(..., description="Columns available for X/Y selection")
#     available_combinations: List[CombinationInfo] = Field(..., description="Data combinations to choose from")
#     created_at: str

# class SimpleScopeResponse(BaseModel):
#     """Simplified response for scope information"""
#     success: bool
#     scope: Optional[SimpleScopeInfo]
#     retrieved_at: str
#     message: str

# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict, Any
# from enum import Enum

# class BasicModelType(str, Enum):
#     """Basic regression model types"""
#     LINEAR = "linear"
#     RIDGE = "ridge"
#     LASSO = "lasso"
#     ELASTIC_NET = "elastic_net"
#     BAYESIAN_RIDGE = "bayesian_ridge"

# class TransformationSpec(BaseModel):
#     """Specification for a single transformation step"""
#     name: str = Field(..., description="Unique name for this transformation step")
#     method: str = Field(..., description="Transformer method: standard, minmax, robust")
#     params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for transformer")
#     columns: List[str] = Field(..., description="List of columns to apply this transformation to")

# class BasicTrainingRequest(BaseModel):
#     """Training request using your StandardResults format"""
#     scope_id: str = Field(..., description="Scope ID containing the data")
#     combination_file_key: str = Field(..., description="Specific combination file to use")
    
#     # Variable selection
#     x_variables: List[str] = Field(..., description="Feature columns (X variables)")
#     y_variable: str = Field(..., description="Target column (Y variable)")
    
#     # Model selection
#     model_type: BasicModelType = Field(..., description="Type of regression model")
    
#     # Basic training parameters
#     test_size: float = Field(default=0.2, description="Test set proportion")
#     random_state: int = Field(default=42, description="Random state for reproducibility")
    
#     # Model-specific parameters (optional)
#     alpha: Optional[float] = Field(None, description="Regularization strength for Ridge/Lasso/ElasticNet")
#     l1_ratio: Optional[float] = Field(None, description="L1 ratio for ElasticNet")
#     max_iter: Optional[int] = Field(None, description="Maximum iterations for Lasso/ElasticNet/BayesianRidge")
    
#     # New: transformation pipeline specification
#     transformations: Optional[List[TransformationSpec]] = Field(
#         default=None,
#         description="Optional list of transformations to apply before training"
#     )
    
#     # Metadata
#     training_name: Optional[str] = Field(None, description="Custom name for training run")

# class BasicTrainingResponse(BaseModel):
#     """Enhanced response including coefficients, intercept, and applied transformations"""
#     success: bool
#     training_id: str
#     training_name: str
#     scope_id: str
#     combination_used: str
    
#     # Training setup
#     x_variables: List[str]
#     y_variable: str
#     model_type: str

#     # Performance metrics
#     model_name: str
#     train_r2: float
#     test_r2: Optional[float] = None
#     train_rmse: float
#     test_rmse: Optional[float] = None
#     train_mae: float
#     test_mae: Optional[float] = None
#     train_mape: Optional[float] = None
#     test_mape: Optional[float] = None
#     train_samples: int
#     test_samples: int
#     aic: Optional[float] = None
#     bic: Optional[float] = None
    
#     # Coefficient information
#     intercept: Optional[float] = Field(default=None, description="Model intercept value")
#     coefficients: Optional[Dict[str, float]] = Field(default=None, description="Feature coefficients")

#     # Transformations
#     transformations_applied: Optional[List[TransformationSpec]] = Field(
#         default=None,
#         description="List of transformations applied during training"
#     )
    
#     # Metadata - ADD DEFAULTS HERE:
#     training_duration_seconds: float = Field(default=0.0, description="Training duration in seconds")
#     created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Training creation timestamp")

    
# #####################constraint models



# __all__ = [
#     "ConstraintType",
#     "ConstraintModelType",
#     "ConstraintDefinition",
#     "ConstraintValidationResult",
#     "TransformationSpec",
#     "ConstraintTrainingRequest",
#     "ConstraintTrainingResponse",
# ]


# class ConstraintType(str, enum.Enum):
#     """Supported coefficient constraints."""

#     NON_NEGATIVE = "non_negative"
#     NON_POSITIVE = "non_positive"
#     BOX = "box"  # lower ≤ coef ≤ upper (not implemented in CustomConstrainedRidge)
#     EXACT = "exact"  # coef == value (not implemented in CustomConstrainedRidge)
#     CUSTOM = "custom"  # arbitrary expression (user-defined)


# class ConstraintModelType(str, enum.Enum):
#     CONSTRAINED_RIDGE = "constrained_ridge"
#     CONSTRAINED_LINEAR = "constrained_linear"


# class ConstraintDefinition(BaseModel):
#     """One logical constraint applied to one **or more** coefficients."""

#     constraint_type: ConstraintType = Field(..., alias="type")
#     features: List[str] = Field(..., min_items=1)

#     min_value: Optional[float] = None
#     max_value: Optional[float] = None
#     exact_value: Optional[float] = None
#     custom_expression: Optional[str] = None

#     @root_validator(skip_on_failure=True)
#     def validate_parameters(cls, values):
#         c_type = values.get("constraint_type")
#         if c_type == ConstraintType.BOX:
#             if values.get("min_value") is None and values.get("max_value") is None:
#                 raise ValueError("BOX constraint requires min_value and/or max_value")
#         elif c_type == ConstraintType.EXACT:
#             if values.get("exact_value") is None:
#                 raise ValueError("EXACT constraint requires exact_value")
#         elif c_type == ConstraintType.CUSTOM:
#             if not values.get("custom_expression"):
#                 raise ValueError("CUSTOM constraint requires custom_expression")
#         return values

#     class Config:
#         allow_population_by_field_name = True
#         schema_extra = {
#             "example": {
#                 "type": "non_negative",
#                 "features": ["price", "discount"],
#             }
#         }


# class ConstraintValidationResult(BaseModel):
#     """Result for each constraint·feature pair after training."""

#     constraint_satisfied: bool
#     violation_magnitude: float
#     affected_features: List[str]


# class TransformationSpec(BaseModel):
#     """Schema for a pipeline step in `TransformationPipeline`."""

#     name: Optional[str] = None
#     method: str = Field(..., regex="^(standard|minmax|robust)$")
#     columns: List[str] = Field(..., min_items=1)
#     params: Dict[str, Optional[float]] = Field(default_factory=dict)

#     @validator("name", always=True)
#     def fill_name(cls, v, values):
#         if v:
#             return v
#         idx = values.get("__index__", 0)
#         return f"step_{idx}"

#     class Config:
#         schema_extra = {
#             "example": {
#                 "name": "scale_price",
#                 "method": "standard",
#                 "columns": ["price"],
#                 "params": {},
#             }
#         }


# class ConstraintTrainingRequest(BaseModel):
#     """Request body for `/train-constraint`."""

#     scope_id: str
#     combination_file_key: str
#     training_name: Optional[str] = None
#     model_type: ConstraintModelType = ConstraintModelType.CONSTRAINED_RIDGE
#     x_variables: List[str] = Field(..., min_items=1)
#     y_variable: str
#     alpha: float = 1.0
#     learning_rate: float = 0.001
#     max_iter: int = 100_000
#     tolerance: float = 1e-6
#     test_size: float = 0.2
#     random_state: int = 42
#     fit_intercept: bool = True
#     constraints: List[ConstraintDefinition] = Field(default_factory=list)
#     transformations: Optional[List[TransformationSpec]] = None

#     @validator("y_variable")
#     def ensure_y_not_in_x(cls, v, values):
#         if "x_variables" in values and v in values["x_variables"]:
#             raise ValueError("y_variable cannot also appear in x_variables")
#         return v


# class ConstraintTrainingResponse(BaseModel):
#     """Response model for `/train-constraint`."""

#     success: bool
#     training_id: str
#     training_name: str
#     scope_id: str
#     combination_used: str
#     x_variables: List[str]
#     y_variable: str
#     model_type: str
#     constraints_applied: List[ConstraintDefinition]
#     transformations_applied: List[TransformationSpec]
#     model_name: str
#     train_r2: Optional[float] = None
#     test_r2: Optional[float] = None
#     train_rmse: Optional[float] = None
#     test_rmse: Optional[float] = None
#     train_mae: Optional[float] = None
#     test_mae: Optional[float] = None
#     train_mape: Optional[float] = None
#     test_mape: Optional[float] = None
#     train_samples: Optional[int] = None
#     test_samples: Optional[int] = None
#     aic: Optional[float] = None
#     bic: Optional[float] = None
#     intercept: Optional[float] = None
#     coefficients: Dict[str, float]
#     constraint_validation: List[ConstraintValidationResult]
#     optimization_converged: bool
#     optimization_iterations: int
#     training_duration_seconds: float
#     created_at: str

#     class Config:
#         orm_mode = True








from __future__ import annotations

import enum
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Remove duplicate imports and fix structure

class SimpleColumn(BaseModel):
    """Simplified column information for X/Y selection"""
    column_name: str = Field(..., description="Column name")
    is_numeric: bool = Field(..., description="Suitable for Y variable (target)")

class CombinationInfo(BaseModel):
    """Information about available data combinations"""
    combination: Dict[str, str] = Field(..., description="Filter combination values")
    file_key: str = Field(..., description="File location")
    record_count: int = Field(..., description="Number of records")

class SimpleScopeInfo(BaseModel):
    """Simplified scope information for variable selection"""
    scope_id: str
    scope_name: str
    scope_type: str
    description: Optional[str]
    total_records: int
    available_columns: List[SimpleColumn] = Field(..., description="Columns available for X/Y selection")
    available_combinations: List[CombinationInfo] = Field(..., description="Data combinations to choose from")
    created_at: str

class SimpleScopeResponse(BaseModel):
    """Simplified response for scope information"""
    success: bool
    scope: Optional[SimpleScopeInfo]
    retrieved_at: str
    message: str

class BasicModelType(str, Enum):
    """Basic regression model types"""
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    BAYESIAN_RIDGE = "bayesian_ridge"

class TransformationSpec(BaseModel):
    """Specification for a single transformation step"""
    name: str = Field(..., description="Unique name for this transformation step")
    method: str = Field(..., pattern="^(standard|minmax|robust)$", description="Transformer method")  # FIXED: pattern instead of regex
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for transformer")
    columns: List[str] = Field(..., description="List of columns to apply this transformation to")

class BasicTrainingRequest(BaseModel):
    """Training request using your StandardResults format"""
    scope_id: str = Field(..., description="Scope ID containing the data")
    combination_file_key: str = Field(..., description="Specific combination file to use")
    
    # Variable selection
    x_variables: List[str] = Field(..., description="Feature columns (X variables)")
    y_variable: str = Field(..., description="Target column (Y variable)")
    
    # Model selection
    model_type: BasicModelType = Field(..., description="Type of regression model")
    
    # Basic training parameters
    test_size: float = Field(default=0.2, description="Test set proportion")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    
    # Model-specific parameters (optional)
    alpha: Optional[float] = Field(None, description="Regularization strength for Ridge/Lasso/ElasticNet")
    l1_ratio: Optional[float] = Field(None, description="L1 ratio for ElasticNet")
    max_iter: Optional[int] = Field(None, description="Maximum iterations for Lasso/ElasticNet/BayesianRidge")
    
    # New: transformation pipeline specification
    transformations: Optional[List[TransformationSpec]] = Field(
        default=None,
        description="Optional list of transformations to apply before training"
    )
    
    # Metadata
    training_name: Optional[str] = Field(None, description="Custom name for training run")

class BasicTrainingResponse(BaseModel):
    """Enhanced response including coefficients, intercept, and applied transformations"""
    success: bool
    training_id: str
    training_name: str
    scope_id: str
    combination_used: str
    
    # Training setup
    x_variables: List[str]
    y_variable: str
    model_type: str

    # Performance metrics
    model_name: str
    train_r2: float
    test_r2: Optional[float] = None
    train_rmse: float
    test_rmse: Optional[float] = None
    train_mae: float
    test_mae: Optional[float] = None
    train_mape: Optional[float] = None
    test_mape: Optional[float] = None
    train_samples: int
    test_samples: int
    aic: Optional[float] = None
    bic: Optional[float] = None
    
    # Coefficient information
    intercept: Optional[float] = Field(default=None, description="Model intercept value")
    coefficients: Optional[Dict[str, float]] = Field(default=None, description="Feature coefficients")

    # Transformations
    transformations_applied: Optional[List[TransformationSpec]] = Field(
        default=None,
        description="List of transformations applied during training"
    )
    
    # Metadata
    training_duration_seconds: float = Field(default=0.0, description="Training duration in seconds")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Training creation timestamp")

# Constraint models
class ConstraintType(str, enum.Enum):
    """Supported coefficient constraints."""
    NON_NEGATIVE = "non_negative"
    NON_POSITIVE = "non_positive"
    BOX = "box"
    EXACT = "exact"
    CUSTOM = "custom"

class ConstraintModelType(str, enum.Enum):
    CONSTRAINED_RIDGE = "constrained_ridge"
    CONSTRAINED_LINEAR = "constrained_linear"

class ConstraintDefinition(BaseModel):
    """One logical constraint applied to one or more coefficients."""
    constraint_type: ConstraintType = Field(..., alias="type")
    features: List[str] = Field(..., min_length=1)  # FIXED: min_items -> min_length in Pydantic v2

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    exact_value: Optional[float] = None
    custom_expression: Optional[str] = None

    @model_validator(mode='after')  # FIXED: root_validator -> model_validator
    def validate_parameters(self):
        c_type = self.constraint_type
        if c_type == ConstraintType.BOX:
            if self.min_value is None and self.max_value is None:
                raise ValueError("BOX constraint requires min_value and/or max_value")
        elif c_type == ConstraintType.EXACT:
            if self.exact_value is None:
                raise ValueError("EXACT constraint requires exact_value")
        elif c_type == ConstraintType.CUSTOM:
            if not self.custom_expression:
                raise ValueError("CUSTOM constraint requires custom_expression")
        return self

    model_config = {  # FIXED: Config -> model_config
        "populate_by_name": True,  # FIXED: allow_population_by_field_name -> populate_by_name
        "json_schema_extra": {
            "example": {
                "type": "non_negative",
                "features": ["price", "discount"],
            }
        }
    }

class ConstraintValidationResult(BaseModel):
    """Result for each constraint·feature pair after training."""
    constraint_satisfied: bool
    violation_magnitude: float
    affected_features: List[str]

class ConstraintTrainingRequest(BaseModel):
    """Request body for `/train-constraint`."""
    scope_id: str
    combination_file_key: str
    training_name: Optional[str] = None
    model_type: ConstraintModelType = ConstraintModelType.CONSTRAINED_RIDGE
    x_variables: List[str] = Field(..., min_length=1)  # FIXED: min_items -> min_length
    y_variable: str
    alpha: float = 1.0
    learning_rate: float = 0.001
    max_iter: int = 100_000
    tolerance: float = 1e-6
    test_size: float = 0.2
    random_state: int = 42
    fit_intercept: bool = True
    constraints: List[ConstraintDefinition] = Field(default_factory=list)
    transformations: Optional[List[TransformationSpec]] = None

    @field_validator("y_variable")  # FIXED: validator -> field_validator
    @classmethod
    def ensure_y_not_in_x(cls, v, info):  # FIXED: Added info parameter
        if info.data and "x_variables" in info.data and v in info.data["x_variables"]:
            raise ValueError("y_variable cannot also appear in x_variables")
        return v

class ConstraintTrainingResponse(BaseModel):
    """Response model for `/train-constraint`."""
    success: bool
    training_id: str
    training_name: str
    scope_id: str
    combination_used: str
    x_variables: List[str]
    y_variable: str
    model_type: str
    constraints_applied: List[ConstraintDefinition]
    transformations_applied: List[TransformationSpec]
    model_name: str
    train_r2: Optional[float] = None
    test_r2: Optional[float] = None
    train_rmse: Optional[float] = None
    test_rmse: Optional[float] = None
    train_mae: Optional[float] = None
    test_mae: Optional[float] = None
    train_mape: Optional[float] = None
    test_mape: Optional[float] = None
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    intercept: Optional[float] = None
    coefficients: Dict[str, float]
    constraint_validation: List[ConstraintValidationResult]
    optimization_converged: bool
    optimization_iterations: int
    training_duration_seconds: float
    created_at: str

    model_config = {"from_attributes": True}  # FIXED: orm_mode -> from_attributes

__all__ = [
    "ConstraintType",
    "ConstraintModelType", 
    "ConstraintDefinition",
    "ConstraintValidationResult",
    "TransformationSpec",
    "ConstraintTrainingRequest",
    "ConstraintTrainingResponse",
    "BasicModelType",
    "BasicTrainingRequest", 
    "BasicTrainingResponse",
]
