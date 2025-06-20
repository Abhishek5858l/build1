
from __future__ import annotations

import time
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from minio import Minio
from pymongo import MongoClient

# internal imports:
from Models.Constraint_models import (
    CustomConstrainedRidge,
)
from Models.Transformation import TransformationPipeline
from schema import (
    ConstraintTrainingRequest,
    ConstraintTrainingResponse,
    ConstraintDefinition,
    ConstraintValidationResult,
    ConstraintType,
    ConstraintModelType,
    TransformationSpec,
)
from config import get_settings

from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pymongo import MongoClient
from minio import Minio
import pandas as pd
from io import BytesIO
from datetime import datetime
import logging
from typing import Optional, List, Dict, Any, Tuple
from fastapi import HTTPException
import pandas as pd
from io import BytesIO
import time
import uuid
from datetime import datetime
from config import get_settings
from schema import SimpleScopeResponse, SimpleColumn, SimpleScopeInfo, CombinationInfo,BasicTrainingResponse,BasicModelType,BasicTrainingRequest,ConstraintTrainingResponse,ConstraintValidationResult,ConstraintTrainingRequest, ConstraintType, ConstraintModelType,ConstraintDefinition, ConstraintTrainingRequest
from config import get_settings
from schema import (
    BasicTrainingResponse, BasicTrainingRequest, BasicModelType
)


from fastapi import APIRouter, HTTPException, Depends
from pymongo import MongoClient
from minio import Minio
import pandas as pd
from io import BytesIO
from datetime import datetime
import uuid
import time
from typing import Any, Dict, List, Optional, Tuple

from config import get_settings
from schema import (
    ConstraintTrainingRequest,
    ConstraintTrainingResponse,
    ConstraintModelType,
    ConstraintDefinition,
    ConstraintValidationResult,
    TransformationSpec,
    ConstraintType
)
from Models.Constraint_models import CustomConstrainedRidge

# Import models with transformation support
from Models.Regression import (
    EnhancedLinearRegression,
    EnhancedRidgeRegression,
    EnhancedLassoRegression,
    EnhancedElasticNetRegression,
    EnhancedBayesianRidge
)

from Models.Constraint_models import CustomConstrainedRidge



logger = logging.getLogger(__name__)
router = APIRouter()
@router.get("/{scope_id}", response_model=SimpleScopeResponse)
async def get_scope_by_id(
    scope_id: str = Path(..., description="Unique scope identifier"),
    force_refresh: bool = Query(False, description="Force refresh from source"),
    settings = Depends(get_settings)
):
    """
    Retrieve scope with intelligent caching strategy
    """
    client = None
    minio_client = None
    
    try:
        # Connect to Build MongoDB
        client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
        build_db = client[settings.build_database]  # "build"
        scope_cache_collection = build_db["scope_cache"]
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_scope = scope_cache_collection.find_one({"scope_id": scope_id})
            if cached_scope:
                # Update access tracking
                scope_cache_collection.update_one(
                    {"scope_id": scope_id},
                    {
                        "$set": {"last_accessed": datetime.now().isoformat()},
                        "$inc": {"access_count": 1}
                    }
                )
                
                return SimpleScopeResponse(
                    success=True,
                    scope=SimpleScopeInfo(**cached_scope["scope_data"]),
                    retrieved_at=datetime.now().isoformat(),
                    message="Scope retrieved from cache"
                )
        
        # Cache miss or force refresh - get from source
        scope_db = client[settings.scope_database]  # "Scope_selection"
        scopes_collection = scope_db[settings.scope_collection]  # "Scopes"
        
        scope_data = scopes_collection.find_one({"scope_id": scope_id})
        if not scope_data:
            return SimpleScopeResponse(
                success=False,
                scope=None,
                retrieved_at=datetime.now().isoformat(),
                message=f"Scope with ID '{scope_id}' not found"
            )
        
        # Process scope data (your existing logic)
        minio_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl
        )
        
        scope_info = await _process_simple_scope_data(scope_data, minio_client, settings)
        
        if scope_info:
            # Save to cache
            cache_document = {
                "scope_id": scope_id,
                "original_source": "Scope_selection",
                "cached_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 1,
                "scope_data": scope_info.dict(),
                "cache_status": "active"
            }
            
            # Upsert to cache
            scope_cache_collection.replace_one(
                {"scope_id": scope_id},
                cache_document,
                upsert=True
            )
            
            return SimpleScopeResponse(
                success=True,
                scope=scope_info,
                retrieved_at=datetime.now().isoformat(),
                message="Scope retrieved and cached successfully"
            )
        
    except Exception as e:
        logger.error(f"Error retrieving scope {scope_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scope: {str(e)}")
    finally:
        if client:
            client.close()

async def _process_simple_scope_data(scope_data: dict, minio_client: Minio, settings) -> Optional[SimpleScopeInfo]:
    """
    Simple processing to get columns and combinations only
    """
    try:
        scope_id = scope_data.get("scope_id")
        scope_type = scope_data.get("scope_type", "unknown")
        
        # Get combinations based on scope type
        combinations = []
        total_records = 0
        sample_file_key = None
        
        if scope_type == "multi_combination":
            combination_files = scope_data.get("combination_files", [])
            for cf in combination_files:
                combinations.append(CombinationInfo(
                    combination=cf.get("combination", {}),
                    file_key=cf.get("file_key", ""),
                    record_count=cf.get("record_count", 0)
                ))
            total_records = scope_data.get("total_filtered_records", 0)
            if combination_files:
                sample_file_key = combination_files[0].get("file_key")
                
        elif scope_type == "multi_filter":
            filter_set_results = scope_data.get("filter_set_results", [])
            for filter_set in filter_set_results:
                combination_files = filter_set.get("combination_files", [])
                for cf in combination_files:
                    combinations.append(CombinationInfo(
                        combination=cf.get("combination", {}),
                        file_key=cf.get("file_key", ""),
                        record_count=cf.get("record_count", 0)
                    ))
            total_records = scope_data.get("overall_filtered_records", 0)
            if filter_set_results and filter_set_results[0].get("combination_files"):
                sample_file_key = filter_set_results[0]["combination_files"][0].get("file_key")
                
        else:
            # Single filtered scope
            file_key = scope_data.get("file_key")
            if file_key:
                combinations.append(CombinationInfo(
                    combination={},
                    file_key=file_key,
                    record_count=scope_data.get("filtered_records_count", 0)
                ))
                sample_file_key = file_key
            total_records = scope_data.get("filtered_records_count", 0)
        
        if not sample_file_key:
            logger.warning(f"No sample file found for scope {scope_id}")
            return None
        
        # Get simple column information
        simple_columns = await _get_simple_columns(sample_file_key, minio_client, settings)
        
        if not simple_columns:
            logger.warning(f"Could not get columns for scope {scope_id}")
            return None
        
        return SimpleScopeInfo(
            scope_id=scope_id,
            scope_name=scope_data.get("name", scope_id),
            scope_type=scope_type,
            description=scope_data.get("description"),
            total_records=total_records,
            available_columns=simple_columns,
            available_combinations=combinations,
            created_at=scope_data.get("created_at", "")
        )
        
    except Exception as e:
        logger.error(f"Error processing simple scope data: {str(e)}")
        return None

async def _get_simple_columns(file_key: str, minio_client: Minio, settings) -> List[SimpleColumn]:
    """
    Get simple column information - just names and numeric flag
    """
    try:
        # Download file from MinIO
        response = minio_client.get_object(settings.scope_bucket, file_key)
        file_data = response.read()
        response.close()
        
        # Read CSV to get column info
        df = pd.read_csv(BytesIO(file_data))
        
        # Create simple column list
        simple_columns = []
        for column in df.columns:
            try:
                col_data = df[column]
                is_numeric = pd.api.types.is_numeric_dtype(col_data)
                
                simple_columns.append(SimpleColumn(
                    column_name=column,
                    is_numeric=is_numeric
                ))
                
            except Exception as e:
                logger.warning(f"Error analyzing column {column}: {str(e)}")
                simple_columns.append(SimpleColumn(
                    column_name=column,
                    is_numeric=False
                ))
        
        return simple_columns
        
    except Exception as e:
        logger.error(f"Error getting simple columns from {file_key}: {str(e)}")
        return []








@router.post("/train-basic", response_model=BasicTrainingResponse)
async def train_basic_regression(
    request: BasicTrainingRequest,
    settings = Depends(get_settings)
):
    """
    Train basic regression models with optional feature transformations.
    """
    client = None
    minio_client = None
    start_time = time.time()

    try:
        # Generate training ID
        training_id = f"basic_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        training_name = request.training_name or \
            f"Basic_{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Connect to MongoDB
        client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[settings.build_database]
        collection = db["basic_training"]

        # Connect to MinIO and load data
        minio_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl
        )
        df = await _load_basic_training_data(
            request.combination_file_key, minio_client, settings
        )

        # Validate presence of variables
        missing = [v for v in request.x_variables + [request.y_variable] if v not in df.columns]
        if missing:
            raise HTTPException(400, f"Variables not found: {missing}")
        if not pd.api.types.is_numeric_dtype(df[request.y_variable]):
            raise HTTPException(400, f"Y variable '{request.y_variable}' must be numeric")

        # In your train_basic_regression function, replace the model_params section:
        model_params: Dict[str, Any] = {
            "test_size": request.test_size,
            "random_state": request.random_state,
            # Convert TransformationSpec objects to the expected tuple format
            "transformations": [
                (spec.name, spec.method, spec.params, spec.columns)
                for spec in (request.transformations or [])
            ]
        }

        # Add algorithm-specific params
        if request.alpha is not None:
            model_params["alpha"] = request.alpha
        if request.l1_ratio is not None:
            model_params["l1_ratio"] = request.l1_ratio
        if request.max_iter is not None:
            model_params["max_iter"] = request.max_iter

        # Initialize appropriate model
        model: Any
        if request.model_type == BasicModelType.LINEAR:
            model = EnhancedLinearRegression(**model_params)
        elif request.model_type == BasicModelType.RIDGE:
            model = EnhancedRidgeRegression(**model_params)
        elif request.model_type == BasicModelType.LASSO:
            model = EnhancedLassoRegression(**model_params)
        elif request.model_type == BasicModelType.ELASTIC_NET:
            model = EnhancedElasticNetRegression(**model_params)
        elif request.model_type == BasicModelType.BAYESIAN_RIDGE:
            model = EnhancedBayesianRidge(**model_params)
        else:
            raise HTTPException(400, f"Unsupported model type: {request.model_type}")

        # Fit the model
        model.fit(df, request.y_variable, x_variables=request.x_variables)

        # Obtain results
        results_df = model.get_results()

        # Extract metrics and coefficients
        rec = results_df.iloc[0]
        coeffs: Dict[str, float] = {}
        for i in range(len(model.feature_names)):
            feat = rec[f"feature_name_{i}"]
            coeffs[feat] = rec[f"coef_{i}"]

        response = BasicTrainingResponse(
            success=True,
            training_id=training_id,
            training_name=training_name,
            scope_id=request.scope_id,
            combination_used=request.combination_file_key,
            x_variables=request.x_variables,
            y_variable=request.y_variable,
            model_type=request.model_type.value,
            model_name=rec["model_name"],
            train_r2=rec["r2_train"],
            test_r2=rec.get("r2_test"),
            train_rmse=rec["rmse_train"],
            test_rmse=rec.get("rmse_test"),
            train_mae=rec["mae_train"],
            test_mae=rec.get("mae_test"),
            train_mape=rec.get("mape_train"),
            test_mape=rec.get("mape_test"),
            train_samples=int(rec["n_train_samples"]),
            test_samples=int(rec.get("n_test_samples", 0)),
            aic=rec.get("aic"),
            bic=rec.get("bic"),
            intercept=rec.get("intercept"),
            coefficients=coeffs,
            transformations_applied=request.transformations or []
        )

        # Persist record
        record = response.dict()
        record.update({"created_at": datetime.now().isoformat()})
        collection.insert_one(record)

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")
    finally:
        if client:
            client.close()


async def _load_basic_training_data(file_key: str, minio_client: Minio, settings) -> pd.DataFrame:
    """Load data for basic training from MinIO"""
    try:
        # Try build bucket first, then scope bucket
        bucket = settings.build_bucket
        try:
            response = minio_client.get_object(bucket, file_key)
            file_data = response.read()
            response.close()
            logger.info(f"Loaded from build bucket: {bucket}/{file_key}")
        except Exception as e:
            logger.warning(f"Build bucket failed: {str(e)}")
            bucket = settings.scope_bucket
            response = minio_client.get_object(bucket, file_key)
            file_data = response.read()
            response.close()
            logger.info(f"Loaded from scope bucket: {bucket}/{file_key}")
        
        df = pd.read_csv(BytesIO(file_data))
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Could not load data: {str(e)}")



#####################################cosntraint models
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
async def _load_constraint_training_data(
    file_key: str,
    minio_client: Minio,
    settings,
) -> pd.DataFrame:
    """
    Download the requested CSV (by *file_key*) from either the *scope*- or
    *build*-bucket and return it as a DataFrame.
    """
    for bucket in (settings.scope_bucket, settings.build_bucket):
        try:
            obj = minio_client.get_object(bucket, file_key)
            data = obj.read()
            obj.close()
            return pd.read_csv(BytesIO(data))
        except Exception:
            # try next bucket
            continue
    raise HTTPException(
        status_code=404,
        detail=f"Could not load data from buckets for key '{file_key}'",
    )


async def _validate_constraints(
    coefficients: Dict[str, float],
    constraints: List[ConstraintDefinition],
) -> List[ConstraintValidationResult]:
    """
    Check every requested constraint against the fitted coefficients and
    produce one *ConstraintValidationResult* per (constraint × feature) combo.
    """
    results: List[ConstraintValidationResult] = []

    for c in constraints:
        for feat in c.features:
            if feat not in coefficients:
                # Column filtered-out or typo – treat as satisfied but warn later
                results.append(
                    ConstraintValidationResult(
                        constraint_satisfied=False,
                        violation_magnitude=float("nan"),
                        affected_features=[feat],
                    )
                )
                continue

            val = coefficients[feat]
            satisfied = True
            magnitude = 0.0

            if c.constraint_type == ConstraintType.NON_NEGATIVE:
                if val < 0:
                    satisfied, magnitude = False, abs(val)

            elif c.constraint_type == ConstraintType.NON_POSITIVE:
                if val > 0:
                    satisfied, magnitude = False, abs(val)

            elif c.constraint_type == ConstraintType.BOX:
                if c.min_value is not None and val < c.min_value:
                    satisfied, magnitude = False, c.min_value - val
                if c.max_value is not None and val > c.max_value:
                    satisfied, magnitude = False, val - c.max_value

            elif c.constraint_type == ConstraintType.EXACT:
                if c.exact_value is not None and abs(val - c.exact_value) > 1e-6:
                    satisfied, magnitude = False, abs(val - c.exact_value)

            # (add further constraint types here)

            results.append(
                ConstraintValidationResult(
                    constraint_satisfied=satisfied,
                    violation_magnitude=magnitude,
                    affected_features=[feat],
                )
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main route
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/train-constraint", response_model=ConstraintTrainingResponse)
async def train_constraint_regression(
    request: ConstraintTrainingRequest,
    settings=Depends(get_settings),
):
    """
    Train a constrained regression model (ridge or linear) with optional
    column-wise preprocessing.  The entire training metadata and results are
    persisted to MongoDB for lineage.
    """
    tic = time.time()
    training_id = f"constraint_{uuid.uuid4().hex[:8]}_{int(tic)}"
    training_name = request.training_name or (
        f"Constraint_{request.model_type}_{datetime.now():%Y%m%d_%H%M%S}"
    )

    # ─── external clients ────────────────────────────────────────────────────
    mongo: Optional[MongoClient] = None
    minio: Optional[Minio] = None

    try:
        # MongoDB
        mongo = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
        db = mongo[settings.build_database]
        coll = db["constraint_training"]

        # MinIO
        minio = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl,
        )

        # ─── 1. Load & basic validation of the data ──────────────────────────
        df = await _load_constraint_training_data(
            request.combination_file_key, minio, settings
        )

        # Check requested columns exist
        missing = [
            col for col in (*request.x_variables, request.y_variable) if col not in df.columns
        ]
        if missing:
            raise HTTPException(400, f"Variables not found in data: {missing}")

        if not pd.api.types.is_numeric_dtype(df[request.y_variable]):
            raise HTTPException(400, f"Target column '{request.y_variable}' must be numeric")

        # ─── 2. Translate request → model params ─────────────────────────────
        model_params: Dict[str, Any] = {
            "alpha": request.alpha,
            "learning_rate": request.learning_rate,
            "max_iter": request.max_iter,
            "tolerance": request.tolerance,
            "test_size": request.test_size,
            "random_state": request.random_state,
            "fit_intercept": request.fit_intercept,
            "verbose": False,
            "constraints": None,
            "transformations": None,
        }

        # 2a. Constraints
        if request.constraints:
            constraint_specs: List[Dict[str, Any]] = []

            for c in request.constraints:
                if c.constraint_type not in {
                    ConstraintType.NON_NEGATIVE,
                    ConstraintType.NON_POSITIVE,
                }:
                    # Only these two are implemented in CustomConstrainedRidge
                    raise HTTPException(
                        400,
                        f"Constraint type '{c.constraint_type}' "
                        "is not supported by this model",
                    )

                constraint_specs.append(
                    {
                        "type": c.constraint_type.value,  # 'non_positive' | 'non_negative'
                        "features": c.features or "all",
                    }
                )

            model_params["constraints"] = constraint_specs

        # 2b. Transformations – build guarded list so malformed specs
        #      surface as clear 400 errors instead of mysterious AttributeError
        if request.transformations:
            validated_transformations = []
            for i, t in enumerate(request.transformations):
                # Accept both raw dict (already validated by Pydantic) or model instance
                spec_dict: Dict[str, Any] = t.dict() if hasattr(t, "dict") else t  # type: ignore

                # --------- extra validation for params ----------------------
                params = spec_dict.get("params", {})
                if isinstance(params, list):
                    if len(params) == 2 and all(isinstance(x, (int, float)) for x in params):
                        # convert to dict understood by DataTransformer
                        spec_dict["params"] = {"feature_range": tuple(params)}
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Invalid transformation spec at index {i}: "
                                "'params' list must contain exactly two numeric values"
                            ),
                        )
                elif not isinstance(params, dict):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid transformation spec at index {i}: 'params' must be a dict"
                        ),
                    )

                validated_transformations.append(spec_dict)

            model_params["transformations"] = validated_transformations

        # ─── 3. Instantiate appropriate model class ──────────────────────────
        if request.model_type == ConstraintModelType.CONSTRAINED_RIDGE:
            model = CustomConstrainedRidge(**model_params)
        else:
            raise HTTPException(400, f"Unsupported model type: {request.model_type}")

        # ─── 4. Train & evaluate ─────────────────────────────────────────────
        model.fit(df, y=request.y_variable, x_variables=request.x_variables)
        results_df = model.get_results()
        rec = results_df.iloc[0].to_dict()

        # ─── 5. Build response payload ───────────────────────────────────────
        coef_map = {
            rec.get(f"feature_name_orig_{i}", rec.get(f"feature_name_{i}")):
            rec.get(f"coef_orig_{i}", rec.get(f"coef_{i}"))
            for i in range(model.n_features_)
        }

        validation = await _validate_constraints(coef_map, request.constraints)

        duration = time.time() - tic

        # Persist full training artefact (optional fields truncated here)
        coll.insert_one(
            {
                "training_id": training_id,
                "training_name": training_name,
                "scope_id": request.scope_id,
                "combination_file_key": request.combination_file_key,
                "x_variables": request.x_variables,
                "y_variable": request.y_variable,
                "model_type": request.model_type.value,
                "model_params": model_params,
                "standard_results": rec,
                "constraint_validation": [v.dict() for v in validation],
                "training_duration_seconds": duration,
                "created_at": datetime.now().isoformat(),
                "created_by": "build_atom_api",
            }
        )

        return ConstraintTrainingResponse(
            success=True,
            training_id=training_id,
            training_name=training_name,
            scope_id=request.scope_id,
            combination_used=request.combination_file_key,
            x_variables=request.x_variables,
            y_variable=request.y_variable,
            model_type=request.model_type.value,
            constraints_applied=request.constraints,
            transformations_applied=request.transformations or [],
            model_name=rec.get("model_name"),
            train_r2=rec.get("r2_train"),
            test_r2=rec.get("r2_test"),
            train_rmse=rec.get("rmse_train"),
            test_rmse=rec.get("rmse_test"),
            train_mae=rec.get("mae_train"),
            test_mae=rec.get("mae_test"),
            train_mape=rec.get("mape_train"),
            test_mape=rec.get("mape_test"),
            train_samples=rec.get("n_train_samples"),
            test_samples=rec.get("n_test_samples"),
            aic=rec.get("aic"),
            bic=rec.get("bic"),
            intercept=rec.get("intercept_orig", rec.get("intercept")),
            coefficients=coef_map,
            constraint_validation=validation,
            optimization_converged=True,
            optimization_iterations=getattr(model, "n_iter_", model.max_iter),
            training_duration_seconds=duration,
            created_at=datetime.now().isoformat(),
        )

    # ─── 6. House-keeping ────────────────────────────────────────────────────
    except HTTPException:
        raise  # propagate FastAPI error as-is
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Constraint training failed: {exc}",
        )
    finally:
        if mongo:
            mongo.close()