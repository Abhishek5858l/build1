# config.py - Build Atom Configuration (CORRECTED)
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """Build Atom API Settings - Simple and Clean"""
    
    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    app_name: str = "Build Atom API"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # =============================================================================
    # API SETTINGS
    # =============================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8004
    api_reload: bool = True
    
    # =============================================================================
    # MONGODB SETTINGS
    # =============================================================================
    mongo_uri: str = "mongodb://admin_dev:pass_dev@10.2.1.65:9005/?authSource=admin"
    
    # Source: Scope data access (read-only)
    scope_database: str = "Scope_selection"
    scope_collection: str = "Scopes"
    
    # Build: Build Atom storage (read-write)
    build_database: str = "build"
    build_logs_collection: str = "build_logs"
    
    # =============================================================================
    # MINIO SETTINGS
    # =============================================================================
    minio_endpoint: str = "10.2.1.65:9003"
    minio_access_key: str = "admin_dev"
    minio_secret_key: str = "pass_dev"
    minio_use_ssl: bool = False
    
    # Source: Scope data bucket (read-only)
    scope_bucket: str = "dataformodel"
    
    # Build: Build models bucket (read-write)
    build_bucket: str = "buildmodels"
    
    # =============================================================================
    # REDIS SETTINGS
    # =============================================================================
    redis_host: str = "10.2.1.65"
    redis_port: int = 9002
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_decode_responses: bool = True
    
    # =============================================================================
    # CORS SETTINGS
    # =============================================================================
    cors_origins: list[str] = ["*"]
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]
    cors_credentials: bool = True
    
    class Config:
        env_prefix = "BUILD_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Global settings instance
settings = get_settings()
