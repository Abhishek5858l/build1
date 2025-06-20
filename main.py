# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routes import router as model_router

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Training API",
    description="Standardized ML Model Training and Evaluation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(model_router, prefix="/build", tags=["ML Models"])

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "ML Model Training API", 
        "status": "active",
        "version": "1.0.0",
        "available_models": [
            "linear", "ridge", "lasso", "elastic_net",
            "constrained_linear", "constrained_ridge", 
            "mixed_effects", "stacked_regression"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-model-api"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8004,  # Using your specified port
        reload=True,
        log_level="info"
    )
