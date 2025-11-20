import os
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram

from .logging_config import setup_logging
from .models import BERTModel
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from .utils import get_system_metrics

# Setup logging
logger = setup_logging()

# Prometheus metrics
PREDICTION_COUNTER = Counter("predictions_total", "Total predictions", ["status"])
BATCH_PREDICTION_COUNTER = Counter(
    "batch_predictions_total", "Total batch predictions", ["status"]
)
EVALUATION_COUNTER = Counter("evaluations_total", "Total evaluations", ["status"])
PREDICTION_DURATION = Histogram("prediction_duration_seconds", "Prediction duration")
BATCH_PREDICTION_DURATION = Histogram(
    "batch_prediction_duration_seconds", "Batch prediction duration"
)
EVALUATION_DURATION = Histogram("evaluation_duration_seconds", "Evaluation duration")
IN_PROGRESS = Gauge("predictions_in_progress", "Number of predictions in progress")
MODEL_LOAD_STATUS = Gauge("model_loaded", "Model load status")

app = FastAPI(
    title="NepaliBERT Sentiment Classification API",
    description="API for Nepali sentiment classification using fine-tuned NepaliBERT model with ONNX optimization",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
MODEL_NAME = os.getenv("MODEL_NAME", "aashish-mahato/nepalibert-sentiment-classifier")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        model = BERTModel(MODEL_NAME)
        MODEL_LOAD_STATUS.set(1)
        logger.info(
            f"NepaliBERT API startup completed successfully with model: {MODEL_NAME}"
        )
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        MODEL_LOAD_STATUS.set(0)


@app.get("/")
async def root():
    return {
        "message": "NepaliBERT Sentiment Classification API",
        "status": "healthy",
        "model": MODEL_NAME,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available() if model else False
    model_loaded = model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        model_name=MODEL_NAME,
        timestamp=time.time(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict text classification"""
    IN_PROGRESS.inc()

    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        logger.info(f"Received prediction request: {request.text[:100]}...")

        # Perform prediction
        with PREDICTION_DURATION.time():
            result = model.predict(request.text)

        PREDICTION_COUNTER.labels(status="success").inc()
        logger.info(f"Prediction successful for text: {request.text[:50]}...")

        return PredictionResponse(**result)

    except Exception as e:
        PREDICTION_COUNTER.labels(status="error").inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        IN_PROGRESS.dec()


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict batch of texts"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    try:
        logger.info(f"Received batch prediction request for {len(request.texts)} texts")

        with BATCH_PREDICTION_DURATION.time():
            result = model.predict_batch(request.texts)

        BATCH_PREDICTION_COUNTER.labels(status="success").inc()
        return BatchPredictionResponse(**result)

    except Exception as e:
        BATCH_PREDICTION_COUNTER.labels(status="error").inc()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest):
    """Evaluate model on a dataset using Hugging Face's built-in methods"""
    try:
        logger.info(f"Received evaluation request for dataset: {request.dataset_path}")

        # Validate file exists
        if not os.path.exists(request.dataset_path):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset file not found: {request.dataset_path}",
            )

        # Check file size
        file_size = os.path.getsize(request.dataset_path)
        max_size = 100 * 1024 * 1024  # 100MB limit
        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_size // (1024 * 1024)}MB",
            )

        with EVALUATION_DURATION.time():
            result = model.evaluate_model(request.dataset_path)

        EVALUATION_COUNTER.labels(status="success").inc()
        logger.info(f"Evaluation completed successfully: {result}")

        return EvaluationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        EVALUATION_COUNTER.labels(status="error").inc()
        logger.error(f"Evaluation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/system-status")
async def system_status():
    """Get detailed system status and metrics"""
    system_metrics = get_system_metrics()

    # Add performance metrics
    performance_metrics = {}
    if model and hasattr(model, "inference_times") and model.inference_times:
        import numpy as np

        performance_metrics = {
            "avg_inference_time": np.mean(model.inference_times),
            "total_predictions": len(model.inference_times),
        }

    return {
        "system_metrics": system_metrics,
        "performance_metrics": performance_metrics,
        "model_status": {
            "loaded": model is not None,
            "device": "GPU" if torch.cuda.is_available() else "CPU",
            "model_name": MODEL_NAME,
            "labels": model.id2label if model else None,
        },
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
