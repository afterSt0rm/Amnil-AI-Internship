import os
import time
from typing import List

import numpy as np
import prometheus_client as prom
import tensorflow as tf
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram

from .logging_config import setup_logging
from .models import PlantDiseaseModel
from .schemas import HealthResponse, PredictionResponse
from .utils import get_system_metrics, load_image

load_dotenv()

# Use environment variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MODEL_PATH = os.getenv("MODEL_PATH", "models/MobileNetV2_Fine_Tuned_Model.keras")

# Setup logging
logger = setup_logging()

# Prometheus metrics
PREDICTION_COUNTER = Counter("predictions_total", "Total predictions", ["status"])
PREDICTION_DURATION = Histogram("prediction_duration_seconds", "Prediction duration")
IN_PROGRESS = Gauge("predictions_in_progress", "Number of predictions in progress")
MODEL_LOAD_STATUS = Gauge("model_loaded", "Model load status")

app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases from images",
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


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        model = PlantDiseaseModel("models/MobileNetV2_Fine_Tuned_Model.keras")
        MODEL_LOAD_STATUS.set(1)
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        MODEL_LOAD_STATUS.set(0)


@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API", "status": "healthy"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system metrics"""
    gpu_available = len(tf.config.experimental.list_physical_devices("GPU")) > 0
    model_loaded = model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        timestamp=time.time(),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return prom.generate_latest()


@app.post("/predict", response_model=PredictionResponse)
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Predict plant disease from image"""
    IN_PROGRESS.inc()
    start_time = time.time()

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image data
        image_data = await file.read()
        logger.info(f"Received image: {file.filename}, size: {len(image_data)} bytes")

        # Load and process image
        image = load_image(image_data)

        # Perform prediction
        with PREDICTION_DURATION.time():
            result = model.predict(image)

        PREDICTION_COUNTER.labels(status="success").inc()
        total_time = (time.time() - start_time) * 1000

        logger.info(
            "Prediction completed",
            extra={
                "processing_time_ms": total_time,
                "inference_time_ms": result["inference_time_ms"],
                "predicted_class": result["top_prediction"]["class"],
                "confidence": result["top_prediction"]["confidence"],
            },
        )

        return PredictionResponse(**result)

    except Exception as e:
        PREDICTION_COUNTER.labels(status="error").inc()
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        IN_PROGRESS.dec()


@app.get("/system-status")
async def system_status():
    """Get detailed system status and metrics"""
    system_metrics = get_system_metrics()

    # Add inference performance metrics if available
    performance_metrics = {}
    if model and hasattr(model, "inference_times"):
        performance_metrics = {
            "avg_inference_time": np.mean(model.inference_times)
            if model.inference_times
            else 0,
            "total_predictions": len(model.inference_times)
            if model.inference_times
            else 0,
        }

    return {
        "system_metrics": system_metrics,
        "performance_metrics": performance_metrics,
        "model_status": {
            "loaded": model is not None,
            "input_shape": model.model.input_shape if model else None,
            "output_shape": model.model.output_shape if model else None,
            "class_count": len(model.class_names) if model else 0,
        },
        "api_status": {
            "uptime": time.time() - start_time if "start_time" in globals() else 0,
            "total_requests": PREDICTION_COUNTER._value.get()
            if hasattr(PREDICTION_COUNTER, "_value")
            else 0,
        },
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        workers=1,  # Multiple workers might cause GPU memory issues
    )
