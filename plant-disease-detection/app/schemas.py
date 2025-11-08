from pydantic import BaseModel
from typing import List, Dict, Any


class PredictionResponse(BaseModel):
    top_prediction: Dict[str, Any]
    top_3_predictions: List[Dict[str, Any]]
    inference_time_ms: float
    timestamp: float


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool
    timestamp: float
