from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    probabilities: Dict[str, float]
    label: str
    confidence: float
    inference_time_ms: float
    timestamp: float


class BatchPredictionRequest(BaseModel):
    texts: List[str]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    avg_inference_time_ms: float
    total_time_ms: float
    timestamp: float


class EvaluationRequest(BaseModel):
    dataset_path: str  # Path to evaluation dataset


class EvaluationResponse(BaseModel):
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    eval_precision: Optional[float] = None
    eval_recall: Optional[float] = None
    eval_f1_score: Optional[float] = None
    eval_runtime: float
    eval_samples_per_second: float
    eval_steps_per_second: float
    eval_avg_confidence: Optional[float] = None
    eval_label_distribution: Optional[Dict[str, int]] = None
    timestamp: float


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool
    model_name: str
    timestamp: float
