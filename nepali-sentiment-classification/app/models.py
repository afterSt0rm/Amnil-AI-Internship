import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from optimum.onnxruntime import ORTModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class BERTModel:
    def __init__(
        self, model_name: str = "aashish-mahato/nepalibert-sentiment-classifier"
    ):
        self.model_name = model_name
        self.pipeline = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self.inference_times = []

        self.id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}

        self.setup_device()
        self.load_model()

    def setup_device(self):
        """Setup GPU if available"""
        if torch.cuda.is_available():
            self.device = 0  # pipeline uses device=0 for GPU
            logger.info(f"GPU configured: {torch.cuda.device_count()} GPU(s) available")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            self.device = -1  # pipeline uses device=-1 for CPU
            logger.warning("No GPU found, using CPU")

    def load_model(self):
        """Load the fine-tuned BERT model using pipeline with ONNX optimization"""
        try:
            start_time = time.time()

            # Load ONNX model for better performance
            self.model = ORTModelForSequenceClassification.from_pretrained(
                self.model_name,
                provider="CUDAExecutionProvider"
                if torch.cuda.is_available()
                else "CPUExecutionProvider",
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Create pipeline with ONNX model
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                return_all_scores=True,
                framework="pt",
            )

            load_time = time.time() - start_time
            logger.info(f"BERT model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Model device: {'GPU' if self.device >= 0 else 'CPU'}")
            logger.info(f"Available labels: {self.id2label}")
            logger.info(f"Model name: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            logger.error(f"Falling back to non-ONNX pipeline...")
            try:
                # Fallback to regular pipeline if ONNX fails
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    return_all_scores=True,
                )
                logger.info("Successfully loaded model using regular pipeline")
            except Exception as fallback_e:
                logger.error(f"Error loading model with fallback: {fallback_e}")
                raise

    def predict(self, text: str) -> Dict[str, Any]:
        """Perform prediction on input text using pipeline"""
        try:
            start_time = time.time()

            # Get predictions from pipeline
            outputs = self.pipeline(text)

            # Sort by score in descending order
            sorted_outputs = sorted(outputs[0], key=lambda x: x["score"], reverse=True)

            # Create probabilities dictionary
            probabilities = {item["label"]: item["score"] for item in sorted_outputs}

            # Get top prediction
            top_prediction = sorted_outputs[0]
            label = top_prediction["label"]
            confidence = top_prediction["score"]

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Track inference time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            logger.info(
                f"Prediction completed in {inference_time:.2f}ms: {label} ({confidence:.4f})"
            )

            return {
                "probabilities": probabilities,
                "label": label,
                "confidence": confidence,
                "inference_time_ms": inference_time,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Perform batch prediction on multiple texts"""
        try:
            start_time = time.time()
            results = []

            for text in texts:
                result = self.predict(text)
                results.append(result)

            total_time = (time.time() - start_time) * 1000
            avg_inference_time = total_time / len(texts) if texts else 0

            logger.info(
                f"Batch prediction completed for {len(texts)} texts in {total_time:.2f}ms"
            )

            return {
                "results": results,
                "avg_inference_time_ms": avg_inference_time,
                "total_time_ms": total_time,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
            raise

    def _preprocess_dataset(
        self, dataset_path: str
    ) -> tuple[List[str], List[int], pd.DataFrame]:
        """Load and preprocess dataset for evaluation"""
        try:
            file_extension = os.path.splitext(dataset_path)[1].lower()

            # Load dataset based on file type
            if file_extension == ".csv":
                df = pd.read_csv(dataset_path)
            elif file_extension == ".json":
                df = pd.read_json(dataset_path)
            elif file_extension in [".txt", ".tsv"]:
                df = pd.read_csv(dataset_path, sep="\t")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")

            # Validate required columns
            if "text" not in df.columns:
                raise ValueError("Dataset must contain a 'text' column")

            texts = df["text"].tolist()

            # Convert labels to integers if label column exists
            labels = None
            if "label" in df.columns:
                labels = (
                    df["label"]
                    .apply(lambda x: self._map_label_to_id(str(x).strip()))
                    .tolist()
                )
                logger.info(
                    f"Converted labels to integer format using mapping: {self.label2id}"
                )
            elif any(
                col in df.columns for col in ["labels", "sentiment", "target", "class"]
            ):
                # Try to find label column
                label_col = next(
                    col
                    for col in ["labels", "sentiment", "target", "class"]
                    if col in df.columns
                )
                labels = (
                    df[label_col]
                    .apply(lambda x: self._map_label_to_id(str(x).strip()))
                    .tolist()
                )
                logger.info(
                    f"Found and converted label column '{label_col}' to integer format"
                )

            return texts, labels, df

        except Exception as e:
            logger.error(f"Error preprocessing dataset: {e}")
            raise

    def _map_label_to_id(self, label_str: str) -> int:
        """Map string label to integer ID using our fixed mapping"""
        label_clean = label_str.strip().capitalize()

        if label_clean in self.label2id:
            return self.label2id[label_clean]

        # Try to find closest match
        for standard_label, label_id in self.label2id.items():
            if (
                label_clean.lower() in standard_label.lower()
                or standard_label.lower() in label_clean.lower()
            ):
                return label_id

        # Fallback logic
        label_lower = label_clean.lower()
        if any(
            word in label_lower for word in ["neg", "bad", "poor", "terrible", "worst"]
        ):
            return 0  # Negative
        elif any(
            word in label_lower
            for word in ["pos", "good", "excellent", "great", "best"]
        ):
            return 2  # Positive
        else:
            return 1  # Neutral

    def evaluate_model(self, dataset_path: str) -> Dict[str, Any]:
        """Evaluate model using pipeline approach"""
        try:
            start_time = time.time()
            logger.info(f"Starting evaluation on dataset: {dataset_path}")

            # 1. Load and preprocess dataset
            texts, true_labels, df = self._preprocess_dataset(dataset_path)

            # 2. Get predictions using pipeline
            logger.info(f"Running predictions on {len(texts)} samples...")
            start_pred_time = time.time()

            # Batch prediction with progress logging
            batch_size = 32
            all_predictions = []
            all_confidences = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_preds = self.pipeline(batch_texts)

                for pred_list in batch_preds:
                    # Sort predictions by score
                    sorted_preds = sorted(
                        pred_list, key=lambda x: x["score"], reverse=True
                    )
                    top_pred = sorted_preds[0]

                    # Map predicted label to our standard format
                    predicted_label = self._map_label_to_standard(top_pred["label"])
                    all_predictions.append(predicted_label)
                    all_confidences.append(top_pred["score"])

                if (i + batch_size) % (batch_size * 10) == 0:  # Log every 10 batches
                    logger.info(
                        f"Processed {min(i + batch_size, len(texts))}/{len(texts)} samples"
                    )

            pred_time = time.time() - start_pred_time
            logger.info(f"Prediction completed in {pred_time:.2f} seconds")

            # 3. Convert predictions to integer IDs
            predicted_labels_int = [self.label2id[label] for label in all_predictions]

            # 4. Calculate metrics if true labels are available
            eval_results = {}
            eval_time = time.time() - start_time

            if true_labels is not None and len(true_labels) > 0:
                # Ensure we have the same number of predictions as true labels
                min_len = min(len(true_labels), len(predicted_labels_int))
                true_labels = true_labels[:min_len]
                predicted_labels_int = predicted_labels_int[:min_len]
                all_confidences = all_confidences[:min_len]

                # Calculate metrics
                accuracy = accuracy_score(true_labels, predicted_labels_int)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, predicted_labels_int, average="macro", zero_division=0
                )

                # Calculate loss approximation
                correct_mask = np.array(true_labels) == np.array(predicted_labels_int)
                correct_confidences = np.array(all_confidences)[correct_mask]
                if len(correct_confidences) > 0:
                    avg_correct_confidence = np.mean(correct_confidences)
                    loss = -np.log(max(avg_correct_confidence, 1e-7))
                else:
                    loss = -np.log(1e-7)  # Default if no correct predictions

                eval_results = {
                    "eval_loss": float(loss),
                    "eval_accuracy": float(accuracy),
                    "eval_precision": float(precision),
                    "eval_recall": float(recall),
                    "eval_f1_score": float(f1),
                    "eval_runtime": float(eval_time),
                    "eval_samples_per_second": len(true_labels) / eval_time
                    if eval_time > 0
                    else 0,
                    "eval_steps_per_second": len(true_labels)
                    / (eval_time / (len(true_labels) // batch_size + 1))
                    if eval_time > 0
                    else 0,
                    "timestamp": time.time(),
                }

                logger.info(
                    f"Evaluation completed successfully. Accuracy: {accuracy:.4f}"
                )
            else:
                # No true labels available - return prediction statistics
                avg_confidence = np.mean(all_confidences) if all_confidences else 0
                label_distribution = {}
                for label in all_predictions:
                    label_distribution[label] = label_distribution.get(label, 0) + 1

                eval_results = {
                    "eval_loss": None,
                    "eval_accuracy": None,
                    "eval_precision": None,
                    "eval_recall": None,
                    "eval_f1_score": None,
                    "eval_runtime": float(eval_time),
                    "eval_samples_per_second": len(texts) / eval_time
                    if eval_time > 0
                    else 0,
                    "eval_steps_per_second": len(texts)
                    / (eval_time / (len(texts) // batch_size + 1))
                    if eval_time > 0
                    else 0,
                    "eval_avg_confidence": float(avg_confidence),
                    "eval_label_distribution": label_distribution,
                    "timestamp": time.time(),
                }

                logger.info(
                    f"Prediction completed for {len(texts)} samples. No true labels for evaluation."
                )

            return eval_results

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}", exc_info=True)
            raise

    def _map_label_to_standard(self, predicted_label: str) -> str:
        """Map predicted label to standard format using our fixed mapping"""
        label_clean = str(predicted_label).strip().capitalize()

        # Check if it matches our standard labels exactly
        if label_clean in self.label2id:
            return label_clean

        # Try to find closest match
        for standard_label in self.label2id.keys():
            if (
                label_clean.lower() in standard_label.lower()
                or standard_label.lower() in label_clean.lower()
            ):
                return standard_label

        # Fallback to most similar based on keywords
        label_lower = label_clean.lower()
        if any(word in label_lower for word in ["neg", "bad", "poor", "terrible"]):
            return "Negative"
        elif any(word in label_lower for word in ["pos", "good", "excellent", "great"]):
            return "Positive"
        else:
            return "Neutral"
