import logging
import time
from typing import Dict, List

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class PlantDiseaseModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.inference_times = []
        self.class_names = [
            "Apple Scab",
            "Apple Black Rot",
            "Cedar Apple Rust",
            "Apple Healthy",
            "Blueberry Healthy",
            "Cherry Powdery Mildew",
            "Cherry Healthy",
            "Corn Cercospora Leaf Spot",
            "Corn Common Rust",
            "Corn Northern Leaf Blight",
            "Corn Healthy",
            "Grape Black Rot",
            "Grape Esca",
            "Grape Leaf Blight",
            "Grape Healthy",
            "Orange Haunglongbing",
            "Peach Bacterial Spot",
            "Peach Healthy",
            "Pepper Bell Bacterial Spot",
            "Pepper Bell Healthy",
            "Potato Early Blight",
            "Potato Late Blight",
            "Potato Healthy",
            "Raspberry Healthy",
            "Soybean Healthy",
            "Squash Powdery Mildew",
            "Strawberry Leaf Scorch",
            "Strawberry Healthy",
            "Tomato Bacterial Spot",
            "Tomato Early Blight",
            "Tomato Late Blight",
            "Tomato Leaf Mold",
            "Tomato Septoria Leaf Spot",
            "Tomato Spider Mites",
            "Tomato Target Spot",
            "Tomato Yellow Leaf Curl Virus",
            "Tomato Mosaic Virus",
            "Tomato Healthy",
        ]

        # Configure GPU
        self.setup_gpu()
        self.load_model()

    def setup_gpu(self):
        """Configure GPU settings for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU configured: {len(gpus)} GPU(s) available")
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
        else:
            logger.warning("No GPU found, using CPU")

    def load_model(self):
        """Load the .keras model"""
        try:
            start_time = time.time()
            self.model = tf.keras.models.load_model(self.model_path)
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model inference"""
        try:
            # Resize image to match model input size
            if len(image.shape) == 3:
                image = tf.image.resize(image, [224, 224])
            # Add batch dimension
            image = tf.expand_dims(image, axis=0)
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def predict(self, image: np.ndarray) -> Dict:
        """Perform prediction on input image"""
        try:
            start_time = time.time()

            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Perform inference
            predictions = self.model.predict(processed_image, verbose=0)

            # Get top prediction
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx])
            class_name = self.class_names[pred_idx]

            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {"class": self.class_names[i], "confidence": float(predictions[0][i])}
                for i in top_3_indices
            ]

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Track inference time for performance monitoring
            self.inference_times.append(inference_time)
            # Keep only last 100 readings to prevent memory issues
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            logger.info(
                f"Prediction completed in {inference_time:.2f}ms: {class_name} ({confidence:.4f})"
            )

            return {
                "top_prediction": {"class": class_name, "confidence": confidence},
                "top_3_predictions": top_3_predictions,
                "inference_time_ms": inference_time,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
