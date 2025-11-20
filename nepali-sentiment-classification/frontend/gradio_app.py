import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List

import GPUtil
import gradio as gr
import numpy as np
import pandas as pd
import psutil
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8001")


def get_system_metrics():
    """Get current system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        gpu_info = "Not available"
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_info = f"{gpu.name}: {gpu.load * 100:.1f}% load, {gpu.memoryUsed}/{gpu.memoryTotal}MB, {gpu.temperature}°C"

        return {
            "cpu": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used": f"{memory.used / (1024**3):.1f}",
            "memory_total": f"{memory.total / (1024**3):.1f}",
            "gpu": gpu_info,
        }
    except Exception as e:
        return {"error": str(e)}


def get_detailed_system_status():
    """Get detailed system status"""
    try:
        response = requests.get(f"{API_URL}/system-status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            metrics = get_system_metrics()

            # Format average inference time
            performance = status.get("performance_metrics", {})
            avg_inference = performance.get("avg_inference_time", 0)
            if isinstance(avg_inference, (int, float)):
                avg_inference = f"{avg_inference:.2f} ms"
            else:
                avg_inference = "N/A"

            model_status = status.get("model_status", {})
            labels = model_status.get("labels", {})
            labels_str = ", ".join(labels.values()) if labels else "N/A"

            status_text = f"""
## 🖥️ System Status

### 🔌 API & Model
- **Status:** ✅ Online
- **Model:** ✅ Loaded
- **Model Name:** {model_status.get("model_name", "N/A")}
- **Device:** {model_status.get("device", "N/A")}
- **Total Predictions:** {performance.get("total_predictions", 0)}
- **Avg Inference Time:** {avg_inference}

### 💻 System Resources
- **CPU Usage:** {metrics["cpu"]}%
- **Memory Usage:** {metrics["memory_percent"]}% ({metrics["memory_used"]}/{metrics["memory_total"]} GB)
- **GPU:** {metrics["gpu"]}

### 🏷️ Available Labels
- {labels_str}

### 📊 Performance Metrics
- **Success Rate:** {performance.get("success_rate", "N/A")}
- **Requests Per Second:** {performance.get("requests_per_second", "N/A")}
"""
            return status_text
        else:
            return "**API Status:** ❌ Offline"
    except Exception as e:
        return f"**API Status:** ❌ Error - {str(e)}"


def predict_single_text(text: str):
    """Send text to BERT API for classification"""
    try:
        # Get system metrics before prediction
        metrics_before = get_system_metrics()

        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        request_time = (time.time() - start_time) * 1000

        # Get system metrics after prediction
        metrics_after = get_system_metrics()

        if response.status_code == 200:
            result = response.json()
            top_pred_label = result["label"]
            top_pred_confidence = result["confidence"]
            probabilities = result["probabilities"]

            # Create formatted output with probabilities
            output = f"""

| Metric | Value |
|--------|-------|
| **Predicted Label** | {top_pred_label} |
| **Confidence** | {top_pred_confidence:.4f} |
| **Inference Time** | {result["inference_time_ms"]:.2f} ms |
| **Total Request Time** | {request_time:.2f} ms |

### System Usage During Inference:
- **CPU:** {metrics_before["cpu"]}% → {metrics_after["cpu"]}%
- **Memory:** {metrics_before["memory_percent"]}%
- **GPU:** {metrics_after["gpu"]}

### All Class Probabilities:
"""

            # Sort probabilities by confidence
            sorted_probs = sorted(
                probabilities.items(), key=lambda x: x[1], reverse=True
            )

            for label, confidence in sorted_probs:
                confidence_percent = confidence * 100
                bar_length = int(confidence_percent / 10)
                bar = "█" * bar_length + "░" * (10 - bar_length)
                output += f"**{label}**: {confidence:.4f} |{bar}| {confidence_percent:.1f}%  \n"

            return output, top_pred_label, top_pred_confidence
        else:
            error_msg = (
                f"**❌ API Error:** {response.status_code} - {response.text[:200]}"
            )
            return error_msg, "Error", 0.0

    except requests.exceptions.Timeout:
        return "**❌ Request Timeout:** The API took too long to respond.", "Error", 0.0
    except requests.exceptions.ConnectionError:
        return "**❌ Connection Error:** Cannot connect to the API.", "Error", 0.0
    except Exception as e:
        return f"**❌ Error:** {str(e)}", "Error", 0.0


def evaluate_model(dataset_file):
    """Evaluate model on uploaded dataset file"""
    try:
        if dataset_file is None:
            return "**❌ Error:** Please upload a dataset file.", None

        # Save uploaded file to temporary location
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(dataset_file.name))

        # Copy the uploaded file to temp location
        shutil.copy2(dataset_file.name, temp_path)

        start_time = time.time()

        # Call evaluation endpoint with the file path
        response = requests.post(
            f"{API_URL}/evaluate",
            json={"dataset_path": temp_path},
            timeout=300,  # 5 minutes timeout for evaluation
        )

        eval_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()

            # Create formatted output with evaluation results
            output = f"""

- **Dataset:** {os.path.basename(dataset_file.name)}
- **Evaluation Time:** {eval_time:.2f} seconds

### Performance Metrics:
| Metric | Value |
|--------|-------|
| **Accuracy** | {result["eval_accuracy"]:.4f} |
| **Precision** | {result["eval_precision"]:.4f} |
| **Recall** | {result["eval_recall"]:.4f} |
| **F1-Score** | {result["eval_f1_score"]:.4f} |
| **Loss** | {result["eval_loss"]:.4f} |

### Performance Statistics:
| Metric | Value |
|--------|-------|
| **Runtime** | {result["eval_runtime"]:.4f} seconds |
| **Samples/Second** | {result["eval_samples_per_second"]:.3f} |
| **Steps/Second** | {result["eval_steps_per_second"]:.3f} |

### Interpretation:
- **Accuracy**: The model correctly classifies {result["eval_accuracy"] * 100:.1f}% of samples
- **F1-Score**: Balanced measure of precision and recall: {result["eval_f1_score"]:.4f}
- **Throughput**: The model processes {result["eval_samples_per_second"]:.0f} samples per second
"""

            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

            return output, result
        else:
            error_msg = f"**❌ Evaluation Error:** {response.status_code} - {response.text[:200]}"
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            return error_msg, None

    except Exception as e:
        return f"**❌ Evaluation Error:** {str(e)}", None


def predict_batch_texts(texts: str):
    """Send batch of texts to BERT API for classification"""
    try:
        if not texts or not texts.strip():
            return "**❌ Error:** No texts provided for batch prediction.", None

        # Split input by newlines to get individual texts
        text_list = [text.strip() for text in texts.split("\n") if text.strip()]
        if not text_list:
            return "**❌ Error:** No valid texts found in input.", None

        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict_batch", json={"texts": text_list}, timeout=60
        )
        total_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            results = result["results"]

            # Create DataFrame for results
            df_data = []
            for i, res in enumerate(results):
                df_data.append(
                    {
                        "Text": text_list[i][:50] + "..."
                        if len(text_list[i]) > 50
                        else text_list[i],
                        "Label": res["label"],
                        "Confidence": f"{res['confidence']:.4f}",
                        "Inference Time (ms)": f"{res['inference_time_ms']:.2f}",
                    }
                )

            df = pd.DataFrame(df_data)

            # Create summary statistics
            avg_confidence = np.mean([float(r["confidence"]) for r in results])
            avg_inference_time = np.mean([r["inference_time_ms"] for r in results])

            output = f"""

- **Total Texts:** {len(text_list)}
- **Total Time:** {total_time:.2f} ms
- **Average Inference Time:** {avg_inference_time:.2f} ms per text
- **Average Confidence:** {avg_confidence:.4f}

### Results Summary:
- **Most Common Label:** {max(set([r["label"] for r in results]), key=[r["label"] for r in results].count)}
- **Highest Confidence:** {max([r["confidence"] for r in results]):.4f}
- **Lowest Confidence:** {min([r["confidence"] for r in results]):.4f}
"""

            return output, df
        else:
            error_msg = (
                f"**❌ API Error:** {response.status_code} - {response.text[:200]}"
            )
            return error_msg, None

    except Exception as e:
        return f"**❌ Batch Prediction Error:** {str(e)}", None


# Create Gradio interface
with gr.Blocks(
    title="NepaliBERT Sentiment Classification", theme=gr.themes.Ocean()
) as demo:
    gr.Markdown("""
    # 🇳🇵 NepaliBERT Sentiment Classification

    Classify Nepali text using a fine-tuned NepaliBERT model. Choose between text prediction, batch prediction, or model evaluation.
    """)

    with gr.Tabs():
        with gr.Tab("Text Prediction"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📝 Input Text")
                    text_input = gr.Textbox(
                        label="Enter Nepali text to classify",
                        placeholder="Type your Nepali text here...",
                        lines=5,
                        max_lines=10,
                    )
                    predict_btn = gr.Button(
                        "🔍 Classify Text", variant="primary", size="lg"
                    )

                    with gr.Accordion("System Status", open=True):
                        status_btn = gr.Button("🔄 Refresh Status")
                        status_display = gr.Markdown()

                with gr.Column():
                    with gr.Row():
                        label_output = gr.Textbox(
                            label="🎯 Predicted Label", interactive=False
                        )
                        confidence_output = gr.Number(
                            label="📈 Confidence Score", interactive=False, precision=4
                        )

                    with gr.Accordion("📋 Classification Results", open=True):
                        result_output = gr.Markdown(
                            " Enter text and click 'Classify Text' to get predictions."
                        )

        with gr.Tab("Model Evaluation"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📊 Model Evaluation")
                    dataset_upload = gr.File(
                        label="Upload Evaluation Dataset",
                        file_types=[".csv", ".json", ".txt"],
                        file_count="single",
                    )
                    evaluate_btn = gr.Button(
                        "🎯 Evaluate Model", variant="primary", size="lg"
                    )

                    gr.Markdown("### ⚠️ Note:")
                    gr.Markdown(
                        "Upload your evaluation dataset file (CSV, JSON, or TXT format). The system will evaluate the model and show comprehensive performance metrics including accuracy, precision, recall, and F1-score."
                    )

                    with gr.Accordion("System Status", open=True):
                        eval_status_btn = gr.Button("🔄 Refresh Status")
                        eval_status_display = gr.Markdown()

                with gr.Column():
                    with gr.Accordion("📈 Evaluation Results", open=True):
                        eval_result_output = gr.Markdown(
                            " Upload dataset and click 'Evaluate Model' to run evaluation."
                        )

                    with gr.Accordion("Detailed Metrics", open=False):
                        eval_metrics_display = gr.JSON(label="Raw Metrics")

        with gr.Tab("Batch Prediction"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📦 Batch Text Prediction")
                    batch_text_input = gr.Textbox(
                        label="Enter Nepali texts (one per line)",
                        placeholder="Text 1\nText 2\nText 3...",
                        lines=10,
                        max_lines=20,
                    )
                    batch_predict_btn = gr.Button(
                        "📦 Predict Batch", variant="primary", size="lg"
                    )

                    gr.Markdown("### 💡 Instructions:")
                    gr.Markdown(
                        "Enter multiple Nepali texts, one per line. The system will process all texts and show results in a table format."
                    )

                    with gr.Accordion("System Status", open=True):
                        batch_status_btn = gr.Button("🔄 Refresh Status")
                        batch_status_display = gr.Markdown()

                with gr.Column():
                    with gr.Accordion("📋 Batch Results", open=True):
                        batch_result_output = gr.Markdown(
                            " Enter texts and click 'Predict Batch' to get results."
                        )
                        batch_results_table = gr.Dataframe(label="Prediction Results")

    # Event handlers
    predict_btn.click(
        predict_single_text,
        inputs=[text_input],
        outputs=[result_output, label_output, confidence_output],
    )

    status_btn.click(get_detailed_system_status, outputs=status_display)

    eval_status_btn.click(get_detailed_system_status, outputs=eval_status_display)

    batch_status_btn.click(get_detailed_system_status, outputs=batch_status_display)

    evaluate_btn.click(
        evaluate_model,
        inputs=[dataset_upload],
        outputs=[eval_result_output, eval_metrics_display],
    )

    batch_predict_btn.click(
        predict_batch_texts,
        inputs=[batch_text_input],
        outputs=[batch_result_output, batch_results_table],
    )

    # Load status on startup for all tabs
    demo.load(get_detailed_system_status, outputs=status_display)
    demo.load(get_detailed_system_status, outputs=eval_status_display)
    demo.load(get_detailed_system_status, outputs=batch_status_display)

if __name__ == "__main__":
    print("🇳🇵 Starting NepaliBERT Sentiment Classification Interface...")
    print(f"🔗 API URL: {API_URL}")
    print("🌐 Starting Gradio server...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("FRONTEND_PORT", "7860")),
        share=False,
    )
