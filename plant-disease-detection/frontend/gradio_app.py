import gradio as gr
import requests
import time
from PIL import Image
import io
import os
import psutil
import GPUtil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


def get_system_metrics():
    """Get current system metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()

        # GPU usage
        gpu_info = "Not available"
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # First GPU
            gpu_info = f"{gpu.name}: {gpu.load * 100:.1f}% load, {gpu.memoryUsed}/{gpu.memoryTotal}MB, {gpu.temperature}°C"

        return {
            "cpu": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used": f"{memory.used / (1024**3):.1f}GB",
            "memory_total": f"{memory.total / (1024**3):.1f}GB",
            "gpu": gpu_info,
        }
    except Exception as e:
        return {"error": str(e)}


def predict_plant_disease(image):
    """Send image to FastAPI for prediction"""
    try:
        # Get system metrics before prediction
        metrics_before = get_system_metrics()

        # Convert image to bytes
        if isinstance(image, str):
            with open(image, "rb") as f:
                image_bytes = f.read()
        else:
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

        # Send request to API
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", files=files)
        request_time = (time.time() - start_time) * 1000

        # Get system metrics after prediction
        metrics_after = get_system_metrics()

        if response.status_code == 200:
            result = response.json()
            top_pred = result["top_prediction"]
            top_3 = result["top_3_predictions"]

            # Create results with system metrics
            output = f"""
- **Top Prediction:** {top_pred["class"]}
- **Confidence:** {top_pred["confidence"]:.4f}
- **Inference Time:** {result["inference_time_ms"]:.2f}ms
- **Total Request Time:** {request_time:.2f}ms

### System Usage During Inference:
- **CPU:** {metrics_before["cpu"]}% → {metrics_after["cpu"]}%
- **Memory:** {metrics_before["memory_percent"]}%
- **GPU:** {metrics_after["gpu"]}

### Top 3 Predictions:
"""

            for i, pred in enumerate(top_3, 1):
                output += f"{i}. **{pred['class']}**: {pred['confidence']:.4f}  \n"

            return output, top_pred["class"], top_pred["confidence"]
        else:
            return f"**Error:** {response.text}", "Error", 0.0

    except Exception as e:
        return f"**Error:** {str(e)}", "Error", 0.0


def get_detailed_status():
    """Get detailed system status"""
    try:
        response = requests.get(f"{API_URL}/system-status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            metrics = get_system_metrics()

            status_text = f"""## 📊 Detailed System Status

**🔌 API Status:** ✅ Online

**🤖 Model Status:** {"✅ Loaded" if status["model_status"]["loaded"] else "❌ Not Loaded"}

### 💻 System Resources:
- **CPU:** {metrics["cpu"]}%
- **Memory:** {metrics["memory_percent"]}% ({metrics["memory_used"]}/{metrics["memory_total"]})
- **GPU:** {metrics["gpu"]}

### 🤖 Model Info:
- **Input Shape:** {status["model_status"]["input_shape"]}
- **Output Shape:** {status["model_status"]["output_shape"]}
- **Classes:** {status["model_status"]["class_count"]}
"""
            return status_text
        else:
            return "**API Status:** ❌ Offline"
    except:
        return "**API Status:** ❌ Offline"


# Create simple Gradio interface
with gr.Blocks(title="Plant Disease Detection", theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# 🌱 Plant Disease Detection System")
    gr.Markdown("Upload an image of a plant leaf to detect potential diseases.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Plant Image", type="pil", sources=["upload", "clipboard"]
            )
            predict_btn = gr.Button("Detect Disease", variant="primary")

            with gr.Accordion("System Status", open=True):
                status_btn = gr.Button("Refresh Status")
                status_display = gr.Markdown()

        with gr.Column():
            class_output = gr.Label(label="Predicted Class")
            confidence_output = gr.Number(label="Confidence Score", precision=4)
            with gr.Accordion("Prediction Results", open=True):
                result_output = gr.Markdown("Upload an image to get started.")

    # Event handlers
    predict_btn.click(
        predict_plant_disease,
        inputs=[image_input],
        outputs=[result_output, class_output, confidence_output],
    )

    status_btn.click(get_detailed_status, outputs=status_display)

    # Load status on startup
    demo.load(get_detailed_status, outputs=status_display)

if __name__ == "__main__":
    print("🌱 Starting Plant Disease Detection Interface...")
    print(f"🔗 API URL: {API_URL}")
    print("🌐 Starting Gradio server...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("FRONTEND_PORT", "7860")),
        share=False,
        show_error=True,
    )
