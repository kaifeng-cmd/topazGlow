import gradio as gr
import requests
import numpy as np
import cv2
import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import onnxruntime as ort

# Load environment variables
load_dotenv()

# Global variables
onnx_session = None
input_name = None

# Configuration
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_URL = os.environ.get("DATABRICKS_URL")
CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
HF_REPO_ID = "KaiSKX/Alzheimer_ConvNeXtCNN"
ONNX_MODEL_PATH = "./onnx/convnext_model.onnx"
LOCAL_ONNX_PATH = "./convnext_model.onnx"

# Light Coral Theme (as fallback option as css is being used)
coral_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fef9f9",    
        c100="#fadee1",  
        c200="#f5c2c7", 
        c300="#f0969a",  
        c400="#ed808b",  
        c500="#e65c6a",  
        c600="#d94452",  
        c700="#c23340",  
        c800="#a02834",  
        c900="#7d1f2a",  
        c950="#4a1115"   
    ),
    secondary_hue=gr.themes.Color(
        c50="#fef9f9", c100="#fadee1", c200="#f5c2c7", c300="#f0969a", c400="#ed808b",
        c500="#e65c6a", c600="#d94452", c700="#c23340", c800="#a02834", c900="#7d1f2a", c950="#4a1115"
    ),
    neutral_hue=gr.themes.Color(
        c50="#fafafa", c100="#f5f5f5", c200="#e5e5e5", c300="#d4d4d4", c400="#a3a3a3",
        c500="#737373", c600="#525252", c700="#404040", c800="#262626", c900="#171717", c950="#0a0a0a"
    ),
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
    text_size=gr.themes.Size(lg="16px", md="14px", sm="12px", xl="20px", xs="11px", xxl="24px", xxs="10px"),
    spacing_size=gr.themes.Size(lg="12px", md="8px", sm="4px", xl="20px", xs="2px", xxl="32px", xxs="1px"),
    radius_size=gr.themes.Size(lg="8px", md="6px", sm="4px", xl="12px", xs="2px", xxl="16px", xxs="1px")
).set(
    body_background_fill="linear-gradient(135deg, #ffffff 0%, #fef9f9 10%, #fadee1 50%, #f5c2c7 100%)",
    button_primary_background_fill="linear-gradient(135deg, #e65c6a, #ed808b)",
    button_primary_background_fill_hover="linear-gradient(135deg, #d94452, #e65c6a)"
)

# Image preprocessing & prediction
val_test_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_image(img):
    img_rgb = np.array(img)
    transformed = val_test_transforms(image=img_rgb)
    img_tensor = transformed["image"]
    return img_tensor.unsqueeze(0).numpy().astype(np.float32)

# Softmax function to convert logits to probabilities
def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)

def load_onnx_model():
    """Load ONNX model from local path or Hugging Face Hub."""
    # I used local onnx model during development,
    # but when deploy to HuggingFace Spaces I use onnx model at HuggingFace Hub,
    # becuz I'm not put the large onnx model file directly into repos although have LFS.
    global onnx_session, input_name

    if onnx_session is None:
        print("🔄 Loading ONNX model...")
        if os.path.exists(LOCAL_ONNX_PATH):
            onnx_session = ort.InferenceSession(LOCAL_ONNX_PATH)
        else:
            onnx_path = hf_hub_download(repo_id=HF_REPO_ID, filename=ONNX_MODEL_PATH)
            onnx_session = ort.InferenceSession(onnx_path)

        input_name = onnx_session.get_inputs()[0].name

    else:
        print("⚡ Using cache ONNX model.")

    return onnx_session

def predict_databricks(img_array):
    if not DATABRICKS_TOKEN:
        raise gr.Error("DATABRICKS_TOKEN not set!")
    try:
        headers = {
            'Authorization': f'Bearer {DATABRICKS_TOKEN}', 
            'Content-Type': 'application/json'
        }

        payload = {'inputs': img_array.tolist()}
        response = requests.post(DATABRICKS_URL, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()

        logits = np.array(response.json()['predictions'][0])
        probabilities = softmax(logits)
        return {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API request failed: {e}")

def predict(image):
    try:
        img_array = preprocess_image(image)

        try:
            session = load_onnx_model()
            outputs = session.run(None, {input_name: img_array})[0]
            probabilities = softmax(outputs[0])
            return {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        except Exception as e:
            print(f"ONNX inference failed, falling back to Databricks API: {e}")
            # Fallback to Databricks API (slow due to Databricks free CPU limit)
            return predict_databricks(img_array)

    except Exception as e:
        raise gr.Error(f"Prediction failed: {e}")

# Gradio Interface
with gr.Blocks(
    theme=coral_theme,
    title="AI Alzheimer Analyzer",
    css="""
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main-header {
            text-align: center;
            margin: 0.2rem 0;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 16px;
            border: 1px solid rgba(230, 92, 106, 0.1);
            box-shadow: 0 4px 6px -1px rgba(230, 92, 106, 0.15);
        }
        
        .main-title {
            font-size: 2.2rem; font-weight: 700; color: #7d1f2a; margin-bottom: 0.2rem; letter-spacing: -0.02em;
        }
        
        .main-subtitle {
            font-size: 1.1rem; color: #a02834; font-weight: 400; margin-bottom: 1rem;
        }
        
        .main-description {
            font-size: 0.8rem; color: #4a1115; line-height: 1.3; max-width: 600px; margin: 0 auto;
        }
        
        .upload-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 1.5rem;
            border: 2px dashed rgba(230, 92, 106, 0.3);
            transition: all 0.3s ease;
        }
        
        .upload-container:hover {
            border-color: rgba(230, 92, 106, 0.5);
            background: rgba(254, 249, 249, 0.95);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px -5px rgba(230, 92, 106, 0.15);
        }
        
        .results-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(230, 92, 106, 0.2);
            box-shadow: 0 2px 4px rgba(230, 92, 106, 0.05);
            /* Flexbox fix to make bars visible and fill height */
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100%;
        }
        .results-container .label-wrap { display: flex; flex-direction: column; gap: 8px; }
        .results-container .label-wrap > div > .label-name { font-size: 1rem; font-weight: 600; color: #7d1f2a; }
        .results-container .confidence-bar {
            height: 24px !important;
            border-radius: 8px;
            background-color: #fadee1;
        }
        .results-container .label-wrap > div:nth-child(1) .confidence-bar > .bar-color { background: linear-gradient(90deg, #d94452, #e65c6a); }
        .results-container .label-wrap > div:nth-child(2) .confidence-bar > .bar-color { background: linear-gradient(90deg, #e65c6a, #ed808b); }
        .results-container .label-wrap > div:nth-child(3) .confidence-bar > .bar-color { background: linear-gradient(90deg, #ed808b, #f0969a); }
        .results-container .label-wrap > div:nth-child(4) .confidence-bar > .bar-color { background: linear-gradient(90deg, #f0969a, #f5c2c7); }

        .predict-button {
            background: linear-gradient(135deg, #e65c6a, #fab9bf);
            border: none; border-radius: 8px; padding: 12px 24px; color: white; font-weight: 600;
            transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(230, 92, 106, 0.2);
        }
        
        .predict-button:hover {
            background: linear-gradient(135deg, #d94452, #ed8c96);
            transform: translateY(-1px); box-shadow: 0 4px 8px rgba(230, 92, 106, 0.3);
        }
        
        .examples-title {
            text-align: center; font-size: 1rem; font-weight: 600; color: #7d1f2a; margin-bottom: 1rem;
        }
        
        .footer-info {
            margin-top: 1rem; padding: 1rem; background: rgba(254, 249, 249, 0.8);
            border-radius: 12px; border: 1px solid rgba(230, 92, 106, 0.1); text-align: center;
        }
        
        .footer-info p { color: #a02834; margin: 0.5rem 0; }
        
        .disclaimer {
            font-size: 0.8rem; color: #30060b; font-style: italic; margin-top: 1rem; padding-top: 1rem; font-weight: bold;
            border-top: 1px solid rgba(230, 92, 106, 0.1);
        }
    """
) as demo:
    
    # Header section
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="main-header">
                    <h1 class="main-title">🧠 AI Alzheimer Analyzer</h1>
                    <p class="main-subtitle">Advanced Deep Learning for Medical Imaging Analysis</p>
                    <p class="main-description">
                        Upload brain MRI scans to analyze and classify different stages of Alzheimer's disease 
                        based on state-of-the-art ConvNeXt neural network architecture.
                    </p>
                </div>
            """)
    
    # Main interface
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Brain MRI Scan", elem_classes="upload-container")
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg", elem_classes="predict-button")
        
        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=4, label="Classification Results", elem_classes="results-container")
    
    # Examples section
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="examples-title">Sample Images (click or drag the test examples)</div>')
            example_images = gr.Examples(
                examples=[
                    ["tests/data/mildDem659.jpg"],
                    ["tests/data/moderateDem42.jpg"],
                    ["tests/data/nonDem2213.jpg"],
                    ["tests/data/verymildDem1462.jpg"],
                    ["tests/data/mildDem257.jpg"],
                    ["tests/data/moderateDem20.jpg"],
                    ["tests/data/nonDem2031.jpg"],
                    ["tests/data/verymildDem1082.jpg"]
                ],
                inputs=image_input
            )
    
    # Footer information
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="footer-info">
                    <p><strong>Technical Details:</strong> ONNX Runtime inference (load from HuggingFace Hub model) with Databricks Model API as fallback</p>
                    <p>🔒 Privacy Protected | ⚡ Fast Processing | 🎯 High Accuracy</p>
                    <p class="disclaimer">
                        ⚠️ This app is for research and educational purposes only.<br/> 
                        It should not replace professional medical diagnosis or consultation.
                    </p>
                </div>
            """)
    
    # Event binding
    predict_btn.click(fn=predict, inputs=image_input, outputs=output_label)

if __name__ == "__main__":
    if not DATABRICKS_TOKEN:
        print("Warning: DATABRICKS_TOKEN not loaded. Please check your .env file.")
    else:
        print("🚀 Initializing ONNX model...")
        try:
            load_onnx_model()
            print("✅ ONNX model loaded successfully.")
        except Exception as e:
            print(f"⚠️ ONNX loading failed. Using Databricks API fallback: {str(e)}")
        
    print("Starting Gradio application...")
    demo.launch()