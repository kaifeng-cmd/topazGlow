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

# global
onnx_session = None
input_name = None

# --- Configuration ---
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_URL = os.environ.get("DATABRICKS_URL")
CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
HF_REPO_ID = "KaiSKX/Alzheimer_ConvNeXtCNN"
ONNX_MODEL_PATH = "./onnx/convnext_model.onnx"
LOCAL_ONNX_PATH = "./convnext_model.onnx"  # Local ONNX file for development

# --- Image transformation and preparation ---
val_test_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_image(img):
    img_rgb = np.array(img)
    transformed = val_test_transforms(image=img_rgb)
    img_tensor = transformed["image"]
    img_array = img_tensor.unsqueeze(0).numpy().astype(np.float32)
    return img_array

def create_request_payload(data):
    return {'inputs': data.tolist()}

# --- Softmax function to convert logits to probabilities ---
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
        print("🔄 Initial loading of ONNX model...")
        if os.path.exists(LOCAL_ONNX_PATH):
            print(f"Loading local ONNX model from {LOCAL_ONNX_PATH}")
            onnx_session = ort.InferenceSession(LOCAL_ONNX_PATH)
        else:
            print(f"Local ONNX model not found, downloading from Hugging Face Hub: {HF_REPO_ID}/{ONNX_MODEL_PATH}")
            onnx_path = hf_hub_download(repo_id=HF_REPO_ID, filename=ONNX_MODEL_PATH, repo_type="model")
            onnx_session = ort.InferenceSession(onnx_path)
        
        input_name = onnx_session.get_inputs()[0].name
        print("✅ ONNX model is finished to load.")
    else:
        print("⚡ Using cache ONNX model.")
    
    return onnx_session

def predict_databricks(img_array):
    if not DATABRICKS_TOKEN:
        raise gr.Error("DATABRICKS_TOKEN environment variable not set!")
    
    try:
        # Create JSON payload
        payload = create_request_payload(img_array)
        data_json = json.dumps(payload)
        
        # Send POST request
        headers = {
            'Authorization': f'Bearer {DATABRICKS_TOKEN}',
            'Content-Type': 'application/json'
        }
        response = requests.post(DATABRICKS_URL, headers=headers, data=data_json, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        logits = np.array(result['predictions'][0])
        probabilities = softmax(logits)
        confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        return confidences
    
    except requests.exceptions.Timeout:
        raise gr.Error("Request timed out. The model may be cold-starting, please try again later.")
    except requests.exceptions.HTTPError as err:
        raise gr.Error(f"API request failed: {err.response.status_code} - {err.response.text}")
    except Exception as e:
        raise gr.Error(f"Databricks API error: {str(e)}")

def predict(image):
    try:
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Try ONNX inference first
        try:
            session = load_onnx_model()
            outputs = session.run(None, {input_name: img_array})[0]
            probabilities = softmax(outputs[0])
            confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
            return confidences
        except Exception as e:
            print(f"ONNX inference failed, falling back to Databricks API: {str(e)}")
            # Fallback to Databricks API (slow due to Databricks free CPU limit)
            return predict_databricks(img_array)
    
    except Exception as e:
        raise gr.Error(f"Inference failed: {str(e)}")

# --- Gradio Interface ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Brain MRI Image"),
    outputs=gr.Label(num_top_classes=4, label="Prediction Result"),
    title="Alzheimer's Disease Staging Classifier",
    description="Upload a brain MRI image to predict dementia stage. Prioritizes ONNX inference (from Hugging Face Hub); Databricks API is a fallback option",
    examples=[
        ["tests/data/mildDem659.jpg"],
        ["tests/data/moderateDem42.jpg"],
        ["tests/data/nonDem2213.jpg"],
        ["tests/data/verymildDem1462.jpg"]
    ]
)

if __name__ == "__main__":
    if not DATABRICKS_TOKEN:
        print("Fatal error: DATABRICKS_TOKEN not loaded. Please check your .env file.")
    else:
        print("Gradio application is ready, starting...")
        demo.launch()

if __name__ == "__main__":
    if not DATABRICKS_TOKEN:
        print("Fatal error: DATABRICKS_TOKEN not loaded. Please check your .env file.")
    else:
        print("🚀 Loading ONNX model...")
        try:
            load_onnx_model()
            print("✅ ONNX model is loaded.")
        except Exception as e:
            print(f"⚠️ ONNX is loading failed. Use fallback Databricks API: {str(e)}")
        
        print("Gradio application is ready, starting...")
        demo.launch()