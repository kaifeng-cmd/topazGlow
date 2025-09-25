import gradio as gr
import requests
import numpy as np
import cv2
import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv

load_dotenv()

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
URL = os.environ.get("URL")

CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Image transformation and preparation
val_test_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_image(img):
    img_rgb = np.array(img)
    transformed = val_test_transforms(image=img_rgb)
    img_tensor = transformed["image"]
    img_array = img_tensor.unsqueeze(0).numpy()
    return img_array

def create_request_payload(data):
    return {'inputs': data.tolist()}

# Softmax function to convert logits to probabilities
def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)

def predict(image):
    if not DATABRICKS_TOKEN:
        raise gr.Error("DATABRICKS_TOKEN environment variable not set!")
    
    try:
        # 1. Preprocess image
        img_array = preprocess_image(image)
        
        # 2. Create JSON payload
        payload = create_request_payload(img_array)
        data_json = json.dumps(payload)
        
        # 3. Send POST request
        headers = {
            'Authorization': f'Bearer {DATABRICKS_TOKEN}',
            'Content-Type': 'application/json'
        }
        response = requests.post(URL, headers=headers, data=data_json, timeout=120)
        response.raise_for_status()
        
        result = response.json()

        # 4. Extract logits from the returned JSON
        logits = np.array(result['predictions'][0])
        
        probabilities = softmax(logits)
        
        # 5. Associating class names with their corresponding probabilities
        confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        
        return confidences
        
    except requests.exceptions.Timeout:
        raise gr.Error("Request timed out. The model may be cold-starting, please try again later.")
    except requests.exceptions.HTTPError as err:
        raise gr.Error(f"API request failed: {err.response.status_code} - {err.response.text}")
    except Exception as e:
        raise gr.Error(f"Unknown error: {str(e)}")

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Brain MRI Image"),
    outputs=gr.Label(num_top_classes=4, label="Prediction Result"),
    title="Alzheimer's Disease Staging Classifier",
    description="Upload a brain MRI image, and the model will predict whether the patient is in the mild, moderate, non, or very mild stage of Alzheimer's disease. The model is provided through Databricks Serving API Endpoint.",
    examples=[
        ["C:/Users/dream/Downloads/test_dataset/ModerateDemented/moderateDem0.jpg"],
        ["C:/Users/dream/Downloads/test_dataset/NonDemented/nonDem607.jpg"]
    ]
)

if __name__ == "__main__":
    if not DATABRICKS_TOKEN:
        print("Fatal error: DATABRICKS_TOKEN not loaded. Please check your .env file.")
    else:
        print("Gradio application is ready, starting...")
        demo.launch()