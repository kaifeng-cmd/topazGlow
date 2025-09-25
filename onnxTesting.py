# Example template for ONNX runtime inference
# pre-requisites: should have onnx model & test data

import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time

# Load ONNX model
session = ort.InferenceSession("convnext_model.onnx")
input_name = session.get_inputs()[0].name

# Define class names for mapping
class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Prepare input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("C:/Users/dream/Downloads/test_dataset/ModerateDemented/moderateDem39.jpg").convert("RGB")
input_data = transform(image).unsqueeze(0).numpy().astype(np.float32)

# Inference with timing
start_time = time.time()
output = session.run(None, {input_name: input_data})[0]
end_time = time.time()

inference_time = (end_time - start_time) * 1000
predicted_class = np.argmax(output, axis=1)[0]

print(f"Predicted class: {class_names[predicted_class]}")
print(f"Inference time: {inference_time:.2f} ms")

