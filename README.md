---
title: Alzheimer Detection
emoji: 🧠
colorFrom: green
colorTo: pink
sdk: gradio
sdk_version: 5.46.1
app_file: app.py
pinned: false
python_version: "3.11"
license: apache-2.0
short_description: To detect Alzheimer based on ConvNeXt from brain MRI image
---
⚠️ **IMPORTANT:** Do not modify the above YAML metadata block. This is required for automatic deployment to Hugging Face Spaces via GitHub Actions.

# AI Alzheimer Analyzer

A comprehensive end-to-end machine learning pipeline for Alzheimer staging detection using brain MRI scans, built with advanced deep learning techniques, CNN architecture (ConvNeXt) and production-ready MLOps practices.

<p align="center">
  <img src="https://img.shields.io/badge/Alzheimer%20Detection-739BD0?style=for-the-badge&logoColor=white" width="100%" />
</p>

<p align="center">
  <i>Built using:</i><br></br>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Kaggle-grey?logo=kaggle"/>
  <img src="https://img.shields.io/badge/MLflow-lightblue?logo=mlflow"/>
  <img src="https://img.shields.io/badge/Databricks-white?logo=databricks"/>
  <img src="https://img.shields.io/badge/ONNX-white?logo=onnx&logoColor=black"/>
  <img src="https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gradio-white?logo=gradio&logoColor=FF6F00&labelColor=white&color=white"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-orange?logo=githubactions&logoColor=black"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=red"/>
</p>

## 🎯 Project Overview

This project represents a complete end-to-end machine learning pipeline for Alzheimer's disease detection using brain MRI scans. This implementation not only involves model training but also focuses on production-ready MLOps practices, app development, and deployment strategies.

**What makes this special:**
- **Full MLOps Pipeline**: From data preprocessing to model API serving with proper model versioning and monitoring.
- **Good Training Techniques**: Progressive unfreezing, Automatic mixed precision (AMP), Early Stopping, ReduceLROnPlateau.
- **Multiple Model Format Support**: PyTorch (.pth), ONNX (.onnx), and MLflow Model with inference speed benchmarking.
- **Model and App Deployment**: REST API model serving via Databricks and Gradio app via Hugging Face Spaces.
- **CI/CD Pipeline**: Automated unit testing and deployment.

## 📋 Summary Pipeline

1. **Data → Preprocessing**: `Kaggle Alzheimer's MRI dataset` with data preprocessing.
2. **Parallel Computing**: Multi-GPU training with `NVIDIA T4` acceleration (free to use at Kaggle).
3. **AMP Training**: `Automatic Mixed Precision` FP16 + FP32 mixed for faster training and reduced memory usage.
4. **Progressive Unfreezing**: Safe transfer learning with staged layer unfreezing.
5. **MLflow Experiment Tracking**: Comprehensive logging (ex. lr, epoch, loss, acc) via `MLflow` on `Databricks` cloud platform.
6. **Explainable AI (XAI)**: `Grad-CAM` heatmaps showing model focus areas.
7. **MLOps with MLflow & Databricks**: Model registry, checkpoint artifact management, model version control, and model API deployment.
8. **Model Format Benchmarking**: `PyTorch native` vs `ONNX runtime` inference speed comparison.
9. **Databricks API Model Serving**: REST API endpoints for model inference.
10. **Hugging Face Hub Integration**: Model storage for easier download by public community.
11. **Gradio Web App**: Interactive UI for model prediction.
12. **CI/CD Pipeline**: Automated unit testing (`pytest`) and deployment (`Hugging Face Spaces`) via `GitHub Actions`.

## 🌐 Live Demo & Resources

| Resource | Link/Path | Description |
|----------|------|-------------|
| 🚀 **Live App** | [Hugging Face Spaces](https://huggingface.co/spaces/KaiSKX/Alzheimer_Detection) | Interactive web app - try it now! |
| 📄 **Model Hub** | [Hugging Face Model](https://huggingface.co/KaiSKX/Alzheimer_ConvNeXtCNN) | Download models and inference documentation |
| 📚 **Kaggle Notebook** | [Complete Pipeline](https://www.kaggle.com/code/kongkaifeng/convnext-cnn-with-databricks-mlflow-mlops) | Full training code with explanations |
| 🧪 **Local Testing** | `onnxTesting.py` | Test ONNX model locally |
| 🖥️ **Local App** | `app.py` | Run Gradio app locally |

## 🛠️ Quick Start Guide

### Option 1: Try the Live App (Recommended)
1. Visit [Hugging Face Spaces](https://huggingface.co/spaces/KaiSKX/Alzheimer_Detection)
2. Upload/Drag ready brain MRI scan
3. Get instant predictions with confidence scores
4. No installation required

### Option 2: Local Development (refer `Local Development Setup section`)
```bash
# Clone the repo
git clone https://github.com/kaifeng-cmd/topazGlow
cd topazGlow

# Set up virtual environment
python -m venv venv
# On Windows: 
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
gradio app.py
```

## 💻 Local Development Setup

### Environment Variables
Create a `.env` file in the root directory:

```bash
# Required for Databricks API serving
DATABRICKS_TOKEN=your_databricks_token
# u need to serve trained MLflow Model as API endpoint first at Databricks Model Serving
DATABRICKS_URL=your_databricks_model_serving_url

# Required for Hugging Face integration
HF_TOKEN=your_huggingface_token
```

**Get these tokens:**
- **Databricks**: Free account at `Databricks Free Edition`
- **Hugging Face**: Free account at `Hugging Face`

### Git LFS Setup
This repository uses Git LFS for large file (for .ipynb in `notebook/`):

```bash
# If you want to download it (optional)
git lfs install
git lfs pull

# If you don't want to download it (optional)
git lfs install
git config filter.lfs.smudge "git-lfs smudge --skip %f"
git config filter.lfs.process "git-lfs filter-process --skip"
```

## 🔧 Configuration

### Model Info
- **Architecture**: ConvNeXt Small (CNN)
- **Input Size**: 224×224 images
- **Output Classes**: 4 (Mild/Moderate/Non/Very Mild Demented)
- **Format support**: .pth, .onnx, MLflow Model

### Training Configuration
- **Progressive Unfreezing**: 3 stages with decreasing learning rates (LR)
- **Floating Point Precision**: Automatic Mixed Precision (AMP) enabled
- **Multi-GPU**: 2× NVIDIA T4 parallel training
- **Batch Size**: 64

*More info pls refer to this [Kaggle Notebook](https://www.kaggle.com/code/kongkaifeng/convnext-cnn-with-databricks-mlflow-mlops).

## 🧪 Local Testing (before using CI/CD)

### Run Unit Tests
```bash
# Run the test file
pytest tests/test_app.py -v
```

### Test ONNX Model
```bash
python onnxTesting.py
```

### Manual App Testing
1. Start the app: `gradio app.py`
2. Upload test images from `tests/data/` or drag from given sample examples
3. Verify predictions match expected classes

*You need to have onnx model at your root directly first, pls refer to [Hugging Face Model](https://huggingface.co/KaiSKX/Alzheimer_ConvNeXtCNN) to learn how to download it.

## 🚀 Deployment

### Deploy to your own Hugging Face Spaces

1. **Fork/Clone** this repository
2. **Create a new Space** on [Hugging Face Spaces](https://huggingface.co/spaces)
3. **Update GitHub Secrets** in your repository for github actions to work:
   ```bash
   DATABRICKS_TOKEN=your_token
   DATABRICKS_URL=your_url
   HF_TOKEN=your_hf_token
   HF_SPACE_REPO=your-username/your-space-name
   ```
4. **Modify the workflow** in `.github/workflows/deploy_to_hf_spaces.yaml`:
   ```yaml
   environment:
     url: https://huggingface.co/spaces/your-username/your-space-name
   ```
    ...and here to change `KaiSKX` to your-username
   ```
   git remote add space https://KaiSKX:$HF_TOKEN@huggingface.co/spaces/${{ secrets.HF_SPACE_REPO }}
   ```
5. **Update Secrets** in your Hugging Face Space settings too:
   ```bash
   DATABRICKS_TOKEN=your_token
   DATABRICKS_URL=your_url
   HF_TOKEN=your_hf_token
   ```
6. **Push to main branch** - automatic deployment will trigger!

## 📸 Screenshots
*Here only cover and show the things that are not in Hugging Face Model Card & Kaggle Notebook.

### Gradio App
![Gradio App](materials/gradio_app.png)

### Hugging Face Model Card
![Model Card](materials/huggingface_modelCard.png)

### Trained Model registered & served as API (on Databricks)
![Model Serving](materials/databricks_modelServing.png)

### PyTest Unit Testing
![pytest](materials/unit_testing.png)

**NOTE**: The deployed Gradio app will automatically download and use model from Hugging Face Hub, the fallback/secondary option is to access Databricks API that serve the model. The API will be slow due to scale to zero (Databricks free tier usage policy).

## 🤝 Contribution

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite**: `pytest tests/ -v`
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>⭐ If you found this project helpful, please give it a star!</strong><br>
  <a href="https://huggingface.co/spaces/KaiSKX/Alzheimer_Detection">Try the Live Demo</a> • 
  <a href="https://www.kaggle.com/code/kongkaifeng/convnext-cnn-with-databricks-mlflow-mlops">View Kaggle Notebook</a> • 
  <a href="https://huggingface.co/KaiSKX/Alzheimer_ConvNeXtCNN">Download Models</a>
</p>

<p align="center">
  <i>Built for fun and research</i> 🙏
</p>

