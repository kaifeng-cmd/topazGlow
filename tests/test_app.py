import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from app import preprocess_image, predict_databricks, softmax, CLASS_NAMES, predict, load_onnx_model

# Test image preprocessing
def test_preprocess_image():
    img = Image.new("RGB", (256, 256), color="white")
    img_array = preprocess_image(img)
    assert img_array.shape == (1, 3, 224, 224)
    assert img_array.dtype == np.float32

# Test softmax function
def test_softmax():
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    probs = softmax(logits)
    assert np.allclose(np.sum(probs), 1.0)
    assert len(probs) == 4

# Mock ONNX inference
@patch("app.ort.InferenceSession")
@patch("app.os.path.exists")
@patch("app.hf_hub_download")
def test_predict_onnx(mock_hf_download, mock_exists, mock_ort_session):
    # Simulate local ONNX model exists
    mock_exists.return_value = True
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.run.return_value = [np.array([[1.0, 2.0, 3.0, 4.0]])]
    mock_ort_session.return_value = mock_session
    
    img = Image.new("RGB", (256, 256), color="white")
    result = predict(img)
    
    expected = {CLASS_NAMES[i]: float(p) for i, p in enumerate(softmax(np.array([1.0, 2.0, 3.0, 4.0])))}
    assert result == pytest.approx(expected, rel=1e-5)

# Mock Databricks API call
@patch("app.requests.post")
def test_predict_databricks(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"predictions": [[1.0, 2.0, 3.0, 4.0]]}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    img_array = np.random.rand(1, 3, 224, 224).astype(np.float32)
    result = predict_databricks(img_array)
    
    expected = {CLASS_NAMES[i]: float(p) for i, p in enumerate(softmax(np.array([1.0, 2.0, 3.0, 4.0])))}
    assert result == pytest.approx(expected, rel=1e-5)
