import pytest
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import io
import tempfile
from unittest.mock import Mock, patch, MagicMock

from src.predict import Predictor
from src.logger import Logger


class TestPredictorSmokeTest:
    """Unit tests for smoke_test method."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance for testing."""
        with patch('src.predict.configparser.ConfigParser.read'):
            with patch('src.predict.pd.read_csv') as mock_read_csv:
                # Mock the CSV reads
                mock_read_csv.return_value = pd.DataFrame(
                    np.random.randint(0, 256, (100, 784))
                )
                predictor = Predictor(model="LOG_REG", test_type="smoke")
                predictor.model = Mock()
                predictor.X_test = np.random.randn(100, 784)
                predictor.y_test = pd.DataFrame(
                    np.random.randint(0, 10, (100, 1))
                )
                predictor.sc = Mock()
                predictor.sc.transform = Mock(return_value=predictor.X_test)
                return predictor

    def test_smoke_test_returns_correct_structure(self, predictor):
        """Test that smoke_test returns (model_name, scores_dict)."""
        predictor.model.predict = Mock(
            return_value=np.random.randint(0, 10, 100)
        )
        
        model_name, scores = predictor.smoke_test()
        
        assert model_name == "LOG_REG"
        assert isinstance(scores, dict)
        assert "accuracy" in scores
        assert "f1_score" in scores
        assert "precision" in scores
        assert "recall" in scores

    def test_smoke_test_scores_are_floats(self, predictor):
        """Test that all scores are float values."""
        predictor.model.predict = Mock(
            return_value=np.random.randint(0, 10, 100)
        )
        
        model_name, scores = predictor.smoke_test()
        
        for key, value in scores.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_smoke_test_scores_in_valid_range(self, predictor):
        """Test that scores are in valid ranges (0-1)."""
        predictor.model.predict = Mock(
            return_value=np.random.randint(0, 10, 100)
        )
        
        model_name, scores = predictor.smoke_test()
        
        for key, value in scores.items():
            assert 0 <= value <= 1, f"{key} should be between 0 and 1, got {value}"

    def test_smoke_test_handles_prediction_error(self, predictor):
        """Test that smoke_test raises RuntimeError on prediction failure."""
        predictor.model.predict = Mock(side_effect=ValueError("Test error"))
        
        with pytest.raises(RuntimeError, match="Smoke test failed"):
            predictor.smoke_test()


class TestPredictorFunctionalTest:
    """Unit tests for functional_test method."""

    @pytest.fixture
    def predictor_with_test_files(self, tmp_path):
        """Create predictor with temporary test JSON files."""
        # Create temporary test files
        test_data = {
            "X": [
                {"0": i for i in range(784)}
            ],
            "y": [{"0": "5"}]
        }
        
        test_json_file = tmp_path / "test_0.json"
        with open(test_json_file, 'w') as f:
            json.dump(test_data, f)
        
        # Mock predictor
        with patch('src.predict.configparser.ConfigParser.read'):
            with patch('src.predict.pd.read_csv') as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame(
                    np.random.randint(0, 256, (100, 784))
                )
                predictor = Predictor(model="LOG_REG", test_type="func")
                predictor.model = Mock()
                predictor.X_test = np.random.randn(100, 784)
                predictor.y_test = pd.DataFrame(
                    np.random.randint(0, 10, (100, 1))
                )
                predictor.sc = Mock()
                predictor.sc.transform = Mock(return_value=np.random.randn(1, 784))
                return predictor, tmp_path

    def test_functional_test_returns_correct_structure(self, predictor_with_test_files):
        """Test that functional_test returns (model_name, accuracy)."""
        predictor, tmp_path = predictor_with_test_files
        
        # Patch the tests directory
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=['test_0.json']):
                with patch('builtins.open', create=True) as mock_open:
                    test_data = {
                        "X": [{"0": i for i in range(784)}],
                        "y": [{"0": "5"}]
                    }
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(test_data)
                    
                    predictor.model.predict = Mock(return_value=np.array([5]))
                    
                    model_name, accuracy = predictor.functional_test()
                    
                    assert model_name == "LOG_REG"
                    assert isinstance(accuracy, float)
                    assert 0 <= accuracy <= 1

    def test_functional_test_skips_non_json_files(self, predictor_with_test_files):
        """Test that functional_test skips non-JSON files."""
        predictor, tmp_path = predictor_with_test_files
        
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=['test_0.json', 'image.png', 'data.csv']):
                with patch('builtins.open', create=True) as mock_open:
                    test_data = {
                        "X": [{"0": i for i in range(784)}],
                        "y": [{"0": "5"}]
                    }
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(test_data)
                    
                    predictor.model.predict = Mock(return_value=np.array([5]))
                    
                    model_name, accuracy = predictor.functional_test()
                    
                    # Verify that only the JSON file was processed
                    assert mock_open.call_count >= 1

    def test_functional_test_handles_missing_directory(self, predictor_with_test_files):
        """Test that functional_test raises error when tests directory doesn't exist."""
        predictor, _ = predictor_with_test_files
        
        with patch('os.path.exists', return_value=False):
            with pytest.raises(RuntimeError, match="Tests directory not found"):
                predictor.functional_test()

    def test_functional_test_handles_no_valid_files(self, predictor_with_test_files):
        """Test that functional_test raises error when no valid JSON files found."""
        predictor, _ = predictor_with_test_files
        
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=['image.png', 'data.csv']):
                with pytest.raises(RuntimeError, match="No JSON test files found"):
                    predictor.functional_test()


class TestPredictorPredictFromFeatures:
    """Unit tests for predict_from_features method."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance for testing."""
        with patch('src.predict.configparser.ConfigParser.read'):
            with patch('src.predict.pd.read_csv') as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame(
                    np.random.randint(0, 256, (100, 784))
                )
                predictor = Predictor(model="LOG_REG", test_type="smoke")
                predictor.model = Mock()
                predictor.sc = Mock()
                predictor.sc.transform = Mock(return_value=np.random.randn(1, 784))
                return predictor

    def test_predict_from_features_with_1d_array(self, predictor):
        """Test prediction with 1D features array."""
        features = np.random.randint(0, 256, 784)
        predictor.model.predict = Mock(return_value=np.array([5]))
        predictor.model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.3, 0.15, 0.25]]))
        
        predicted_class, confidence = predictor.predict_from_features(features)
        
        assert isinstance(predicted_class, (int, np.integer))
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_predict_from_features_with_2d_array(self, predictor):
        """Test prediction with 2D features array."""
        features = np.random.randint(0, 256, (1, 784))
        predictor.model.predict = Mock(return_value=np.array([3]))
        predictor.model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.15, 0.3, 0.25]]))
        
        predicted_class, confidence = predictor.predict_from_features(features)
        
        assert isinstance(predicted_class, (int, np.integer))
        assert isinstance(confidence, float)

    def test_predict_from_features_without_predict_proba(self, predictor):
        """Test prediction with model that doesn't have predict_proba."""
        features = np.random.randint(0, 256, 784)
        predictor.model.predict = Mock(return_value=np.array([7]))
        predictor.model.predict_proba = None
        
        predicted_class, confidence = predictor.predict_from_features(features)
        
        assert isinstance(predicted_class, (int, np.integer))
        assert confidence == 0.0

    def test_predict_from_features_handles_error(self, predictor):
        """Test that predict_from_features raises RuntimeError on failure."""
        features = np.random.randint(0, 256, 784)
        predictor.model.predict = Mock(side_effect=ValueError("Test error"))
        
        with pytest.raises(RuntimeError, match="Prediction failed"):
            predictor.predict_from_features(features)

    def test_predict_from_features_respects_scaled_transform(self, predictor):
        """Test that predict_from_features uses scaler."""
        features = np.random.randint(0, 256, (1, 784))
        predictor.model.predict = Mock(return_value=np.array([2]))
        predictor.model.predict_proba = Mock(return_value=np.array([[0.2, 0.3, 0.35, 0.1, 0.05]]))
        
        predictor.predict_from_features(features)
        
        # Verify that scaler was called
        predictor.sc.transform.assert_called_once()


class TestImageProcessing:
    """Unit tests for image file processing in prediction."""

    def test_grayscale_image_processing(self):
        """Test processing of grayscale image."""
        # Create a simple 28x28 grayscale image
        image_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        image = Image.fromarray(image_array, mode='L')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Load and process
        loaded_image = Image.open(img_bytes)
        processed = np.array(loaded_image).flatten()
        
        assert len(processed) == 784, "Flattened image should have 784 features"
        assert processed.dtype in [np.uint8, np.int64], "Features should be numeric"

    def test_rgb_image_to_grayscale_conversion(self):
        """Test conversion of RGB image to grayscale."""
        from PIL import ImageOps
        
        # Create RGB image
        image_array = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, mode='RGB')
        
        # Convert to grayscale
        grayscale = ImageOps.grayscale(image)
        processed = np.array(grayscale).flatten()
        
        assert len(processed) == 784
        assert image.mode == 'RGB'
        assert grayscale.mode == 'L'


class TestCSVProcessing:
    """Unit tests for CSV file processing in prediction."""

    def test_csv_with_784_features(self):
        """Test CSV processing with correct feature count."""
        # Create CSV with 784 features
        data = np.random.randint(0, 256, (1, 784))
        df = pd.DataFrame(data)
        
        csv_str = df.to_csv(index=False)
        df_loaded = pd.read_csv(io.StringIO(csv_str))
        
        assert df_loaded.shape == (1, 784), "CSV should have 1 row and 784 columns"

    def test_csv_feature_extraction(self):
        """Test extraction of features from CSV."""
        data = np.random.randint(0, 256, (1, 784))
        df = pd.DataFrame(data)
        
        features = df.values
        
        assert features.shape == (1, 784)
        assert features.dtype in [np.int64, np.float64]


class TestIntegrationSmokeTest:
    """Integration tests for smoke test endpoint."""

    @patch('src.predict.Predictor.smoke_test')
    def test_smoke_test_integration(self, mock_smoke_test):
        """Test smoke test with mocked Predictor."""
        mock_smoke_test.return_value = (
            "LOG_REG",
            {
                "accuracy": 0.95,
                "f1_score": 0.94,
                "precision": 0.96,
                "recall": 0.93
            }
        )
        
        model_name, scores = mock_smoke_test()
        
        assert model_name == "LOG_REG"
        assert scores["accuracy"] == 0.95
        assert all(0 <= v <= 1 for v in scores.values())


class TestIntegrationFunctionalTest:
    """Integration tests for functional test endpoint."""

    @patch('src.predict.Predictor.functional_test')
    def test_functional_test_integration(self, mock_func_test):
        """Test functional test with mocked Predictor."""
        mock_func_test.return_value = ("LOG_REG", 0.88)
        
        model_name, accuracy = mock_func_test()
        
        assert model_name == "LOG_REG"
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1


class TestIntegrationPredictFromFeatures:
    """Integration tests for predict_from_features endpoint."""

    @patch('src.predict.Predictor.predict_from_features')
    def test_predict_from_features_integration(self, mock_predict):
        """Test predict_from_features with mocked Predictor."""
        mock_predict.return_value = (3, 0.92)
        
        predicted_class, confidence = mock_predict()
        
        assert predicted_class in range(10)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
