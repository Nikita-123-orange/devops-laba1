from typing import Any

from fastapi import APIRouter
from datetime import datetime
from src.predict import Predictor
from src.train import MultiModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PREFIX  = "/model"

router = APIRouter(prefix=PREFIX, tags=['model'])

@router.get(PREFIX)
def train_model():
    return {
        "status": "Model trained successfully",
        "datetime": datetime.now()
        
    }
    
    
@router.post(PREFIX)
def test_model(model: str = "LOG_REG", test_type: str = "smoke") -> dict[str, Any]:
    """Тестирование модели"""
    
    predictor = Predictor(model=model, test_type=test_type)
    model_name, score = predictor.predict()
    
    return {
        "status": "Prediction made successfully",
        "datetime": datetime.now(),
        "model": model_name,
        "score": score
    }