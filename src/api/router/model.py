from typing import Any

from fastapi import APIRouter, HTTPException
from datetime import datetime
from src.predict import Predictor
from src.train import Model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PREFIX  = "/model"

router = APIRouter(tags=["model"])

@router.get(PREFIX)
def train_model():
    return {
        "status": "Model trained successfully",
        "datetime": datetime.now()
        
    }
    
    
@router.post(PREFIX)
def test_model(test_type: str = "smoke") -> dict[str, Any]:
    """Тестирование модели
    Args:
        test_type (str, optional): Тип теста ("smoke" или "func"). Defaults to "smoke".
    
    """
    try:
        predictor = Predictor(test_type=test_type)
        model_name, score = predictor.predict()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    return {
        "status": "Prediction made successfully",
        "datetime": datetime.now(),
        "model": model_name,
        "score": score
    }