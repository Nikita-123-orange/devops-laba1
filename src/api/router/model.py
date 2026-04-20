from fastapi import APIRouter
from datetime import datetime



prefix  = "/model"

router = APIRouter(prefix=prefix, tags=['model'])

@router.get(prefix)
def train_model():
    return {
        "status": "Model trained successfully",
        "datetime": datetime.now()
    }
    
    
@router.get(prefix)
def predict_model():
    return {
        "status": "Prediction made successfully",
        "datetime": datetime.now()
    }