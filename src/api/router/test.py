from typing import Any
from fastapi import APIRouter, HTTPException, File, UploadFile
from datetime import datetime
import io
import numpy as np
import pandas as pd
from PIL import Image

from src.api.model.test import TestResponse, PredictionResponse
from src.predict import Predictor
from src.logger import Logger


SHOW_LOG = True
MODEL = "LOG_REG"
logger = Logger(SHOW_LOG).get_logger(__name__)

router = APIRouter(prefix="/test", tags=["testing"])





@router.post("/smoke", response_model=TestResponse)
async def smoke_test() -> dict[str, Any]:
    """smoke тестирование модели (быстрая проверка базовой функциональности)."""
    try:
        logger.info(f"Запуск smoke теста для модели: {MODEL}")
        predictor = Predictor(model=MODEL, test_type="smoke")
        model_name, scores = predictor.smoke_test()
        logger.info(f"smoke тест пройден. Точность: {scores.get('accuracy', 0):.4f}")
        
        return {
            "status": "success",
            "datetime": datetime.now().isoformat(),
            "model": model_name,
            "test_type": "smoke",
            "scores": scores
        }
    except Exception as e:
        logger.error(f"smoke тест не удался: {str(e)}")
        raise HTTPException(status_code=500, detail=f"smoke тест не удался: {str(e)}")


@router.post("/func", response_model=TestResponse)
async def functional_test() -> dict[str, Any]:
    """Функциональное тестирование модели."""
    try:
        logger.info(f"Запуск функционального теста для модели: {MODEL}")
        predictor = Predictor(model=MODEL, test_type="func")
        model_name, scores = predictor.functional_test()
        accuracy = scores.get('accuracy', 0.0)  # извлечение точности для лога
        logger.info(f"Функциональный тест пройден. Точность: {accuracy:.4f}")
        
        return {
            "status": "success",
            "datetime": datetime.now().isoformat(),
            "model": model_name,
            "test_type": "func",
            "scores": scores   # словарь метрик
        }
    except Exception as e:
        logger.error(f"Функциональный тест не удался: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Функциональный тест не удался: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
async def predict_from_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """Предсказание по загруженному изображению или CSV-файлу."""
    try:
        logger.info(f"Запуск предсказания с моделью: {MODEL}")
        contents = await file.read()

        # Обработка в зависимости от расширения файла
        if file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            except pd.errors.EmptyDataError:
                raise HTTPException(400, "CSV-файл пуст или не содержит данных")
            except UnicodeDecodeError:
                raise HTTPException(400, "Некорректная кодировка CSV-файла")
            
            if df.empty:
                raise HTTPException(400, "CSV-файл пуст")
            features = df.values
        elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
            if image_array.size == 0:
                raise HTTPException(400, "Изображение не содержит данных")
            if len(image_array.shape) == 3:
                from PIL import ImageOps
                image = ImageOps.grayscale(image)
                image_array = np.array(image)
            features = image_array.flatten().reshape(1, -1)
        else:
            raise HTTPException(400, f"Неподдерживаемый тип файла: {file.filename}")

        # Проверка наличия данных
        if features.shape[0] == 0:
            raise HTTPException(400, "В загруженном файле нет образцов")

        predictor = Predictor(model=MODEL, test_type="smoke")
        predicted_class, confidence = predictor.predict_from_features(features)

        logger.info(f"Предсказание успешно: класс {predicted_class}, уверенность: {confidence:.4f}")

        return {
            "status": "success",
            "datetime": datetime.now().isoformat(),
            "model": MODEL,                    
            "scores": {                    
                "confidence": confidence,
                "predicted_class": predicted_class
            },
            "message": f"Предсказанный класс: {predicted_class}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Предсказание не удалось: {str(e)}")
        raise HTTPException(500, f"Предсказание не удалось: {str(e)}")