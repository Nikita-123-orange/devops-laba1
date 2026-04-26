# test_api.py

import pytest
from fastapi.testclient import TestClient
from io import BytesIO
import numpy as np
from PIL import Image

# Импортируем твой роутер и приложение (указать правильный путь)
from your_main_module import app  # где app = FastAPI() и подключён router

client = TestClient(app)

# Пути к тестовым файлам (или можно читать прямо в теле теста)
CSV_PATH = "tests/sample_test_2.csv"
IMAGE_PATH = "tests/sample_image_0.png"  


def test_smoke():
    """Проверка дымового теста модели."""
    response = client.post("/test/smoke")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model"] == "LOG_REG"
    assert data["test_type"] == "smoke"
    assert "accuracy" in data["scores"]


def test_functional():
    """Проверка функционального теста модели."""
    response = client.post("/test/func")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model"] == "LOG_REG"
    assert data["test_type"] == "func"
    assert "accuracy" in data["scores"]


def test_predict_with_csv():
    """Предсказание по CSV-файлу (две строки)."""
    with open(CSV_PATH, "rb") as f:
        content = f.read()

    files = {"file": ("sample_test_2.csv", content, "text/csv")}
    response = client.post("/test/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model"] == "LOG_REG"
    assert "predicted_class" in data["scores"]
    assert "confidence" in data["scores"]
    # Можно проверить, что класс — целое число (0 или 1, зависит от модели)
    assert isinstance(data["scores"]["predicted_class"], int)


def test_predict_with_image():
    """Предсказание по grayscale-изображению 28x28."""
    # Создаём тестовое изображение (если нет реального файла)
    img = Image.new("L", (28, 28), color=128)  # "L" = grayscale
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"file": ("test_image.png", img_bytes, "image/png")}
    response = client.post("/test/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model"] == "LOG_REG"
    assert "predicted_class" in data["scores"]
    assert "confidence" in data["scores"]


def test_predict_invalid_file():
    """Передача неподдерживаемого формата файла."""
    files = {"file": ("test.txt", b"just text", "text/plain")}
    response = client.post("/test/predict", files=files)
    assert response.status_code == 400
    assert "неподдерживаемый тип" in response.json()["detail"].lower()


def test_predict_empty_csv():
    """Пустой CSV-файл."""
    empty_csv = BytesIO(b"")
    files = {"file": ("empty.csv", empty_csv, "text/csv")}
    response = client.post("/test/predict", files=files)
    assert response.status_code == 400
    assert "пуст" in response.json()["detail"].lower()