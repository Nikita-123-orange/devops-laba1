# ML API для Fashion MNIST

Проект предоставляет REST API для обучения, тестирования и инференса модели логистической регрессии на датасете Fashion MNIST.

## Основные возможности
- Загрузка и предобработка данных (`preprocess.py`)
- Обучение модели с масштабированием признаков (`train.py`)
- API на FastAPI: проверка здоровья, smoke/функциональные тесты, предсказание по CSV или изображению (`/health`, `/test/smoke`, `/test/func`, `/test/predict`)
- Логирование в файл и консоль
- Юнит-тесты и покрытие (coverage)

## Быстрый старт

### Локальный запуск
```bash
pip install -r requirements.txt
python src/preprocess.py   # распаковка и split данных
python src/train.py        # обучение и сохранение scaler + model
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Запуск через Docker
```bash
docker-compose up --build
```

### Тестирование
```bash
coverage run -m src.unit_tests.test_preprocess
coverage run -a -m src.unit_tests.test_training
coverage report -m
```

## CI/CD (Jenkins)
- Автоматический клон репозитория, сборка Docker-образа, запуск контейнера, выполнение тестов, логи и coverage, публикация образа в Docker Hub.
- Подпись образа с помощью Cosign (публичный ключ в `cosign.pub`).

Конфигурация через `config.ini`, логи хранятся в `logfile.log`.