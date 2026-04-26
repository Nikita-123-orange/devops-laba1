FROM python:3.14-slim

ENV PYTHONUNBUFFERED 1

WORKDIR / 

# Сначала копируем только requirements.txt (для кэширования слоя с зависимостями)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код
COPY . .
