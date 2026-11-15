FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-api.txt ./requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt
COPY backend ./backend

CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
