FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py catalog.py providers.py ./
COPY static ./static

RUN useradd --system --no-create-home app
USER app

EXPOSE 8080
# One worker on purpose: caches, the Jev router's similarity index and rate limits live in process memory.
# The service hops call back into this same process, so SELF_URL must point at its own port.
CMD ["sh", "-c", "export SELF_URL=${SELF_URL:-http://127.0.0.1:$PORT}; exec uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1"]
