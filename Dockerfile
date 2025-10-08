FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src /app/src
ENV PYTHONPATH=/app/src

# Uncomment to bake a trained model into the image
# COPY artifacts /app/artifacts

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "health_sepsis.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
