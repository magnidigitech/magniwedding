FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index_photos.py ./
COPY static ./static

ENV STORAGE_DIR=/storage
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
