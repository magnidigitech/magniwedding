FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libglib2.0-0 libgl1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless \
    && python -c "import cv2; print('OpenCV successfully loaded:', cv2.__version__)"

COPY app.py index_photos.py ./
COPY static ./static

ENV STORAGE_DIR=/storage
EXPOSE 3000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-3000}"]

