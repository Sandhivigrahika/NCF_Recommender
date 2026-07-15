FROM python:3.10-slim

# Keras 3 saved this model, so we do NOT set TF_USE_LEGACY_KERAS.
# tensorflow-cpu==2.19.1 bundles Keras 3, matching the file's keras_version 3.11.2.
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    DATA_DIR=/data

# HF Spaces runs the container as UID 1000. Create that user and a
# writable data dir it owns — code lives in /app (read-only), state in /data.
RUN useradd -m -u 1000 user \
    && mkdir -p /data \
    && chown -R user:user /data

WORKDIR /app

# Install deps first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app, owned by the runtime user.
COPY --chown=user:user . .

USER user

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]