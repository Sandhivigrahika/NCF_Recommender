FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR =/data


RUN useradd -m -u 1000 user \
    && mkdir -p /data \
    && chown -R user:user /data


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY --chown=user:user . .

USER user

#render provides $PORT at runtime; shell form so it expands.
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
