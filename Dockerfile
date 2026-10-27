# Image de l'API : le modèle est entraîné pendant le build pour que l'image livrée soit autonome
# et que chaque image corresponde à une version de modèle identifiable.
FROM python:3.12-slim AS train
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data/samples ./data/samples
ARG TRAIN_DATA=data/samples/tickets_sample.csv
RUN python -m src.data.prepare --input ${TRAIN_DATA} && python -m src.model.train

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api ./api
COPY --from=train /app/models ./models
COPY --from=train /app/reports ./reports
RUN useradd -m appuser && mkdir -p monitoring && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
