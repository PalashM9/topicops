FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PROJECT_ROOT=/project
WORKDIR /project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home worker
COPY app ./app
RUN mkdir -p /project/state && chown -R worker:worker /project
USER worker
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
