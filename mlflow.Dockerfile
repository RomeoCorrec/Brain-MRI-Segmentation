FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mlflow==2.14.3 psycopg2-binary boto3==1.38.0
EXPOSE 5000
