FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.14.3 psycopg2-binary boto3==1.38.0
CMD ["mlflow", "server", "--help"]
