FROM apache/airflow:2.5.1

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

USER airflow
# Install Python packages - versions that actually exist for Python 3.7
RUN pip install --no-cache-dir \
    tqdm \
    python-dotenv \
    requests \
    pandas \
    seaborn \
    matplotlib \
    numpy \
    scipy \
    mlflow==1.30.0 \
    lightgbm==3.3.5 \
    xgboost==1.6.2 \
    scikit-learn==1.0.2 \
    joblib