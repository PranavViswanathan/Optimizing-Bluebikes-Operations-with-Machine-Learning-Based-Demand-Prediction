# Use newer Airflow with Python 3.10
FROM apache/airflow:2.9.3-python3.10

# Install system dependencies as root
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        git \
        curl \
        gh && \
    rm -rf /var/lib/apt/lists/*

# Switch back to airflow user for runtime and Python deps
USER airflow

ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"

# Copy and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
