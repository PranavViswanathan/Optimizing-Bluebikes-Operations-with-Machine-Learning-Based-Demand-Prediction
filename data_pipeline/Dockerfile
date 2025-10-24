# Use official Airflow image
FROM apache/airflow:2.5.1

# Switch to airflow user (non-root)
USER airflow

# Install extra packages
RUN pip install --no-cache-dir tqdm python-dotenv requests pandas
