FROM python:3.9-slim
WORKDIR /app

# Install system dependencies if needed (e.g. for pandas/numpy compilation)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy app source
COPY . .

# Ensure models directory exists and is copied
# (Assuming models are in ./models relative to this Dockerfile)

# Cloud Run injects PORT
ENV PORT=8080
EXPOSE 8080

# Run with Gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 ml-service:app
