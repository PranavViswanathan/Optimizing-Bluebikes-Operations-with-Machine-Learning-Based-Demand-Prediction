# Optimizing Bluebikes Operations with Machine Learning Based Demand Prediction

![airflow workflow](https://github.com/PranavViswanathan/Optimizing-Bluebikes-Operations-with-Machine-Learning-Based-Demand-Prediction/actions/workflows/airflow_ci.yaml/badge.svg)


![bluebikes-prediction-ui](assets/image.png)

![architecture](<assests/architecture.png>)

## About

### Overview
This project applies predictive analytics to Boston's Bluebikes bike-sharing system to address supply-demand mismatches that cause revenue loss and customer dissatisfaction when stations are empty or full.

Bluebikes serves 4.7 million annual rides but faces persistent challenges with bike availability at stations. Current mitigation relies on the "Bike Angels" user incentive program, which cannot adequately respond to dynamic demand from weather changes, events, or peak hours.

Bluebikes generates rich spatiotemporal datasets capturing cycling patterns, station utilization, and user behavior. By leveraging this data through predictive modeling, we can anticipate demand and proactively optimize bike distribution.

### Goals
- Reduce revenue loss from unavailable bikes
- Improve user satisfaction by ensuring bike availability
- Enable proactive operations instead of reactive responses
- Support city-wide sustainability and traffic reduction initiatives

### Approach
Develop predictive models using historical ridership patterns, weather data, seasonal variations, and event-driven demand spikes to forecast when and where bikes will be needed most.

## Project Structure

```
bluebikes-mlops/
├── data_pipeline/           # Data collection & processing
│   ├── dags/               # Airflow DAGs
│   ├── scripts/            # Processing scripts
│   └── data/               # Raw & processed data
│
├── model_pipeline/          # Model training & deployment
│   ├── dags/               # Training DAGs
│   ├── scripts/            # Training scripts
│   ├── models/             # Saved models
│   └── monitoring/         # Drift detection
│
├── model_deployment/        # Cloud Run deployment
│   ├── app.py              # Flask API
│   └── Dockerfile          # Container config
│
├── bluebikes-ui/           # Web interface
│   ├── frontend/           # React app
│   └── backend/            # Express.js API
│
├── docker-compose.yaml     # Service orchestration
├── Dockerfile              # Airflow image
├── setup.sh               # Setup wizard
└── .env.example           # Environment template
```

## Installation/Replication 

The following steps will ensure that anyone can set up and reproduce the Bluebikes pipeline - either locally or using the full Dockerized Airflow environment.

# Quick Start Guide

## First Time Setup (5 minutes)

```bash
# 1. Clone and enter directory
git clone https://github.com/YOUR_USERNAME/bluebikes-mlops.git
cd bluebikes-mlops

# 2. Run setup
chmod +x setup.sh
./setup.sh

# 3. Add your API keys
nano .env
# Fill in: NOAA_API_KEY, GITHUB_TOKEN, GITHUB_REPO

# 4. Start services
./start-airflow.sh

# 5. Open Airflow
open http://localhost:8080
# Login: airflow / airflow
```

## Daily Commands

| Action | Command |
|--------|---------|
| Start | `./start-airflow.sh` |
| Stop | `./stop-airflow.sh` |
| View logs | `docker compose logs -f` |
| Check status | `docker compose ps` |
| Restart | `docker compose restart` |
| Full reset | `./stop-airflow.sh --clean` |

## Trigger DAGs Manually

```bash
# Data collection
docker compose exec airflow-webserver airflow dags trigger data_pipeline_dag

# Model training
docker compose exec airflow-webserver airflow dags trigger bluebikes_integrated_bias_training

# Drift check
docker compose exec airflow-webserver airflow dags trigger drift_monitoring_dag
```

## Troubleshooting

**Services won't start?**
```bash
docker compose down -v
docker compose up airflow-init
docker compose up -d
```

**Out of memory?**
- Increase Docker memory to 8GB+
- Docker Desktop → Settings → Resources

**Permission denied?**
```bash
echo "AIRFLOW_UID=$(id -u)" >> .env
```

## API Keys Needed

| Service | Get it from | Required? |
|---------|-------------|-----------|
| NOAA | https://www.ncdc.noaa.gov/cdo-web/token | Yes |
| GitHub | GitHub -> Settings -> Developer settings | Yes |
| Discord | Your Discord server webhook | Optional |
| GCS | Google Cloud Console | Optional |

## Support

- Check logs: `docker compose logs -f`
- Issues: GitHub Issues page
- Reset: `./stop-airflow.sh --clean && ./setup.sh`


### Prerequisites


| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| Git | 2.30+ | `git --version` |
| Python | 3.10+ | `python3 --version` |

### Clone the repository
```bash
git clone https://github.com/PranavViswanathan/Optimizing-Bluebikes-Operations-with-Machine-Learning-Based-Demand-Prediction.git
cd Optimizing-Bluebikes-Operations-with-Machine-Learning-Based-Demand-Prediction
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│  Data Pipeline  │────▶│  Feature Store  │
│  - BlueBikes    │     │  (Airflow DAG)  │     │  (Processed)    │
│  - NOAA Weather │     │  Daily @ 12AM   │     │                 │
│  - Boston Cols  │     └─────────────────┘     └────────┬────────┘
└─────────────────┘                                      │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Cloud Run API  │◀────│  Model Registry │◀────│  Model Training │
│  /predict       │     │  (GCS Bucket)   │     │  (Airflow DAG)  │
│  /health        │     │                 │     │  Weekly         │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │  Drift Monitor  │◀─────────────┘
                        │  (Evidently AI) │
                        │  Triggers Retrain│
                        └─────────────────┘
```


## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup wizard
./setup.sh
```

This will:
- Check prerequisites
- Create necessary directories
- Set up environment configuration
- Build Docker images
- Initialize Airflow

### Option 2: Manual Setup

```bash
# 1. Create environment file
cp .env.example .env

# 2. Edit with your API keys
nano .env

# 3. Create required directories
mkdir -p keys
mkdir -p data_pipeline/data/{raw,processed}/{bluebikes,NOAA_weather,boston_clg}
mkdir -p model_pipeline/{models,mlruns,artifacts}
mkdir -p model_pipeline/monitoring/{baselines,reports,logs}

# 4. Set Airflow user ID
echo "AIRFLOW_UID=$(id -u)" >> .env

# 5. Build and initialize
docker compose build
docker compose up airflow-init
```

## Configuration

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `NOAA_API_KEY` | Yes | NOAA weather API key |
| `DISCORD_WEBHOOK_URL` | No | Discord notifications |
| `GITHUB_REPO` | Yes | Your GitHub repo (user/repo) |
| `GITHUB_TOKEN` | Yes | GitHub personal access token |
| `GCS_MODEL_BUCKET` | No* | GCS bucket for models |
| `AIRFLOW_UID` | Yes | Your user ID (run `id -u`) |

*Required only for cloud deployment features

### GCS Service Account (Optional)

If you want to use Google Cloud features:

1. Go to [GCP Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Go to IAM & Admin → Service Accounts
4. Create service account with roles:
   - Storage Admin
   - Storage Object Admin
5. Create and download JSON key
6. Save as `keys/gcs_service_account.json`

## Running the Pipeline

### Start Services

```bash
# Start all services in background
./start-airflow.sh
# or
docker compose up -d

# View logs
docker compose logs -f

# Check service health
./airflow-health-check.sh
```

### Access Interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | airflow / airflow |
| MLflow UI | http://localhost:5000 | None |

### Run DAGs

1. **Data Pipeline** (runs daily automatically)
   - Collects BlueBikes trip data
   - Fetches weather data from NOAA
   - Processes and creates features

2. **Model Training** (runs weekly automatically)
   - Trains XGBoost, LightGBM, Random Forest
   - Performs bias detection and mitigation
   - Promotes best model to production

3. **Drift Monitoring** (triggered after data pipeline)
   - Checks for data drift using Evidently AI
   - Triggers retraining if drift detected

### Manual DAG Triggers

```bash
# Trigger data pipeline
docker compose exec airflow-webserver airflow dags trigger data_pipeline_dag

# Trigger model training
docker compose exec airflow-webserver airflow dags trigger bluebikes_integrated_bias_training

# Trigger drift monitoring
docker compose exec airflow-webserver airflow dags trigger drift_monitoring_dag
```

### Stop Services

```bash
./stop-airflow.sh
# or
docker compose down
```


## API Endpoints

Once deployed to Cloud Run:

```bash
# Health check
curl https://your-cloud-run-url/health

# Get prediction
curl -X POST https://your-cloud-run-url/predict \
  -H "Content-Type: application/json" \
  -d '{"hour": 8, "day_of_week": 1, "temp_avg": 20}'

# Reload model
curl -X POST https://your-cloud-run-url/reload
```

## Troubleshooting

### Common Issues

**Docker permission denied**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Airflow webserver not starting**
```bash
# Check logs
docker compose logs airflow-webserver

# Restart services
docker compose down && docker compose up -d
```

**Database connection issues**
```bash
# Reset everything
docker compose down -v
docker compose up airflow-init
docker compose up -d
```

**Out of memory errors**
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 8GB+
```

### Getting Help

1. Check the [Issues](https://github.com/your-username/bluebikes-mlops/issues) page
2. Review Airflow logs: `docker compose logs -f`
3. Check individual task logs in Airflow UI

## Model Performance

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| XGBoost | 0.87 | 18.5 | 24.3 |
| LightGBM | 0.86 | 19.2 | 25.1 |
| Random Forest | 0.84 | 20.8 | 27.2 |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Team

- Nikhil Anil Prakash
- Harsh Shah
- Ananya Hegde
- Pranav Viswanathan
- Gyula Planky

---

Built with ❤️ for Northeastern University MLOps Course (December 2025)