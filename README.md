# Predictive Urban Mobility using Bluebikes & Boston GIS Data

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


## Planned Folder Structure
```
bluebikes-mlops/
│── README.md # Project description and setup
│── requirements.txt # Python dependencies
│── config/ # Config files for data sources, models, API
│── data/ # Raw and processed datasets (samples only, not full dumps)
│ ├── raw/ # Direct API pulls (GBFS, GIS, Weather)
│ └── processed/ # Cleaned and feature-engineered data
│── notebooks/ # Jupyter notebooks for EDA and prototyping
│── src/ # Source code
│ ├── ingestion/ # Scripts to fetch Bluebikes + GIS + weather data
│ ├── preprocessing/ # Cleaning and merging datasets
│ ├── features/ # Feature engineering
│ ├── training/ # Model training scripts
│ ├── evaluation/ # Model evaluation (RMSE, MAE, classification metrics)
│ ├── deployment/ # FastAPI app, Dockerfile
│ └── monitoring/ # Pipeline & model monitoring scripts
│── tests/ # Unit and integration tests
│── docs/ # Diagrams, data cards, scoping document
```