# Optimizing Bluebikes Operations with Machine Learning Based Demand Prediction

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


## Folder Structure
```
Optimizing-Bluebikes-Operations-with-Machine-Learning-Based-Demand-Prediction/
│── README.md # Project description and setup
│── requirements.txt # Python dependencies
│── config/ # Config files for data sources, models, API
│── data_pipeline/ 
│── notebooks/ # Jupyter notebooks for EDA and prototyping
│── Model/
├── Deployment/
│── tests/ # Unit and integration tests
│── docs/ # Diagrams, data cards, scoping document
```

## Installation

### Clone the repository
```bash
git clone https://github.com/PranavViswanathan/Optimizing-Bluebikes-Operations-with-Machine-Learning-Based-Demand-Prediction.git
cd Optimizing-Bluebikes-Operations-with-Machine-Learning-Based-Demand-Prediction
```
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Preliminary Project Timeline
```mermaid
    gantt
        title Project Timeline
        dateFormat  YYYY-MM-DD
        axisFormat  %b %d
    
        section Milestones
        Project Scoping          :done,    ms1, 2025-09-30, 1d
        Data Pipeline            :active,  ms2, 2025-10-01, 2025-10-28
        Model Development        :         ms3, 2025-10-29, 2025-11-11
        Model Deployment         :         ms4, 2025-11-12, 2025-12-09
        MLOps Expo               :milestone, ms5, 2025-12-12, 1d
```

## Status
This repository is in the **data pipeline phase**.  
Code, data pipelines, and models will be added in upcoming sprints.

