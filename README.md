# Energy-ML: Seasonal Household Energy Consumption Prediction

## Description
This project develops machine learning models to predict household electricity consumption in Kerala, India, during **monsoon and summer seasons**.  
The project focuses on understanding energy usage patterns in residential buildings and identifying key drivers like home size, number of occupants, floors, and orientation.

---

## Problem Statement
Electricity consumption varies seasonally due to climate and occupant behavior.  
This project aims to:
- Predict electricity bills for monsoon and summer months.
- Analyze feature importance to understand energy drivers.
- Provide interpretable insights for energy-efficient home design.

---

## Dataset
- Source: Collected as part of the project *“Building Self Sustainable Smart Cities through Energy Efficient Homes using Intelligent Design”*
- Rows: 500
- Columns:
  - `Total Area (sqft)` — Total built-up area of the home
  - `Number of Occupants` — Total residents
  - `Number of Floors` — Number of floors
  - `Orientation` — Main direction the house faces
  - `KSEB bill in monsoon` — Target for monsoon model
  - `KSEB bill in summer` — Target for summer model

---

## Features & Targets

**Input Features:**
- Total Area (sqft)
- Number of Occupants
- Number of Floors
- Orientation

**Target Variables:**
- KSEB bill in monsoon
- KSEB bill in summer

---

## Methodology
- Preprocessing:
  - Label encoding for categorical features
  - Standard scaling
- Models:
  - CatBoost Regressor (primary)
- Evaluation:
  - 5-fold cross-validation
  - R² score calculation
  - Feature importance analysis

**Separate models were trained for monsoon and summer** to capture season-specific patterns.

---

## Results

**Monsoon Model:**
- R² (CV): 0.71
- Top Feature: Total Area (71.8%)

**Summer Model:**
- R² (CV): 0.65
- Top Feature: Total Area (71.1%), Orientation more important than in monsoon

> Insight: Summer electricity consumption is more influenced by occupant behavior and solar exposure, while monsoon usage is dominated by structural factors.

---

## Model Diagnostics
Residual analysis was performed for both monsoon and summer models.
Monsoon residuals show stable, random error distribution, while summer
residuals exhibit higher variance due to unobserved behavioral factors.



## How to Run

1. Install requirements:

```bash
pip install -r requirements.txt
