# Chronic Kidney Disease (CKD) Prediction System

This project involves building a machine learning model to predict the likelihood of Chronic Kidney Disease (CKD) using patient clinical data. The model was trained using Python and scikit-learn, then deployed locally as a Flask API and tested using Postman.

## Problem Statement

Chronic Kidney Disease is often underdiagnosed in its early stages. Early prediction using machine learning can help support timely clinical decisions and improve patient outcomes.

## Features

- Predicts CKD likelihood based on clinical parameters
- RESTful API endpoint for predictions
- Returns probability scores and classification results

## Tech Stack

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Flask
- Postman

## Methodology

1. Data cleaning and preprocessing
2. Encoding of categorical variables
3. Feature scaling
4. Feature selection
5. Model training using Random Forest
6. Model evaluation
7. Local API deployment using Flask

## How to Run the Project

1. Install dependencies: `pip install -r requirements.txt`
2. Start the Flask application: `python app.py`
3. The app will run locally at: http://127.0.0.1:5000/

## How to Test the API

Send a POST request to `http://127.0.0.1:5000/predict` with patient data in JSON format using Postman or curl.

Example request body:
```json
{
  "age": 48,
  "blood_pressure": 80,
  "specific_gravity": 1.020,
  ...
}
```

## Dataset
The dataset used in this project contains clinical and laboratory features related to Chronic Kidney Disease.  
It includes both numerical and categorical variables commonly used in CKD diagnosis.
The dataset was sourced from a publicly available CKD dataset and used solely for educational and research purposes.
