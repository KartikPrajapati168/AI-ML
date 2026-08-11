# CricIQ - IPL Prediction System

## Project Overview

CricIQ is a Machine Learning and Django-based web application that uses historical IPL cricket data to make predictions based on the trained Machine Learning model.

The project uses a Random Forest Classifier to learn patterns from the IPL dataset and generate predictions from user-provided inputs.

The Machine Learning model is integrated with a Django web application to provide a user-friendly interface for making predictions.

## Objective

The main objectives of this project are:

- Analyze historical IPL data
- Perform data preprocessing and feature engineering
- Train a Machine Learning classification model
- Use Random Forest Classifier for prediction
- Save the trained model and encoders
- Integrate the Machine Learning model with Django
- Provide predictions through a web application

## Machine Learning Model

The project uses:

**Random Forest Classifier**

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to improve prediction performance and reduce overfitting.

## Dataset

The project is based on historical IPL cricket data.

The dataset is processed and used to train the Random Forest Classifier.

The model learns relationships between the selected IPL match features and the target variable to generate predictions.

## Technologies Used

### Backend
- Python
- Django

### Machine Learning
- Scikit-learn
- Random Forest Classifier
- Pandas
- NumPy

### Development
- Jupyter Notebook
- VS Code
- Python Virtual Environment (`venv`)

## Project Structure

```text
CricIQ_project/
│
├── criciq_project/
│   ├── settings.py
│   ├── urls.py
│   └── ...
│
├── data/
│   ├── model.pkl
│   ├── team_encoder.pkl
│   ├── venue_encoder.pkl
│   └── ...
│
├── manage.py
├── criciq_model_training.ipynb
├── requirements.txt
├── README.md
└── venv/