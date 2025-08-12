# report link
https://www.overleaf.com/1913259688rbpmrgjvdrcj#7c9aeb

# Heart Disease Prediction video presentation
https://drive.google.com/file/d/1KqJ9Lj48AH6X8US3wUg_O1KnSoYlm1oR/view?usp=drivesdk

# Heart disease predictionn app
https://drive.google.com/drive/folders/1ox5AEEJ_XTLLIS64LUpqehPR9rt6SB58?usp=sharing

This project aims to predict the likelihood of heart disease in a patient based on various medical attributes. It includes a machine learning model developed in a Jupyter Notebook and a Flask API to serve predictions.

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Dataset](#dataset)
* [Machine Learning Model](#machine-learning-model)
    * [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
    * [Model Training](#model-training)
    * [Model Evaluation](#model-evaluation)
    * [Model Saving](#model-saving)
* [API Endpoints](#api-endpoints)
* [Setup and Installation](#setup-and-installation)
    * [Prerequisites](#prerequisites)
    * [Cloning the Repository](#cloning-the-repository)
    * [Setting up the Python Environment](#setting-up-the-python-environment)
    * [Running the Flask API](#running-the-flask-api)
* [Usage](#usage)
    * [Making Predictions via API](#making-predictions-via-api)
* [File Structure](#file-structure)
* [Future Enhancements](#future-enhancements)

## Project Overview

This project utilizes a dataset of patient medical records to train various machine learning models for predicting the presence of heart disease. The best-performing model (Random Forest in this case, achieving 100% accuracy on the test set) is then saved and exposed via a Flask API, allowing for easy integration into other applications.

## Features

* **Data Analysis & Visualization**: Explores the heart disease dataset to understand feature distributions and correlations.
* **Machine Learning Model Training**: Trains and evaluates several classification models, including:
    * Logistic Regression
    * Naive Bayes
    * Support Vector Machine (SVM)
    * K-Nearest Neighbors (KNN)
    * Decision Tree
    * Random Forest
    * XGBoost
    * Neural Network (Keras)
* **Prediction API**: A Flask-based API (`app.py`) to make real-time predictions.
* **Model Persistence**: Saves the trained Random Forest model using `pickle` for later use by the API.

## Dataset

The dataset used is `heart.csv`. It contains 1025 rows and 14 columns (features). The target variable indicates the presence (1) or absence (0) of heart disease.

**Features:**

* **age**: Age of the patient.
* **sex**: Sex of the patient (1: male, 0: female).
* **cp**: Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic).
* **trestbps**: Resting blood pressure (in mm Hg).
* **chol**: Serum cholesterol in mg/dl.
* **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
* **restecg**: Resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy).
* **thalach**: Maximum heart rate achieved.
* **exang**: Exercise-induced angina (1 = yes; 0 = no).
* **oldpeak**: ST depression induced by exercise relative to rest.
* **slope**: The slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping).
* **ca**: Number of major vessels (0-3) colored by fluoroscopy.
* **thal**: Thallium stress test result (0: NULL, 1: fixed defect, 2: normal, 3: reversible defect).
* **target**: Presence of heart disease (0 = no, 1 = yes).

*(Note: The descriptions for `cp`, `restecg`, `slope`, and `thal` are based on common interpretations of these features in heart disease datasets. The notebook provides slightly different value mappings for `cp` and `thal` during data exploration. For consistency, ensure the values used for prediction match the model's training data encoding.)*

## Machine Learning Model

The primary model development is done in the `ML_Heart_Disease_dataset.ipynb` Jupyter Notebook.

### Data Exploration and Preprocessing

* Libraries such as NumPy, Pandas, Matplotlib, and Seaborn are used for data loading, manipulation, and visualization.
* The dataset is loaded and initial exploratory data analysis (EDA) is performed (e.g., `dataset.head()`, `dataset.shape`, `dataset.describe()`, `dataset.info()`).
* The correlation of each feature with the target variable is analyzed.
* Visualizations like count plots and bar plots are used to understand the relationship between features and the target.

### Model Training

* The data is split into training and testing sets (80% train, 20% test).
* Several classification algorithms are implemented and trained:
    * Logistic Regression
    * Naive Bayes (GaussianNB)
    * Support Vector Machine (SVC with linear kernel)
    * K-Nearest Neighbors (KNN with n\_neighbors=7)
    * Decision Tree Classifier
    * Random Forest Classifier
    * XGBoost Classifier
    * A simple Neural Network using Keras

### Model Evaluation

* Accuracy score is the primary metric used for evaluating the models.
* The Random Forest, Decision Tree, and XGBoost models achieve 100% accuracy on the test set with the chosen `random_state`.
* Logistic Regression: 86.34% accuracy
* Naive Bayes: 85.37% accuracy
* SVM: 83.9% accuracy
* KNN: 72.2% accuracy
* Neural Network: 85.85% accuracy (after rounding predictions)

### Model Saving

The trained Random Forest model (`rf`) is saved to a file named `heart_disease_random_forest_model.pkl` using the `pickle` library. This allows the model to be loaded and used by the Flask API.

## API Endpoints

The Flask application (`app.py`) provides the following endpoint:

* **`POST /predict`**:
    * **Request**: JSON payload containing the 13 input features.
        ```json
        {
            "age": 52,
            "sex": 1,
            "cp": 0,
            "trestbps": 125,
            "chol": 212,
            "fbs": 0,
            "restecg": 1,
            "thalach": 168,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "ca": 2,
            "thal": 3
        }
        ```
    * **Response**: JSON object containing the prediction, a human-readable label, and confidence scores.
        ```json
        {
            "prediction": 0,
            "prediction_label": "No Heart Disease",
            "confidence": {
                "no_heart_disease (class 0)": 0.9, // Example probability
                "heart_disease (class 1)": 0.1  // Example probability
            },
            "input_features": { /* ... input features ... */ }
        }
        ```

## Setup and Installation

### Prerequisites

* Python 3.x
* pip (Python package installer)

### Cloning the Repository

```bash
git clone [https://github.com/nanviya/HeartDiseaseApp.git](https://github.com/nanviya/HeartDiseaseApp.git)
cd HeartDiseaseApp
