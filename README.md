
# Medical Insurance Cost Prediction

A machine learning project for predicting medical insurance costs using Linear Regression.

## Project Objective

The objective of this project is to predict medical insurance charges based on patient information such as age, BMI, number of children, sex, smoking status, and region.

## Dataset

The dataset contains 1338 records and 7 columns:

- age
- sex
- bmi
- children
- smoker
- region
- charges

During data cleaning, one duplicate record was removed, leaving 1337 records for modeling.

## Project Workflow

1. Load the dataset
2. Explore the data
3. Check missing values and duplicates
4. Encode categorical variables
5. Split the data into training and testing sets
6. Train a Linear Regression model
7. Make predictions
8. Evaluate the model
9. Build a simple Streamlit interface

## Data Preprocessing

Categorical columns (`sex`, `smoker`, and `region`) were converted into numerical values using one-hot encoding.

The final input features were:

- age
- bmi
- children
- sex_male
- smoker_yes
- region_northwest
- region_southeast
- region_southwest

The target variable is `charges`.

## Model

**Algorithm:** Linear Regression

The dataset was divided into:

- 80% training data
- 20% testing data

**Random State:** 42

## Model Evaluation

- MAE: 4181.19
- MSE: 33596915.85
- RMSE: 5796.28
- R² Score: 0.7836

## Streamlit App

A simple Streamlit application is included in `app.py`. It allows the user to enter patient information and get an estimated medical insurance charge.

## Project Files

- `medical_insurance_prediction.ipynb` — Jupyter/Colab notebook containing the project work
- `insurance.csv` — dataset
- `train_model.py` — trains the model and creates `model.pkl`
- `model.pkl` — trained Linear Regression model
- `app.py` — Streamlit prediction application
- `requirements.txt` — required Python libraries
- `.gitignore` — files and folders ignored by Git
- `README.md` — project documentation

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
Train the model:

python train_model.py

Run the Streamlit app:

streamlit run app.py
