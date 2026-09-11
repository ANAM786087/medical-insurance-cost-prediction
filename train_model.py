import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Put insurance.csv in the same folder as this file.
df = pd.read_csv("insurance.csv")

# Remove duplicate record found during data cleaning.
df = df.drop_duplicates()

# Convert categorical columns to numerical columns.
df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

# Make sure boolean columns are numeric.
for column in df_encoded.columns:
    if df_encoded[column].dtype == bool:
        df_encoded[column] = df_encoded[column].astype(int)

X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("model.pkl created successfully.")
