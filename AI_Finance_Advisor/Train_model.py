import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils import load_data


# LOAD DATA

df = load_data("data/expenses.csv")

# FEATURE ENGINEERING

df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
# one hot encoding
df = pd.get_dummies(df, columns=['category'])

# PREPARE FEATURES AND TARGET

features = [
    'day',
    'month',
    'category_Bills',
    'category_Food',
    'category_Shopping',
    'category_Travel'
]

# Add any missing columns with 0 (safety check)
for col in features:
    if col not in df.columns:
        df[col] = 0

X = df[features]
y = df['amount']


# TRAIN MODEL

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("Model trained successfully!")


# SAVE MODEL

os.makedirs("saved_model", exist_ok=True)
pickle.dump(model, open("saved_model/model.pkl", "wb"))

print("Model saved at saved_model/model.pkl")
