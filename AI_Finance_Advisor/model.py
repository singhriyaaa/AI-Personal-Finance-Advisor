import pickle
from sklearn.ensemble import RandomForestRegressor
from utils import load_data, preprocess

# Load data
df = load_data("data/expenses.csv")

# Preprocess
X, y = preprocess(df)

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
with open("saved_model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
