import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Load data
data = load_diabetes()
X = data.data
y = data.target

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X, y)

def predict(values):
    values = scaler.transform([values])
    result = model.predict(values)
    return round(float(result[0]), 2)
