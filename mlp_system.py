# ============================================================
# MULTILAYER PERCEPTRON (MLP) FOR PREDICTIVE ANALYSIS
# ARCHITECTURE-BASED IMPLEMENTATION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. DATA INPUT LAYER
# ============================================================
def data_input_layer():
    print("=== DATA INPUT LAYER ===")
    data = load_diabetes()
    X = data.data
    y = data.target
    print("Dataset loaded.")
    return X, y


# ============================================================
# 2. DATA PREPROCESSING LAYER
# ============================================================
def preprocessing_layer(X, y):
    print("\n=== PREPROCESSING LAYER ===")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data split and scaled.")
    return X_train, X_test, y_train, y_test


# ============================================================
# 3. MODEL TRAINING LAYER (MLP)
# ============================================================
def model_training_layer(X_train, y_train):
    print("\n=== MODEL TRAINING LAYER (MLP) ===")

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    mlp.fit(X_train, y_train)
    print("MLP model trained.")
    return mlp


# ============================================================
# 4. PREDICTION LAYER
# ============================================================
def prediction_layer(model, X_test):
    print("\n=== PREDICTION LAYER ===")
    predictions = model.predict(X_test)
    print("Predictions completed.")
    return predictions


# ============================================================
# 5. EVALUATION LAYER
# ============================================================
def evaluation_layer(y_test, predictions):
    print("\n=== EVALUATION LAYER ===")

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    return mse, r2


# ============================================================
# 6. VISUALIZATION LAYER
# ============================================================
def visualization_layer(y_test, predictions):
    print("\n=== VISUALIZATION LAYER ===")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values (MLP)")
    plt.grid(True)
    plt.show()


# ============================================================
# MAIN SYSTEM WORKFLOW (CONNECTS ALL LAYERS)
# ============================================================
def main():
    # Layer 1
    X, y = data_input_layer()

    # Layer 2
    X_train, X_test, y_train, y_test = preprocessing_layer(X, y)

    # Layer 3
    model = model_training_layer(X_train, y_train)

    # Layer 4
    predictions = prediction_layer(model, X_test)

    # Layer 5
    evaluation_layer(y_test, predictions)

    # Layer 6
    visualization_layer(y_test, predictions)


# Run the system
if __name__ == "__main__":
    main()
 