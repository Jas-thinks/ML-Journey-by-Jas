from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Input numbers (features)
X = np.array([[1], [2], [3], [4], [5]])
# Output numbers (labels) = double the input
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict for a range of values
X_test = np.array([[0], [1], [2], [3], [4], [5], [6], [10]])
y_pred = model.predict(X_test)

# Print some predictions
print("Prediction for 6:", model.predict([[6]]))
print("Prediction for 10:", model.predict([[10]]))

# Plot
plt.scatter(X, y, color="blue", label="Data Points")     # original data
plt.plot(X_test, y_pred, color="red", label="Model Line") # ML learned line
plt.xlabel("Input (X)")
plt.ylabel("Output (y)")
plt.title("Linear Regression - Learn Double the Number")
plt.legend()
plt.show()
