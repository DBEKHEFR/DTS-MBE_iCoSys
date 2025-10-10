import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Load the data
df = pd.read_csv(
    "SessionTest_Encoder.csv", skiprows=8, usecols=lambda x: "Unnamed" not in x
)

# df.head()

"""
# Script to make sure it running python jobs on the cluster works
plt.figure(figsize=(12, 6))
for column in df.columns[1:]:
    plt.plot(df['time'], df[column], label=column)
plt.xlabel('Time (s)')
plt.ylabel('Signal Value')
plt.title('Time Series Data of Robot Coordinates')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('time_series_plot.png')
plt.close()
"""

# Prepare the data for Gaussian Process modeling
X = df[["time"]].values
y = df[["f1\\s1", "f2\\s2", "f3\\s3"]].values

# Define the Gaussian Process model with an RBF kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Testing the training of the model on the first coordinate
X_train = X
y_train = y[:, 0]
gp.fit(X_train, y_train)
# Predict using the model
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred, sigma = gp.predict(X_pred, return_std=True)

# Plot the prediction
plt.figure(figsize=(12, 6))
plt.plot(X_train, y_train, "b.", markersize=10, label="Observations")
plt.plot(X_pred, y_pred, "r-", label="Prediction")
plt.fill_between(
    X_pred.flatten(),
    y_pred - 1.96 * sigma,
    y_pred + 1.96 * sigma,
    alpha=0.2,
    color="k",
    label="95% confidence interval",
)
plt.xlabel("Time (s)")
plt.ylabel("Signal Value")
plt.title("Gaussian Process Regression on Robot Coordinates")
plt.legend()
plt.grid(True)
plt.savefig("gaussian_process_plot.png")
plt.close()


print("Script ran successfully")
