import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree

# ----------------------------
# Load dataset
# ----------------------------
data = pd.read_csv("xy_data.csv")
points = data[['x', 'y']].values

print("Dataset loaded successfully")
print("Total points:", len(points))

# Uniform sampling of t
t = np.linspace(6, 60, 1500)

# KD Tree for nearest-neighbour search
tree = cKDTree(points)

# ----------------------------
# Objective Function
# ----------------------------
def objective(params):
    theta_deg, M, X = params

    theta = np.deg2rad(theta_deg)

    x = (
        t * np.cos(theta)
        - np.exp(M * np.abs(t))
        * np.sin(0.3 * t)
        * np.sin(theta)
        + X
    )

    y = (
        42
        + t * np.sin(theta)
        + np.exp(M * np.abs(t))
        * np.sin(0.3 * t)
        * np.cos(theta)
    )

    generated = np.column_stack((x, y))

    distance, _ = tree.query(generated)

    return np.mean(distance)

# ----------------------------
# Parameter Estimation
# ----------------------------
bounds = [
    (0, 50),       # theta
    (-0.05, 0.05), # M
    (0, 100)       # X
]

result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=30,
    polish=True
)

theta, M, X = result.x

print("\nEstimated Parameters")
print("---------------------")
print(f"Theta : {theta:.6f} degrees")
print(f"M     : {M:.6f}")
print(f"X     : {X:.6f}")
print(f"L1 Error : {result.fun:.6f}")

# ----------------------------
# Generate Final Curve
# ----------------------------
theta_rad = np.deg2rad(theta)

x_pred = (
    t * np.cos(theta_rad)
    - np.exp(M * np.abs(t))
    * np.sin(0.3 * t)
    * np.sin(theta_rad)
    + X
)

y_pred = (
    42
    + t * np.sin(theta_rad)
    + np.exp(M * np.abs(t))
    * np.sin(0.3 * t)
    * np.cos(theta_rad)
)

# ----------------------------
# Save Results
# ----------------------------
with open("results.txt", "w") as f:
    f.write(f"Theta = {theta:.6f} degrees\n")
    f.write(f"M = {M:.6f}\n")
    f.write(f"X = {X:.6f}\n")
    f.write(f"L1 Error = {result.fun:.6f}\n")

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8,6))

plt.scatter(
    points[:,0],
    points[:,1],
    s=10,
    label="Actual"
)

plt.plot(
    x_pred,
    y_pred,
    linewidth=2,
    label="Predicted"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Actual vs Predicted Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("plots/comparison.png")

plt.show()

print("\nResults saved successfully.")