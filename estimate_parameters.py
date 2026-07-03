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

import os

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(10,6))

plt.scatter(
    points[:,0],
    points[:,1],
    s=8,
    alpha=0.6,
    color="royalblue",
    label=f"Actual Data ({len(points)} points)"
)

plt.plot(
    x_pred,
    y_pred,
    color="crimson",
    linewidth=2.5,
    label="Estimated Curve"
)

plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)

plt.title(
    "Parametric Curve Fitting using Differential Evolution\n"
    f"Theta = {theta:.4f}°    "
    f"M = {M:.6f}    "
    f"X = {X:.4f}    "
    f"Mean L1 Error = {result.fun:.6f}",
    fontsize=14
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

save_path = os.path.join("plots", "comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"\nPlot saved to: {save_path}")

plt.show()

print("\nResults saved successfully.")