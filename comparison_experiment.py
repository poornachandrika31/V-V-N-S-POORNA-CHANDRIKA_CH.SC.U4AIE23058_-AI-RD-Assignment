import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("xy_data.csv")
points = data[['x', 'y']].values

print("Dataset loaded successfully")
print("Total points:", len(points))

t = np.linspace(6, 60, 1500)

tree = cKDTree(points)

# ----------------------------
# Objective Function
# ----------------------------
def objective(params):
    theta_deg, M, X = params

    theta = np.deg2rad(theta_deg)

    x = (
        t*np.cos(theta)
        - np.exp(M*np.abs(t))*np.sin(0.3*t)*np.sin(theta)
        + X
    )

    y = (
        42
        + t*np.sin(theta)
        + np.exp(M*np.abs(t))*np.sin(0.3*t)*np.cos(theta)
    )

    generated = np.column_stack((x, y))

    distance, _ = tree.query(generated)

    return np.mean(distance)

# ----------------------------
# Parameter Bounds
# ----------------------------
bounds = [
    (0, 50),
    (-0.05, 0.05),
    (0, 100)
]

# =====================================================
# Nelder-Mead
# =====================================================

initial_guess = [20, 0.01, 40]

start = time.perf_counter()

nm_result = minimize(
    objective,
    initial_guess,
    method="Nelder-Mead"
)

nm_time = time.perf_counter() - start

# =====================================================
# Differential Evolution
# =====================================================

start = time.perf_counter()

de_result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=30,
    polish=True
)

de_time = time.perf_counter() - start

# =====================================================
# Print Comparison
# =====================================================

print("\n" + "="*65)
print("Optimization Comparison")
print("="*65)

print(f"{'Method':25s} {'L1 Error':>12s} {'Runtime(s)':>12s}")

print("-"*65)

print(f"{'Nelder-Mead':25s} {nm_result.fun:12.6f} {nm_time:12.3f}")

print(f"{'Differential Evolution':25s} {de_result.fun:12.6f} {de_time:12.3f}")

print("="*65)

print("\nEstimated Parameters")

print("\nNelder-Mead")

print(f"Theta : {nm_result.x[0]:.6f}")
print(f"M     : {nm_result.x[1]:.6f}")
print(f"X     : {nm_result.x[2]:.6f}")

print("\nDifferential Evolution")

print(f"Theta : {de_result.x[0]:.6f}")
print(f"M     : {de_result.x[1]:.6f}")
print(f"X     : {de_result.x[2]:.6f}")

print("\nConclusion:")

if abs(de_result.fun - nm_result.fun) < 1e-6:
    print("Both optimization methods converged to nearly identical solutions.")
    print("Differential Evolution was selected because it performs a global search")
    print("without requiring an initial parameter estimate.")
elif de_result.fun < nm_result.fun:
    print("Differential Evolution achieved a lower reconstruction error.")
else:
    print("Nelder-Mead achieved a lower reconstruction error.")