# AI Research & Development Assignment

## Objective

Estimate the unknown parameters (θ, M, X) of the given parametric curve using the provided dataset (`xy_data.csv`).

---

## Problem Statement

The curve is defined by:

\[
x=t\cos(\theta)-e^{M|t|}\sin(0.3t)\sin(\theta)+X
\]

\[
y=42+t\sin(\theta)+e^{M|t|}\sin(0.3t)\cos(\theta)
\]

where

- 0° < θ < 50°
- -0.05 < M < 0.05
- 0 < X < 100
- 6 ≤ t ≤ 60

The objective is to estimate θ, M and X such that the generated curve best matches the provided dataset.

---
## Why Differential Evolution?

The objective function is nonlinear and contains exponential and trigonometric components, making gradient-based optimization difficult. Differential Evolution is a global optimization algorithm that efficiently searches the parameter space without requiring an initial guess, making it well suited for this problem.

## Approach

1. Loaded the dataset using Pandas.
2. Generated uniformly sampled values of t between 6 and 60.
3. Implemented the parametric equations.
4. Used **Differential Evolution** from SciPy to estimate θ, M and X.
5. Used a KD-Tree to compute the nearest-neighbour L1 distance between the generated curve and the observed points.
6. Visualized the predicted and actual curves using Matplotlib.

---

## Libraries Used

- NumPy
- Pandas
- SciPy
- Matplotlib

---

## Estimated Parameters

| Parameter | Value |
|-----------|--------|
| θ | 29.999557° |
| M | 0.030000 |
| X | 54.999214 |

L1 Error:

0.021236

---

## Visualization

The predicted curve almost perfectly overlaps the observed data.

(Insert comparison.png here)

---

## Desmos Equation

```
(t*cos(0.523591)-e^(0.03*abs(t))*sin(0.3*t)*sin(0.523591)+54.999214,
42+t*sin(0.523591)+e^(0.03*abs(t))*sin(0.3*t)*cos(0.523591))
```

---

## Future Improvements

- Bayesian Optimization
- CMA-ES optimization
- Noise robust parameter estimation
- Multi-objective optimization
