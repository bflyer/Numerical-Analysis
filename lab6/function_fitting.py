import numpy as np
import matplotlib.pyplot as plt

# Data preparation
t = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8])
y = np.array([33.4, 79.5, 122.65, 159.05, 189.15, 214.15, 238.65, 252.2, 
              267.55, 280.50, 296.65, 301.65, 310.40, 318.15, 325.15])
yln = np.log(y)

# ==================== Quadratic fitting y = a + bt + ct² ====================
# Construct design matrix A
A_quad = np.column_stack([np.ones_like(t), t, t**2])

# Normal equation (A^T A)x = A^T y
G_quad = A_quad.T @ A_quad
b_quad = A_quad.T @ y

# Solve equation (using Cholesky decomposition)
try:
    L_quad = np.linalg.cholesky(G_quad)  # G = L L^T
    temp = np.linalg.solve(L_quad, b_quad)
    x_quad = np.linalg.solve(L_quad.T, temp)
except np.linalg.LinAlgError:
    x_quad = np.linalg.solve(G_quad, b_quad)  # Fallback to standard solve if Cholesky fails

# Calculate fitted values
approx_quad = x_quad[2] * (t**2) + x_quad[1] * t + x_quad[0]

# Calculate RMSE
rmse_quad = np.sqrt(np.mean((approx_quad - y)**2))
print("Quadratic fit parameters (a, b, c):", x_quad)
print("Quadratic fit RMSE:", rmse_quad)

# ==================== Exponential fitting y = a e^(bt) ====================
# Construct design matrix A (linearized ln(y) = ln(a) + bt)
A_exp = np.column_stack([np.ones_like(t), t])

# Normal equation (A^T A)x = A^T ln(y)
G_exp = A_exp.T @ A_exp
b_exp = A_exp.T @ yln

# Solve equation (using Cholesky decomposition)
try:
    L_exp = np.linalg.cholesky(G_exp)
    temp = np.linalg.solve(L_exp, b_exp)
    x_exp = np.linalg.solve(L_exp.T, temp)
except np.linalg.LinAlgError:
    x_exp = np.linalg.solve(G_exp, b_exp)  # Fallback to standard solve if Cholesky fails

# Calculate fitted values
a_exp = np.exp(x_exp[0])
b_exp = x_exp[1]
approx_exp = a_exp * np.exp(b_exp * t)

# Calculate RMSE
rmse_exp = np.sqrt(np.mean((approx_exp - y)**2))
print("Exponential fit parameters (a, b):", a_exp, b_exp)
print("Exponential fit RMSE:", rmse_exp)

# ==================== Plotting ====================
plt.figure(figsize=(10, 6))
plt.scatter(t, y, color='red', label='Original data')
plt.plot(t, approx_quad, 'b-', label=f'Quadratic fit (RMSE={rmse_quad:.2f})')
plt.plot(t, approx_exp, 'g--', label=f'Exponential fit (RMSE={rmse_exp:.2f})')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Least Squares Curve Fitting Comparison')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('image/(6-1)curve_fitting_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

"""
Quadratic fit parameters (a, b, c): [-45.29423077  94.19429218  -6.12682612]
Quadratic fit RMSE: 5.683931823476435
Exponential fit parameters (a, b): 67.3937925784558 0.23898343793723434
Exponential fit RMSE: 56.52224402531059
"""