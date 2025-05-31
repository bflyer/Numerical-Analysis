import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from numpy.linalg import lstsq

# 数据点
t1 = np.array([1, 1.5, 2, 2.5, 3.0, 3.5, 4])
y1 = np.array([33.40, 79.50, 122.65, 159.05, 189.15, 214.15, 238.65])

t2 = np.array([4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8])
y2 = np.array([252.2, 267.55, 280.50, 296.65, 301.65, 310.40, 318.15, 325.15])

# 合并数据
t = np.concatenate((t1, t2))
y = np.concatenate((y1, y2))

# 二次函数拟合
def quadratic_func(t, a, b, c):
    return a + b * t + c * t**2

params_quad, _ = curve_fit(quadratic_func, t, y)
a_quad, b_quad, c_quad = params_quad
y_pred_quad = quadratic_func(t, *params_quad)

# 指数函数拟合
def exp_func(t, a, b):
    return a * np.exp(b * t)

params_exp, _ = curve_fit(exp_func, t, y)
a_exp, b_exp = params_exp
y_pred_exp = exp_func(t, *params_exp)

# 计算统计指标
mse_quad = mean_squared_error(y, y_pred_quad)
r2_quad = r2_score(y, y_pred_quad)

mse_exp = mean_squared_error(y, y_pred_exp)
r2_exp = r2_score(y, y_pred_exp)

# 输出统计指标
print("Quadratic Fit - MSE: {:.2f}, R-squared: {:.2f}".format(mse_quad, r2_quad))
print("Exponential Fit - MSE: {:.2f}, R-squared: {:.2f}".format(mse_exp, r2_exp))

# 绘制数据点和拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(t, y, color='red', label='Data Points')
plt.plot(t, quadratic_func(t, *params_quad), label='Quadratic Fit', color='blue')
plt.plot(t, exp_func(t, *params_exp), label='Exponential Fit', color='green')
plt.xlabel('Time (t)')
plt.ylabel('Measurement (y)')
plt.title('Curve Fitting')
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig('curve_fitting.png')

plt.show()

# 输出拟合参数
print("Quadratic Fit Parameters: a = {:.2f}, b = {:.2f}, c = {:.2f}".format(a_quad, b_quad, c_quad))
print("Exponential Fit Parameters: a = {:.2f}, b = {:.2f}".format(a_exp, b_exp))

"""
Quadratic Fit - MSE: 32.31, R-squared: 1.00
Exponential Fit - MSE: 1444.39, R-squared: 0.81
Quadratic Fit Parameters: a = -45.29, b = 94.19, c = -6.13
Exponential Fit Parameters: a = 108.05, b = 0.15
"""