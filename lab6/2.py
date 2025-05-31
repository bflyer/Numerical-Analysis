# import numpy as np
# from scipy.interpolate import CubicSpline

# # 采样点坐标
# x = np.array([0.520, 3.1, 8.0, 17.95, 28.65, 39.62, 50.65, 78, 104.6, 156.6, 
#               208.6, 260.7, 312.50, 364.4, 416.3, 468, 494, 507, 520])
# y = np.array([5.288, 9.4, 13.84, 20.20, 24.90, 28.44, 31.10, 35, 36.9, 36.6, 
#               34.6, 31.0, 26.34, 20.9, 14.8, 7.8, 3.7, 1.5, 0.2])

# # 边界条件
# y_prime_0 = 1.86548
# y_prime_n = -0.046115

# # 创建三次样条插值对象
# cs = CubicSpline(x, y, bc_type=((1, y_prime_0), (1, y_prime_n)))

# # 需要计算的点
# points = np.array([2, 30, 130, 350, 515])

# # 计算函数值及其导数
# values = cs(points)
# first_derivatives = cs(points, 1)
# second_derivatives = cs(points, 2)

# # 输出结果
# print("点的坐标及函数值:")
# for i, point in enumerate(points):
#     print(f"x = {point}, y = {values[i]:.4f}, y' = {first_derivatives[i]:.4f}, y'' = {second_derivatives[i]:.4f}")
    
# """
# 点的坐标及函数值:
# x = 2, y = 7.8252, y' = 1.5568, y'' = -0.2213
# x = 30, y = 25.3862, y' = 0.3549, y'' = -0.0078
# x = 130, y = 37.2138, y' = -0.0104, y'' = -0.0014
# x = 350, y = 22.4751, y' = -0.1078, y'' = -0.0002
# x = 515, y = 0.5427, y' = -0.0899, y'' = 0.0081
# """
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 采样点坐标
x = np.array([0.520, 3.1, 8.0, 17.95, 28.65, 39.62, 50.65, 78, 104.6, 156.6, 
              208.6, 260.7, 312.50, 364.4, 416.3, 468, 494, 507, 520])
y = np.array([5.288, 9.4, 13.84, 20.20, 24.90, 28.44, 31.10, 35, 36.9, 36.6, 
              34.6, 31.0, 26.34, 20.9, 14.8, 7.8, 3.7, 1.5, 0.2])

# 边界条件
y_prime_0 = 1.86548
y_prime_n = -0.046115

# 创建三次样条插值对象
cs = CubicSpline(x, y, bc_type=((1, y_prime_0), (1, y_prime_n)))

# 需要计算的点
points = np.array([2, 30, 130, 350, 515])

# 计算函数值及其导数
values = cs(points)
first_derivatives = cs(points, 1)
second_derivatives = cs(points, 2)

# 输出结果
print("点的坐标及函数值:")
for i, point in enumerate(points):
    print(f"x = {point}, y = {values[i]:.4f}, y' = {first_derivatives[i]:.4f}, y'' = {second_derivatives[i]:.4f}")

# 绘制三次样条插值曲线
x_fine = np.linspace(min(x), max(x), 300)
y_fine = cs(x_fine)

plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_fine, label='Cubic Spline Interpolation', color='blue')
plt.scatter(x, y, color='red', label='Data Points', zorder=5)
plt.scatter(points, values, color='green', label='Calculated Points', zorder=5)

# 绘制一阶导数和二阶导数曲线
y_fine_first_derivative = cs(x_fine, 1)
y_fine_second_derivative = cs(x_fine, 2)

plt.plot(x_fine, y_fine_first_derivative, label='First Derivative', color='orange')
plt.plot(x_fine, y_fine_second_derivative, label='Second Derivative', color='purple')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Spline Interpolation and Derivatives')
plt.legend()
plt.grid(True)
plt.savefig('./cubic_spline_interpolation.png')
plt.show()