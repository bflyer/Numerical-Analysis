import numpy as np

def cubic_spline_interpolation(x_data, y_data, dy0, dyn, eval_points):
    """
    三次样条插值实现（第一种边界条件）
    
    参数:
        x_data, y_data - 数据点坐标
        dy0, dyn - 两端点的一阶导数值
        eval_points - 需要计算的点
        
    返回:
        每个点的函数值、一阶导数和二阶导数
    """
    n = len(x_data) - 1  # 区间数
    h = np.diff(x_data)  # 计算步长
    
    # 初始化λ和μ
    lambda_ = np.zeros(n)
    mu = np.zeros(n)
    lambda_[0] = 1  # 边界条件
    mu[-1] = 1      # 边界条件
    
    for i in range(1, n):
        lambda_[i] = h[i] / (h[i-1] + h[i])
        mu[i-1] = 1 - lambda_[i]
    
    # 构建d向量
    d = np.zeros(n+1)
    d[0] = (6/h[0]) * ((y_data[1]-y_data[0])/h[0] - dy0)  # 左边界
    d[-1] = (6/h[-1]) * (dyn - (y_data[-1]-y_data[-2])/h[-1])  # 右边界
    
    for i in range(1, n):
        d[i] = 6 * (
            (y_data[i-1] - y_data[i])/(h[i-1]*(h[i-1]+h[i])) +
            (y_data[i+1] - y_data[i])/(h[i]*(h[i-1]+h[i]))
        )
    
    # 解三对角方程组（追赶法）
    M = np.zeros(n+1)
    b = 2 * np.ones(n+1)  # 主对角线元素
    m = np.zeros(n+1)      # 消元系数
    
    # 前向消元
    for i in range(1, n+1):
        m[i] = mu[i-1] / b[i-1]
        b[i] = b[i] - m[i] * lambda_[i-1]
        d[i] = d[i] - m[i] * d[i-1]
    
    # 回代求解
    M[-1] = d[-1] / b[-1]
    for i in range(n-1, -1, -1):
        M[i] = (d[i] - lambda_[i] * M[i+1]) / b[i]
    
    # 计算结果点
    results = []
    for x in eval_points:
        # 找到x所在的区间
        k = np.searchsorted(x_data, x) - 1
        k = max(0, min(k, n-1))  # 确保k在有效范围内
        
        # 计算相对位置
        dx = x - x_data[k]
        hk = h[k]
        
        # 计算函数值
        val = (
            M[k] * (x_data[k+1]-x)**3 / (6*hk) +
            M[k+1] * (x-x_data[k])**3 / (6*hk) +
            (y_data[k] - M[k]*hk**2/6) * (x_data[k+1]-x)/hk +
            (y_data[k+1] - M[k+1]*hk**2/6) * (x-x_data[k])/hk
        )
        
        # 计算一阶导数
        der = (
            -M[k] * (x_data[k+1]-x)**2 / (2*hk) +
            M[k+1] * (x-x_data[k])**2 / (2*hk) +
            (y_data[k+1] - y_data[k])/hk -
            hk * (M[k+1] - M[k])/6
        )
        
        # 计算二阶导数
        der2 = M[k] * (x_data[k+1]-x)/hk + M[k+1] * (x-x_data[k])/hk
        
        results.append((x, val, der, der2))
    
    return results

# 测试数据
x_data = [0.520, 3.1, 8.0, 17.95, 28.65, 39.62, 50.65, 78, 104.6, 156.6, 
          208.6, 260.7, 312.50, 364.4, 416.3, 468, 494, 507, 520]
y_data = [5.288, 9.4, 13.84, 20.20, 24.90, 28.44, 31.10, 35, 36.9, 36.6, 
          34.6, 31.0, 26.34, 20.9, 14.8, 7.8, 3.7, 1.5, 0.2]
dy0 = 1.86548
dyn = -0.046115
eval_points = [2, 30, 130, 350, 515]

# 计算并输出结果
results = cubic_spline_interpolation(x_data, y_data, dy0, dyn, eval_points)
print("x\t\tf(x)\t\tf'(x)\t\tf''(x)")
for x, val, der, der2 in results:
    print(f"{x:.1f}\t\t{val:.4f}\t\t{der:.4f}\t\t{der2:.4f}")
    
"""
x               f(x)            f'(x)           f''(x)
2.0             7.8252          1.5568          -0.2213
30.0            25.3862         0.3549          -0.0078
130.0           37.2138         -0.0104         -0.0014
350.0           22.4751         -0.1078         -0.0002
515.0           0.5427          -0.0899         0.0081
"""