import numpy as np

# ===================== 公式定义 =====================
def composite_simpson(f, a, b, n):
    if n % 2 == 1:
        raise ValueError("Simpson 要求 n 为偶数")
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    fx = f(x)
    return h/3 * (fx[0] + fx[-1] + 4*fx[1:-1:2].sum() + 2*fx[2:-1:2].sum())

def composite_gauss2(f, a, b, n):
    h = (b-a)/n
    xi = a + np.arange(n)*h
    xm = xi + h/2
    # 两个 Gauss 点偏移
    d = h/(2*np.sqrt(3))
    x1 = xm - d
    x2 = xm + d
    return (h/2) * np.sum(f(x1) + f(x2))

def compute_n(a, b, h_max):
    # 计算子区间数 n 并向上取偶数
    n = int(np.ceil((b-a)/h_max))
    if n % 2 == 1:
        n += 1
    return n

tol = 0.5e-8   # 目标误差
a1, b1 = 1, 2  # ln2 积分区间
a2, b2 = 0, 1  # pi 积分区间

# ===================== 区间数量计算 =====================
# —— ln2 部分 ——
# 先估算 f^{(4)} 在 [0,1] 上的最大值 M4：
M4 = 24.0
# 误差上界: E <= (b-a)/coefficient * M4 * h^4 => h_max = (coefficient*tol/M4/(b-a))**0.25
h_max_ln2_simpson = (180*tol/M4/(b2-a2))**0.25
h_max_ln2_gauss = (4320*tol/M4/4/(b2-a2))**0.25
n_ln2_simpson = compute_n(a1, b1, h_max_ln2_simpson)
n_ln2_gauss = compute_n(a1, b1, h_max_ln2_gauss)

# —— π 部分 ——
# 先估算 f^{(4)} 在 [0,1] 上的最大值 M4：
M4 = 24.0
# 误差上界: E <= 4 * (b-a)/coefficient * M4 * h^4 => h_max = (coefficient*tol/M4/4/(b-a))**0.25
h_max_pi_simpson = (180*tol/M4/4/(b2-a2))**0.25
h_max_pi_gauss = (4320*tol/M4/4/(b2-a2))**0.25
n_pi_simpson = compute_n(a2, b2, h_max_pi_simpson)
n_pi_gauss = compute_n(a2, b2, h_max_pi_gauss)

# ===================== 数值积分 =====================
ln2_simpson = composite_simpson(lambda x: 1/x, a1, b1, n_ln2_simpson)
pi_simpson = composite_simpson(lambda x: 4/(1+x**2), a2, b2, n_pi_simpson)
ln2_gauss = composite_gauss2(lambda x: 1/x, a1, b1, n_ln2_gauss)
pi_gauss = composite_gauss2(lambda x: 4/(1+x**2), a2, b2, n_pi_gauss)

# ===================== 误差计算 =====================
ln2_exact = np.log(2)
pi_exact  = np.pi

err_ln2_simpson = abs(ln2_simpson - ln2_exact)
err_pi_simpson = abs(pi_simpson - pi_exact)
err_ln2_gauss = abs(ln2_gauss - ln2_exact)
err_pi_gauss = abs(pi_gauss - pi_exact)

# ===================== 输出结果 =====================
print(f"Simpson ln2: n={n_ln2_simpson}, approx={ln2_simpson:.12f}, error={err_ln2_simpson:.2e}")
print(f"Simpson  π: n={n_pi_simpson}, approx={pi_simpson:.12f}, error={err_pi_simpson:.2e}")
print(f"Gauss  ln2: n={n_ln2_gauss}, approx={ln2_gauss:.12f}, error={err_ln2_gauss:.2e}")
print(f"Gauss   π: n={n_pi_gauss}, approx={pi_gauss:.12f}, error={err_pi_gauss:.2e}")

"""
Simpson ln2: n=72, approx=0.693147181722, error=1.16e-09
Simpson  π: n=102, approx=3.141592653590, error=3.51e-14
Gauss  ln2: n=34, approx=0.693147179586, error=9.74e-10
Gauss   π: n=46, approx=3.141592653590, error=4.62e-14
"""