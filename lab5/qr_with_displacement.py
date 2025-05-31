import numpy as np
def shifted_qr_algorithm(A, max_iter=1000, tol=1e-6):
    """
    带位移的 QR 算法计算矩阵的特征值
    
    参数:
        A: 输入矩阵 (n x n)
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        
    返回:
        特征值数组
    """
    n = A.shape[0]
    A_k = np.copy(A)
    
    for i in range(max_iter):
        # 位移取右下角元素
        s = A_k[-1, -1]
        
        # 对 A - sI 进行 QR 分解
        Q, R = np.linalg.qr(A_k - s * np.eye(n))
        
        # 更新矩阵
        A_k = R @ Q + s * np.eye(n)
        
        # 检查收敛
        off_diag = np.abs(A_k - np.diag(np.diag(A_k)))
        if np.max(off_diag) < tol:
            break
            
    return i, np.diag(A_k)

A = np.array([[0.5, 0.5, 0.5, 0.5],
              [0.5, 0.5, -0.5, -0.5],
              [0.5, -0.5, 0.5, -0.5],
              [0.5, -0.5, -0.5, 0.5]])

# 测试
steps, eigenvalues_shifted = shifted_qr_algorithm(A)
print(f"带位移 QR 算法共迭代 {steps} 步，得到特征值: {eigenvalues_shifted}")

# 带位移 QR 算法共迭代 3 步，得到特征值: [-1.  1.  1.  1.]