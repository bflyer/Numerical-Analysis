import numpy as np

def qr_algorithm(A, max_iter=1000, tol=1e-6):
    """
    QR 算法计算矩阵的特征值
    
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
        # QR 分解
        Q, R = np.linalg.qr(A_k)
        
        # 更新矩阵
        A_k = R @ Q
        
        # 检查收敛 (非对角线元素是否足够小)
        off_diag = np.abs(A_k - np.diag(np.diag(A_k)))
        if np.max(off_diag) < tol:
            break
            
    return i, np.diag(A_k)


# 测试
A = np.array([[0.5, 0.5, 0.5, 0.5],
              [0.5, 0.5, -0.5, -0.5],
              [0.5, -0.5, 0.5, -0.5],
              [0.5, -0.5, -0.5, 0.5]])

steps, eigenvalues = qr_algorithm(A)
print(f"QR 算法共迭代 {steps} 次，得到特征值: {eigenvalues}")

# QR 算法共迭代 999 次，得到特征值: [0.5 0.5 0.5 0.5]