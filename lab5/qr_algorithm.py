import numpy as np

def qr_eigenvalues(A, threshold=1e-8, max_iter=1000):
    """
    QR 迭代法计算 A 的所有特征值。
    
    input: 
        A : n * n 实矩阵
        threshold : float
            迭代停止阈值，当 A 中所有下三角元素绝对值
            小于 threshold 时认为收敛。
        max_iter : int
            最大迭代步数，防止不收敛时死循环。
    
    output: 
        eigs : list of float or complex
            A 的近似特征值。
    """
    dim = A.shape[0]
    
    for k in range(max_iter):
        # 检查下三角元素是否均小于 threshold
        off_diagonal = np.tril(A, -1)
        if np.all(np.abs(off_diagonal) < threshold):
            break
        
        # QR 分解
        Q, R = np.linalg.qr(A)
        
        # A_{k+1} = R @ Q
        A = R @ Q
        
        if k == max_iter - 1: 
            print("Can't converge")
        
    return np.diag(A)

if __name__ == "__main__":
    A = np.array([
        [ 0.5,  0.5,  0.5,  0.5],
        [ 0.5, 0.5, -0.5,  -0.5],
        [ 0.5, -0.5,  0.5, -0.5],
        [ 0.5,  -0.5, -0.5, 0.5]
    ])

    eigs = qr_eigenvalues(A)
    print("QR 迭代近似特征值：", eigs)
    
# 完全不收敛