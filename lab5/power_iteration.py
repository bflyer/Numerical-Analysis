import numpy as np
def power_iteration(A, threshold, max_iter=1000):
    """
    input: A, threshold
    output: x1, lambda1
    """
    dim = A.shape[0]
    v = np.ones(dim) 
    u = v / np.linalg.norm(v)
    lambda_old = 0
    
    for _ in range(max_iter):
        v = A @ u
        lambda_new = v[np.argmax(np.abs(v))]
        u = v / np.linalg.norm(v)
        if np.abs(lambda_new - lambda_old) < threshold:
            break
        lambda_old = lambda_new
    
    return u, lambda_new

if __name__ == "__main__":
    threshold = 1e-5
    A = np.array([
        [5, -4, 1],
        [-4, 6, -4],
        [1, -4, 7]
    ])
    
    B = np.array([
        [25, -41, 10, -6],
        [-41, 68, -17, 10],
        [10, -17, 5, -3],
        [-6, 10, -3, 2]
    ])
    
    x_A, lambda_A = power_iteration(A, threshold)
    x_B, lambda_B = power_iteration(B, threshold)
    
    print(f"x_A = {x_A}, lambda_A = {lambda_A}")
    print(f"x_B = {x_B}, lambda_B = {lambda_B}")
    
"""
x_A = [ 0.4497837  -0.66731639  0.59361895], 
lambda_A = -8.17750578617666

x_B = [-0.50156506  0.83044375 -0.2085536   0.12369746], 
lambda_B = 81.8167283711789
"""