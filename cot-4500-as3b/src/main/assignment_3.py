import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    aug_matrix = np.hstack([A, b.reshape(-1, 1)])
    
    # Partial pivoting
    for i in range(n):
        max_row = np.argmax(abs(aug_matrix[i:, i])) + i
        aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]
        
        for j in range(i+1, n):
            factor = aug_matrix[j][i] / aug_matrix[i][i]
            aug_matrix[j] = aug_matrix[j] - factor * aug_matrix[i]
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (aug_matrix[i][-1] - np.dot(aug_matrix[i][i+1:n], x[i+1:n])) / aug_matrix[i][i]
    
    return x

def lu_factorization(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    
    for k in range(n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return L, U

# Question 1
A1 = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]], dtype=float)
b1 = np.array([6, 0, -3], dtype=float)
solution_q1 = gaussian_elimination(A1, b1)

print(solution_q1)
print("\n")

# Question 2
A2 = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=float)
L2, U2 = lu_factorization(A2)
det_A2 = np.prod(np.diag(U2).astype(np.float64))

# Print exactly as shown in expected output
print(det_A2)
print("\n")
print(L2)
print("\n")
print(U2)

