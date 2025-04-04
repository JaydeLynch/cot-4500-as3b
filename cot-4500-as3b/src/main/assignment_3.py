import numpy as np

def question1():
    # Solve system using Gaussian elimination
    A = np.array([[2, -1, 1],
                  [1, 3, 1],
                  [-1, 5, 4]], dtype=float)
    b = np.array([6, 0, -3], dtype=float)
    
    # Augmented matrix
    aug = np.column_stack((A, b))
    
    # Forward elimination
    n = len(b)
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(abs(aug[i:, i])) + i
        aug[[i, max_row]] = aug[[max_row, i]]
        
        # Elimination
        for j in range(i+1, n):
            factor = aug[j, i] / aug[i, i]
            aug[j, i:] -= factor * aug[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (aug[i, -1] - np.dot(aug[i, i+1:n], x[i+1:n])) / aug[i, i]
    
    print(x[0])
    print(x[1])
    print(x)

def question2():
    # LU factorization
    A = np.array([[1, 1, 0, 3],
                  [2, 1, -1, 1],
                  [3, -1, -1, 2],
                  [-1, 2, 3, -1]], dtype=float)
    
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    
    for k in range(n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    # Determinant
    det = np.prod(np.diag(U))
    print(det)
    
    # L matrix
    print(L)
    
    # U matrix
    print(U)

def question3():
    # Check diagonal dominance
    A = np.array([[9, 0, 5, 2, 1],
                  [3, 9, 1, 2, 1],
                  [0, 1, 7, 2, 3],
                  [4, 2, 3, 12, 2],
                  [3, 2, 4, 0, 8]])
    
    diag = np.diag(np.abs(A))
    off_diag = np.sum(np.abs(A), axis=1) - diag
    is_dom = np.all(diag >= off_diag)
    print(is_dom)

def question4():
    # Check positive definite
    A = np.array([[2, 2, 1],
                  [2, 3, 0],
                  [1, 0, 2]])
    
    # Check if symmetric
    if not np.allclose(A, A.T):
        print(False)
        return
    
    # Check if all eigenvalues are positive
    eigvals = np.linalg.eigvals(A)
    is_pos_def = np.all(eigvals > 0)
    print(is_pos_def)

if __name__ == "__main__":
    question1()
    question2()
    question3()
    question4()
   
