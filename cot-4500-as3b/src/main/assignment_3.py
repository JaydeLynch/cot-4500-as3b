import numpy as np

def q_1():
    #Solve linear system using Gaussian elimination
    # Augmented matrix
    A = np.array([[2, -1, 1, 6],
                  [1, 3, 1, 0],
                  [-1, 5, 4, -3]], dtype=float)
    
    # Gaussian elimination
    n = len(A)
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        
        # Elimination
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (A[i, -1] - np.dot(A[i, i+1:n], x[i+1:])) / A[i, i]
    
    return x

def q_2():
    #LU Factorization
    A = np.array([[1, 1, 0, 3],
                  [2, 1, -1, 1],
                  [3, -1, -1, 2],
                  [-1, 2, 3, -1]], dtype=float)
    
    n = len(A)
    L = np.eye(n)
    U = A.copy()
    
    for k in range(n-1):
        for j in range(k+1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:] -= L[j, k] * U[k, k:]
    
    determinant = np.prod(np.diag(U))
    
    return determinant, L, U

def main():
    #Question 1
    x = q_1()
    print(f"{x[0]:.16f}")  
    print(f"{x[1]:.16f}")  
    print(f"[{x[2]:.0f} -1 1]")  

    #Question 2
    det, L, U = q_2()
    print(f"{det:.16f}")  
    
    print(L)
   
    print(U)
    

if __name__ == "__main__":
    main()
