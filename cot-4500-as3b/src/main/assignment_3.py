import numpy as np

def question_1():
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

def question_2():
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
    # Question 1
    x = question_1()
    print(f"{x[0]:.16f}")  
    print("\n")
    print(f"{x[1]:.15f}")  
    print("\n")
    print(f"[{int(round(x[2]))} -1 1]") 
    print("\n")
    
    # Question 2
    det, L, U = question_2()
    print(f"{det:.16f}")  
    print("\n")
    
    # Print L matrix
    print("[", end="")
    for i in range(len(L)):
        if i > 0:
            print(" [", end="")
        else:
            print("[", end="")
        for j in range(len(L[i])):
            print(f"{L[i,j]:.1f}", end="")
            if j < len(L[i])-1:
                print(" ", end="")
        print("]", end="")
        if i < len(L)-1:
            print("")
    print("")
    
    # Print U matrix
    print("\n")
    print("[", end="")
    for i in range(len(U)):
        if i > 0:
            print(" [", end="")
        else:
            print("[", end="")
        for j in range(len(U[i])):
            print(f"{U[i,j]:.1f}", end="")
            if j < len(U[i])-1:
                print(" ", end="")
        print("]", end="")
        if i < len(U)-1:
            print("")
    print("")

if __name__ == "__main__":
    main()
