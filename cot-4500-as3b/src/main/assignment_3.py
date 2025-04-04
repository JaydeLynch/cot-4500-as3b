{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
\
def question_1():\
    """Solve linear system using Gaussian elimination"""\
    # Augmented matrix\
    A = np.array([[2, -1, 1, 6],\
                  [1, 3, 1, 0],\
                  [-1, 5, 4, -3]], dtype=float)\
    \
    # Gaussian elimination\
    n = len(A)\
    for i in range(n):\
        # Partial pivoting\
        max_row = np.argmax(abs(A[i:, i])) + i\
        A[[i, max_row]] = A[[max_row, i]]\
        \
        # Elimination\
        for j in range(i+1, n):\
            factor = A[j, i] / A[i, i]\
            A[j, i:] -= factor * A[i, i:]\
    \
    # Back substitution\
    x = np.zeros(n)\
    for i in range(n-1, -1, -1):\
        x[i] = (A[i, -1] - np.dot(A[i, i+1:n], x[i+1:])) / A[i, i]\
    \
    return x\
\
def question_2():\
    """LU Factorization"""\
    A = np.array([[1, 1, 0, 3],\
                  [2, 1, -1, 1],\
                  [3, -1, -1, 2],\
                  [-1, 2, 3, -1]], dtype=float)\
    \
    n = len(A)\
    L = np.eye(n)\
    U = A.copy()\
    \
    for k in range(n-1):\
        for j in range(k+1, n):\
            L[j, k] = U[j, k] / U[k, k]\
            U[j, k:] -= L[j, k] * U[k, k:]\
    \
    determinant = np.prod(np.diag(U))\
    \
    return determinant, L, U\
\
def question_3():\
    """Check if matrix is diagonally dominant"""\
    A = np.array([[9, 0, 5, 2, 1],\
                  [3, 9, 1, 2, 1],\
                  [0, 1, 7, 2, 3],\
                  [4, 2, 3, 12, 2],\
                  [3, 2, 4, 0, 8]])\
    \
    diagonally_dominant = True\
    for i in range(len(A)):\
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])\
        if np.abs(A[i, i]) <= row_sum:\
            diagonally_dominant = False\
            break\
    \
    return diagonally_dominant\
\
def question_4():\
    """Check if matrix is positive definite"""\
    A = np.array([[2, 2, 1],\
                  [2, 3, 0],\
                  [1, 0, 2]])\
    \
    # Check if symmetric\
    if not np.allclose(A, A.T):\
        return False\
    \
    # Check all leading principal minors have positive determinant\
    for i in range(1, len(A)+1):\
        sub_matrix = A[:i, :i]\
        if np.linalg.det(sub_matrix) <= 0:\
            return False\
    \
    return True\
\
def main():\
    print("Question 1 Solution:")\
    x = question_1()\
    for val in x:\
        print(val)\
    \
    print("\\nQuestion 2:")\
    det, L, U = question_2()\
    print("a. Determinant:", det)\
    print("b. L matrix:")\
    print(L)\
    print("c. U matrix:")\
    print(U)\
    \
    print("\\nQuestion 3:")\
    dd = question_3()\
    print("Is diagonally dominant:", dd)\
    \
    print("\\nQuestion 4:")\
    pd = question_4()\
    print("Is positive definite:", pd)\
\
if __name__ == "__main__":\
    main()}