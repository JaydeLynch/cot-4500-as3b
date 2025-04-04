import numpy as np
from assignment_3 import gaussian_elimination, lu_factorization

def test_gaussian_elimination():
    # Test case from Question 1
    A = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]], dtype=float)
    b = np.array([6, 0, -3], dtype=float)
    expected_solution = np.array([2, -1, 1], dtype=float)
    computed_solution = gaussian_elimination(A, b)
    
    # Check if the solution matches (allowing for small floating-point errors)
    assert np.allclose(computed_solution, expected_solution, rtol=1e-10), \
        f"Gaussian elimination failed: Expected {expected_solution}, got {computed_solution}"

def test_lu_factorization():
    # Test case from Question 2
    A = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=float)
    L, U = lu_factorization(A)
    
    # Check if LU decomposition reconstructs the original matrix (A ≈ L @ U)
    reconstructed_A = np.dot(L, U)
    assert np.allclose(A, reconstructed_A, rtol=1e-10), \
        "LU factorization failed: L @ U does not reconstruct A"
    
    # Check if L is lower triangular with 1's on diagonal
    assert np.allclose(L, np.tril(L)), "L is not lower triangular"
    assert np.all(np.diag(L) == 1), "L does not have 1's on the diagonal"
    
    # Check if U is upper triangular
    assert np.allclose(U, np.triu(U)), "U is not upper triangular"

def test_determinant():
    # Test determinant computation from LU factorization (Question 2)
    A = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=float)
    L, U = lu_factorization(A)
    computed_det = np.prod(np.diag(U))
    expected_det = 39.0  # Exact value (floating-point may give 38.999...)
    
    # Allow slight floating-point deviation
    assert np.isclose(computed_det, expected_det, rtol=1e-10), \
        f"Determinant incorrect: Expected ~{expected_det}, got {computed_det}"

if __name__ == "__main__":
    test_gaussian_elimination()
    test_lu_factorization()
    test_determinant()
