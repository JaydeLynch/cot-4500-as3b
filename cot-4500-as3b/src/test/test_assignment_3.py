import numpy as np
from main.assignment_3 import q_1, q_2, q_3, q_4

def test_q_1():
    x = q_1()
    expected = np.array([1.2446381, 1.25131659, 2.0])  # Approximate values
    assert np.allclose(x, expected, rtol=1e-5)

def test_q_2():
    det, L, U = q_2()
    assert np.isclose(det, -13.0)
    assert L.shape == (4, 4)
    assert U.shape == (4, 4)

def test_q_3():
    assert q_3() == True

def test_q_4():
    assert q_4() == True}
