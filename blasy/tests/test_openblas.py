import numpy as np
from numpy.testing import *
from blasy.openblas import dgemm, ddot


def test_ddot():

    a = np.array([0, 1, 2, 3], 'f8')
    b = np.array([0, 2, 3, 4], 'f8')
    assert_equal(ddot(a, b), np.dot(a, b))


def test_dgemm():
    A = np.array([[0, 1, 2], [3, 4, 5]], dtype='f8')
    B = np.array([[0, 1], [2, 3], [4, 5]], dtype='f8')
    C = np.zeros((2, 2))
    alpha = 2.
    beta = 3.
    dgemm(A, B, C, alpha, beta)

    C2 = np.zeros((2, 2))
    assert_array_almost_equal(C, alpha * np.dot(A, B) + beta * C2)
