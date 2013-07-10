import numpy as np
from numpy.testing import *
from blasy.cblas import dgemm, ddot, dgemv, dger


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


def test_dgemv():
    A = np.array([[0, 1], [2, 3], [4, 5]], dtype='f8')
    x = np.array([0, 1], 'f8')
    y = np.zeros(A.shape[0])

    dgemv(A, x, y)
    assert_array_almost_equal(y, np.dot(A, x))

    A = np.array([[0, 1], [2, 3], [4, 5]], dtype='f8')
    A = np.ascontiguousarray(A.T)
    x = np.array([0, 1, 2], 'f8')
    y = np.array([0, 0], 'f8')
    dgemv(A, x, y)
    assert_array_almost_equal(y, np.dot(A, x))


def test_dger():
    x = np.array([0, 1, 2, 3], 'f8')
    y = np.array([0, 2, 3, 4], 'f8')

    A = np.zeros((4, 4))

    dger(x, y, A)
    assert_array_almost_equal(A, np.dot(x[:, None], y[None, :]))

    x = np.array([0, 1, 2, 3], 'f8')
    y = np.array([0, 2, 3], 'f8')

    A = np.zeros((4, 3))
    dger(x, y, A)
    assert_array_almost_equal(A, np.dot(x[:, None], y[None, :]))