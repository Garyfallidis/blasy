import numpy as np
from numpy.testing import *
from blasy.cblas import dgemm, ddot, dgemv, dger
from blasy.lapacke import dgelsd


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


def test_dgelsd():
    A = np.array([[0.12, -8.19, 7.69, -2.26, -4.71],
                  [-6.91,  2.22, -5.12, -9.08,  9.96],
                  [-3.33, -8.94, -6.72, -4.40, -9.98],
                  [3.97,  3.33, -2.74, -7.92, -3.20]])
    B = np.array([[7.30,  0.47, -6.28],
                  [1.33,  6.58, -3.42],
                  [2.68, -1.71, 3.46],
                  [-9.62, -0.79, 0.41],
                  [0, 0, 0]])

    X_corr = np.array([[-0.69, -0.24, 0.06],
                      [-0.8, -0.08, 0.21],
                      [ 0.38, 0.12, -0.65],
                      [ 0.29, -0.24, 0.42],
                      [ 0.29, 0.35, -0.30]])


    s = np.zeros(A.shape[0])

    info = dgelsd(A, B, s)

    assert_array_almost_equal(B, X_corr, 2)

    A = np.random.rand(5, 3)
    B = np.random.rand(5, 1)
    s = np.zeros(A.shape[0])

    x = np.linalg.lstsq(A, B)[0]

    info = dgelsd(A, B, s)

    x2 = B[:3]

    assert_array_almost_equal(x, x2)


