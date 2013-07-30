cimport cython
cimport lapacke

DEF LAPACK_ROW_MAJOR = 101
DEF LAPACK_COL_MAJOR = 102

@cython.wraparound(False)
@cython.boundscheck(False)
def dgelsd(double[:, ::1] A, double[:, ::1] B, double[::1] s):
    """ computes the minimum-norm solution to a real linear least  squares
    problem:
                    minimize 2-norm(| b - A*x |)
    using the singular value decomposition (SVD) of A. A is an M-by-N
    matrix which may be rank-deficient.
    """
    cdef:
        lapack_int info
        lapack_int m = A.shape[0]
        lapack_int n = A.shape[1]
        lapack_int nrhs = B.shape[1]
        double * a = &A[0, 0]
        double * b = &B[0, 0]
        double * s_ = &s[0]
        lapack_int rank

    with nogil:
        info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, m, n, nrhs, a, n,
                              b, nrhs, s_, -1.0, &rank)

    return info




