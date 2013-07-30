cimport lapacke

DEF LAPACK_ROW_MAJOR = 101
DEF LAPACK_COL_MAJOR = 102


def dgelsd(double[:, ::1] A, double[:, ::1] B, double[::1] x):
    """Ax = B
    """
    cdef: 
        lapack_int info
        lapack_int m = A.shape[0]
        lapack_int n = A.shape[1]
        lapack_int nrhs = B.shape[1]
        double * a = &A[0, 0]
        double * b = &B[0, 0]
        double * s = &x[0]
        lapack_int rank
    
    info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, m, n, nrhs, a, n,
                          b, nrhs, s, -1.0, &rank)

    return info




