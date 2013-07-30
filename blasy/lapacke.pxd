ctypedef int lapack_int


cdef extern from "lapacke.h" nogil:

    lapack_int LAPACKE_dgelsd(int matrix_order, lapack_int m, lapack_int n,
                              lapack_int nrhs, double * a, lapack_int lda,
                              double * b, lapack_int ldb, double * s, double rcond,
                              lapack_int * rank)
