import cython
import numpy as np
cimport numpy as cnp

ctypedef int blasint 

cdef extern from "cblas.h" nogil:
    enum CBLAS_ORDER:
        CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE:
        CblasNoTrans, CblasTrans, CblasConjTrans, CblasConjNoTrans    	
    enum CBLAS_UPLO:
        CblasUpper, CblasLower
    enum CBLAS_DIAG:
        CblasNonUnit, CblasUnit
    enum CBLAS_SIDE:
        CblasLeft, CblasRight


    # Level 1
    double cblas_ddot(blasint n, 
                      double *x, 
                      blasint incx, 
                      double *y, 
                      blasint incy)

    # Level 2


    # Level 3
    void cblas_dgemm(CBLAS_ORDER order, 
                     CBLAS_TRANSPOSE TransA, 
                     CBLAS_TRANSPOSE TransB, 
                     blasint M, 
                     blasint N, 
                     blasint K,
                     double alpha, 
                     double *A, 
                     blasint lda,
                     double *B, 
                     blasint ldb, 
                     double beta, 
                     double *C, 
                     blasint ldc)

