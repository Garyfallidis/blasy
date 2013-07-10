import cython
import numpy as np
cimport numpy as cnp

ctypedef int blasint

cdef extern from "cblas.h" nogil:
    
    enum CBLAS_ORDER:
        CblasRowMajor = 101
        CblasColMajor = 102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans =111
        CblasTrans = 112
        CblasConjTrans = 113
        CblasConjNoTrans = 114
    enum CBLAS_UPLO:
        CblasUpper = 121
        CblasLower = 122
    enum CBLAS_DIAG:
        CblasNonUnit = 131
        CblasUnit = 132
    enum CBLAS_SIDE:
        CblasLeft = 141
        CblasRight = 142


    # Level 1
    double cblas_ddot(const blasint n,
                      const double * x,
                      const blasint incx,
                      const double * y,
                      const blasint incy)

   
    void cblas_dger(const CBLAS_ORDER order,
                    const blasint M,
                    const blasint N,
                    const double alpha,
                    const double * x,
                    const blasint incx,
                    const double * y,
                    const blasint incy,
                    double * A,
                    const blasint lda)

    # Level 2
    void cblas_dgemv(const CBLAS_ORDER order,
                     const CBLAS_TRANSPOSE trans,
                     const blasint m,
                     const blasint n,
                     const double alpha,
                     const double * a,
                     const blasint lda,
                     const double * x,
                     const blasint incx,
                     const double beta,
                     double * y,
                     const blasint incy)

    # Level 3
    void cblas_dgemm(CBLAS_ORDER order,
                     CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB,
                     blasint M,
                     blasint N,
                     blasint K,
                     double alpha,
                     double * A,
                     blasint lda,
                     double * B,
                     blasint ldb,
                     double beta,
                     double * C,
                     blasint ldc)
