import cython
import numpy as np
cimport numpy as cnp

ctypedef int blasint 

cdef extern from "cblas.h" nogil:
    enum CBLAS_ORDER:
        CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE:
        CblasNoTrans, CblasTrans, CblasConjTrans, CblasConjNoTrans    	

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


def dgemm(cnp.ndarray[double, ndim=2] A, cnp.ndarray[double, ndim=2] B,
          cnp.ndarray[double, ndim=2] C, double alpha=1.0, double beta=0.0):
    """ C = alpha*dot(A,B) + beta*C
    """

    cdef double * a = <double *> A.data
    cdef double * b = <double *> B.data
    cdef double * c = <double *> C.data

    cdef:
        int M = A.shape[0]
        int N = B.shape[1]
        int K = B.shape[0]
        int lda = K
        int ldb = N
        int ldc = N

    with nogil:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 
                alpha, a, lda, b, ldb, beta, c, ldc)

    

