cimport openblas


def dgemm(double[:, :] A, double[:, :] B, double[:, :] C, 
          double alpha=1.0, double beta=0.0):
    """ C = alpha*dot(A,B) + beta*C
    """

    cdef: 
        double * a = &A[0, 0]
        double * b = &B[0, 0]
        double * c = &C[0, 0]
        int M = A.shape[0]
        int N = B.shape[1]
        int K = B.shape[0]
        int lda = K
        int ldb = N
        int ldc = N

    with nogil:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 
                    alpha, a, lda, b, ldb, beta, c, ldc)


def ddot(double[:] a, double[:] b):

    cdef: 
        double * ap = &a[0]
        double * bp = &b[0]
        int M = a.shape[0]
        double res = 0    

    with nogil:        
        res = cblas_ddot(M, ap, 1, bp, 1)
    return res