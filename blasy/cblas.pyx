cimport cblas

"""
How BLAS jargon makes sense
http://www.gnu.org/software/gsl/manual/html_node/BLAS-Support.html
"""

#Level 1
def ddot(double[::1] x, double[::1] y, int incx=1, int incy=1):
    """ dot(x, y)
    """

    cdef: 
        double * xp = &x[0]
        double * yp = &y[0]
        int M = x.shape[0]
        double res = 0    

    with nogil:        
        res = cblas_ddot(M, xp, incx, yp, incy)
    return res


def dger(double[::1] x, double[::1] y, double[:, ::1] A, double alpha=1, int incx=1, int incy=1):
    """ A = alpha*x*y' + A (outer product)
    """

    cdef:
        double * xp = &x[0]
        double * yp = &y[0]
        double * a = &A[0, 0]
        int M = A.shape[0]
        int N = A.shape[1]

    with nogil:
        cblas_dger(CblasRowMajor, M, N, alpha, xp, incx, yp, incy, a, M)


#Level 2
def dgemv(double[:, ::1] A, double[::1] x, double[::1] y, double alpha=1., 
    double beta=0., int incx=1, int incy=1):
    """ y = alpha*A*x + beta*y
    """
    cdef:
        double * xp = &x[0]
        double * yp = &y[0]
        double * a = &A[0, 0]
        int M = A.shape[0]
        int N = A.shape[1]

    with nogil:
        cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, a, N, xp, 
                    incx, beta, yp, incy)
    

#Level 3
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



