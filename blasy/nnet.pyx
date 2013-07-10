# cython: profile=True
cimport cython
from blasy.cblas cimport cblas_dgemv, cblas_dgemm, cblas_ddot, \
                         cblas_dger, CblasRowMajor, CblasColMajor, \
                         CblasTrans, CblasNoTrans
import numpy as np
from numpy.testing import assert_array_almost_equal
from libc.math cimport sqrt, exp
#from cython.parallel import prange


class Nnet:

    def __init__(self):
        # 1337 is the seed of the mersenne twister
        self.rng = np.random.mtrand.RandomState(1337)  
        self.input = self.rng.rand(2, 500)
        self.W = self.rng.rand(500, 400) - 0.5
        self.V = self.rng.rand(400, 25) - 0.5
        self.actF = lambda x: 1 / (1 + np.exp(-x))
        self.dActF = lambda s: s * (1 - s)
        self.target = self.rng.rand(2, 25)
        self.lr = 0.01

    def prop(self, i):
        # std prop : ei is at neuron entry, si at its exit
        self.e1 = np.dot(self.input[i], self.W)
        self.s1 = self.actF(self.e1)
        self.e2 = np.dot(self.s1, self.V)
        self.s2 = self.actF(self.e2)

    def backprop(self, i):
        ds2 = self.s2 - self.target[i]
        de2 = self.dActF(self.s2) * ds2
        ds1 = np.dot(self.V, de2)
        de1 = self.dActF(self.s1) * ds1
        self.dV = np.outer(self.s1, de2)
        self.dW = np.outer(self.input[i], de1)

    def update(self):
        self.W -= self.lr * self.dW
        self.V -= self.lr * self.dV

    def train(self):        
        for i in range(2000):
            j = i % 2
            self.prop(j)
            self.backprop(j)
            self.update()


cdef class CNnet:
    cdef:
        public double [:, ::1] inp, W, V, targ, W2, dV, dW
        public double [::1] e1, e2, de1, de2, s1, s2, ds1, ds2 
        double [::1] tmp_s1, tmp_s2
        public double lr
    
    def __init__(self):
        # 1337 is the seed of the mersenne twister
        rng = np.random.mtrand.RandomState(1337)  
        
        # Initialize secondary variables
        self.inp = rng.rand(2, 500)
        self.W = rng.rand(500, 400) - 0.5
        self.V = rng.rand(400, 25) - 0.5
        self.targ = rng.rand(2, 25)

        self.lr = 0.01
                
        self.e1 = np.zeros(self.W.shape[1])        
        self.e2 = np.zeros(self.V.shape[1])

        e1_sh = self.e1.shape[0]
        e2_sh = self.e2.shape[0] 

        self.de1 = np.zeros(e1_sh)        
        self.de2 = np.zeros(e2_sh)        
        
        self.ds1 = np.zeros(e1_sh)        
        self.ds2 = np.zeros(e2_sh)
        
        self.s1 = np.zeros(e1_sh)        
        self.s2 = np.zeros(e2_sh)
        
        self.ds1 = np.zeros(e1_sh)        
        self.ds2 = np.zeros(e2_sh)

        self.dV = np.zeros((400, 25))
        self.dW = np.zeros((500, 400))
        
        # Initialize temporary variables
        self.tmp_s1 = np.zeros(self.e1.shape[0])        
        self.tmp_s2 = np.zeros(self.e2.shape[0])
    

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def prop(self, int i):

        cdef:
            int M = self.W.shape[0]
            int N = self.W.shape[1]
            int K = self.V.shape[0]
            int L = self.V.shape[1]

        with nogil:
            cblas_dgemv(CblasRowMajor, CblasTrans, M, N, 1., 
                        &self.W[0, 0], N, &self.inp[i, 0], 
                        1, 0., &self.e1[0], 1)
            #assert_array_almost_equal(np.dot(np.asarray(self.inp[i]), np.asarray(self.W)), np.asarray(self.e1))
            cactF(self.e1, self.s1)
            cblas_dgemv(CblasRowMajor, CblasTrans, K, L, 1., 
                        &self.V[0, 0], L, &self.s1[0], 
                        1, 0., &self.e2[0], 1)
            cactF(self.e2, self.s2)        

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def backprop(self, int i):
        cdef:
            int K = self.V.shape[0]
            int L = self.V.shape[1]
            int M = self.dV.shape[0]
            int N = self.dV.shape[1]
            int P = self.dW.shape[0]
            int O = self.dW.shape[1]
            
        with nogil:
            #ds2 = self.s2 - self.target[i]
            sub(self.s2, &self.targ[i, 0], self.ds2)
            #de2 = self.dActF(self.s2) * ds2
            dactF(self.s2, self.tmp_s2)
            mul(self.tmp_s2, self.ds2, self.de2)
            #ds1 = np.dot(self.V, de2)
            cblas_dgemv(CblasRowMajor, CblasNoTrans, K, L, 1., 
                        &self.V[0, 0], L, &self.de2[0], 
                        1, 0., &self.ds1[0], 1)
            #de1 = self.dActF(self.s1) * ds1
            dactF(self.s1, self.tmp_s1)
            mul(self.tmp_s1, self.ds1, self.de1)
            #self.dV = np.outer(self.s1, de2)
            cblas_dger(CblasRowMajor, M, N, 1., &self.s1[0], 1, &self.de2[0], 1, 
                       &self.dV[0, 0], N)
            #self.dW = np.outer(self.input[i], de1)
            cblas_dger(CblasRowMajor, P, O, 1., &self.inp[i, 0], 1, &self.de1[0], 1, 
                       &self.dW[0, 0], O)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def update(self):

        cdef: 
            int M = self.dV.shape[0] * self.dV.shape[1]
            int P = self.dW.shape[0] * self.dW.shape[1]
            double lr = self.lr

        with nogil:            
            #self.W -= self.lr * self.dW
            subsm(&self.W[0, 0], &self.dW[0, 0], P, lr)
            #self.V -= self.lr * self.dV
            subsm(&self.V[0, 0], &self.dV[0, 0], M, lr)        

    def train(self):
        cdef int i, j
        #with nogil:
        for i in range(2000):
            j = i % 2
            self.prop(j)
            self.backprop(j)
            self.update()


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cactF(double[:] a, double[:] b) nogil:
    cdef int M = a.shape[0]
    for m in range(M):
        b[m] = 1 / (1 + exp(- a[m]))
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dactF(double[:] a, double[:] b) nogil:
    cdef int M = a.shape[0]
    for m in range(M):
        b[m] = a[m] * (1 - a[m])
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sub(double[:] a, double * b, double[:] c) nogil:
    cdef int M = a.shape[0]
    for m in range(M):
        c[m] = a[m] - b[m]
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mul(double[:] a, double[:] b, double[:] c) nogil:
    """ Element wise multiplication
    """
    cdef int M = a.shape[0]
    for m in range(M):
        c[m] = a[m] * b[m]
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void subsm(double * A, double * B, int size, double value) nogil:    
    for m in range(size):        
        A[m] = A[m] - value * B[m]
    return