cimport cython
from blasy.cblas cimport cblas_dgemv, cblas_dgemm, cblas_ddot, CblasRowMajor, CblasColMajor, CblasTrans, CblasNoTrans
import numpy as np
from numpy.testing import assert_array_almost_equal
from libc.math cimport sqrt, exp


class Nnet:

    def __init__(self):
        self.rng = np.random.mtrand.RandomState(1337)  # 1337 is the seed of the mersenne twister
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
        for i in xrange(2000):
            j = i % 2
            self.prop(j)
            self.backprop(j)
            self.update()


cdef class CNnet:
    cdef:
        public double [:, ::1] inp, W, V, targ, W2, C
        public double [::1] e1, e2, s1, s2
        public double lr
    
    def __init__(self):
        # 1337 is the seed of the mersenne twister
        rng = np.random.mtrand.RandomState(1337)  
        self.inp = rng.rand(2, 500)
        self.W = rng.rand(500, 400) - 0.5
        self.V = rng.rand(400, 25) - 0.5
        self.targ = rng.rand(2, 25)
        self.lr = 0.01

        self.e1 = np.zeros(self.W.shape[1])        
        self.e2 = np.zeros(self.V.shape[1])
        self.s1 = np.zeros(self.e1.shape[0])        
        self.s2 = np.zeros(self.e2.shape[0])

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

    


    def train(self):
        for i in xrange(2000):
            j = i % 2
            self.prop(j)
        
    def test_dgemm(self):

        cdef:
            double * a = &self.inp[0, 0]
            double * b = &self.W[0, 0]
            double * c = &self.C[0, 0]
            int M = self.inp.shape[0]
            int N = self.W.shape[1]
            int K = self.W.shape[0]

        with nogil:
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        M, N, K, 1.0, a, K, b, N, 0, c, N)
        

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cactF(double[:] a, double[:] b) nogil:
    cdef int M = a.shape[0]
    for m in range(M):
        b[m] = 1 / (1 + exp(- a[m]))

    return

