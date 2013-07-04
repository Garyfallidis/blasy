import numpy as np
from openblas import dgemm
from time import time
"""
For OpenBlas
git clone https://github.com/xianyi/OpenBLAS.git
make
make PREFIX=. install
Check for include and library directories
"""


if __name__ == '__main__':

    A = np.array([[0, 1, 2], [3, 4, 5]], dtype='f8')
    B = np.array([[0, 1], [2, 3], [4, 5]], dtype='f8')
    C = np.zeros((2, 2))
    alpha = 2.
    beta = 3.

    """
    alpha*dot(A,B) + beta
    """
    dgemm(A, B, C, alpha, beta)
    print(C)
    print(alpha*np.dot(A, B)+beta)


    A = np.random.rand(500, 400).astype('f8')
    B = np.random.rand(400, 300).astype('f8')
    C = np.zeros((500, 300), 'f8')

    times = []
    for i in range(200):
        t0 = time()
        dgemm(A, B, C)
        t00 = time() - t0

        t1 = time()
        C2 = np.dot(A, B)
        t10 = time() - t1

        times.append(t10/t00)

    print('Average speedup after 200 runs %f' % np.array(times).mean())



