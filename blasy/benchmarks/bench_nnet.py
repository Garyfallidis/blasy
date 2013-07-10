from numpy.testing import assert_array_almost_equal as aaae
from blasy.nnet import Nnet, CNnet, YNnet
import timeit as ti


# nnet = Nnet()
# nnet.prop(0)

# cnnet = CNnet()
# cnnet.prop(0)

# aaae(nnet.e1, np.asarray(cnnet.e1))
# aaae(nnet.s1, np.asarray(cnnet.s1))

# nnet.backprop(0)
# cnnet.backprop(0)

# aaae(nnet.dW, np.asarray(cnnet.dW))
# aaae(nnet.dV, np.asarray(cnnet.dV))

# nnet.update()
# cnnet.update()

# aaae(nnet.W, np.asarray(cnnet.W))
# aaae(nnet.V, np.asarray(cnnet.V))

# New version

nnet = Nnet()
nnet.train()

ynnet = YNnet()
ynnet.train()

#aaae(nnet.e1, np.asarray(ynnet.e1))
#aaae(nnet.s1, np.asarray(ynnet.s1))

# nnet.backprop(0)
# ynnet.backprop(0)

# aaae(nnet.dW, np.asarray(ynnet.dW))
# aaae(nnet.dV, np.asarray(ynnet.dV))

# nnet.update()
# ynnet.update()

aaae(nnet.W, np.asarray(ynnet.W))
aaae(nnet.V, np.asarray(ynnet.V))