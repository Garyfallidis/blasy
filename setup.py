from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
# numpy.showconfig() gives information about an available LAPACK
import os


fullpath = os.path.realpath(__file__)
dname, filename = os.path.split(fullpath)


include_dirs = [os.path.join(dname, 'include'), np.get_include()]
library_dirs = [os.path.join(dname, 'lib')]

ext_modules=[ 
    Extension("openblas",
              ["openblas.pyx"], 
              extra_objects=[os.path.join(dname, 'lib', 'libopenblas.a')],
              library_dirs=library_dirs,
              include_dirs=include_dirs)
]

setup(
  name = 'OpenBLAS wrapping demo',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)