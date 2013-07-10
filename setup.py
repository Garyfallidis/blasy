from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
# numpy.showconfig() gives information about an available LAPACK
import os
from os.path import join as pjoin


fullpath = os.path.realpath(__file__)
dname, filename = os.path.split(fullpath)


include_dirs = [pjoin(dname, 'include'), np.get_include()]
library_dirs = [pjoin(dname, 'lib')]

ext_modules=[
    Extension('blasy.cblas',
              [pjoin('blasy','cblas.pyx')], 
              #extra_objects=[pjoin(dname, 'lib', 'libopenblas.a')],
              library_dirs=library_dirs,
              include_dirs=include_dirs,
              extra_compile_args=['-fopenmp'],    
              extra_link_args=['-lopenblas', '-fopenmp'],              
              runtime_library_dirs=library_dirs),
    Extension('blasy.nnet',
             [pjoin('blasy','nnet.pyx')],
             library_dirs=library_dirs,
             include_dirs=include_dirs,
             extra_compile_args=['-fopenmp'],    
             extra_link_args=['-lopenblas', '-fopenmp'],
             runtime_library_dirs=library_dirs)
]

setup(
  name = 'BLaSy',
  packages = ['blasy'],
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)
